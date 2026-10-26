"""Shared helpers for the oracle-gap signal series (scripts/oracle_gap/).

Goal of the series: how much of the oracle-minus-lookup gap on RouterBench is
recoverable from the query text, measured on a ladder of frozen representations
from a lexical floor to a 7B LLM hidden state. Everything here is read-only on
the panel and on the frozen feature files; no outcome ever enters a feature.

Conventions match exp30 so numbers are comparable: the five standard
family-stratified 70/30 splits, RidgeCV on the standardised representation with
the study's fixed alpha grid, AUC of the ridge score, stratified AUC weighted by
stratum size with strata under 20 rows or with one class skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, topic_str  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402

# Tiers fixed on 2026-09-19 by whole-panel solve rate, BEFORE any gap-signal
# experiment was run. Cuts at the natural gaps: 0.856 | 0.647-0.698 | 0.455-0.571 | 0.202-0.351.
TIERS = {
    1: ["gpt-4-1106-preview"],
    2: ["claude-v2", "Yi-34B-Chat", "claude-v1", "gpt-3.5-turbo-1106", "claude-instant-v1"],
    3: ["mixtral-8x7b-chat", "WizardLM-13B-V1.2"],
    4: ["llama-2-70b-chat", "mistral-7b-chat", "code-llama-instruct-34b-chat"],
}

_FEAT = ANALYSIS_ROOT / "features" / PANEL / "prompt"
REP_FILES = {
    "bow":        _FEAT / "hash-bow.npy",
    "gte":        _FEAT / "gte-qwen2-7b.npy",
    "gte_l21":    _FEAT / "hidden" / "gte-qwen2-7b" / "L21_last.npy",
    "reader_l21": _FEAT / "hidden" / "qwen2.5-7b-instruct" / "L21_last.npy",
}
REP_LABEL = {
    "bow": "hashed bag-of-words, 4,096-d (lexical floor)",
    "gte": "emitted GTE-Qwen2-7B, 3,584-d",
    "gte_l21": "GTE-Qwen2-7B layer-21 last token, 3,584-d",
    "reader_l21": "Qwen2.5-7B-Instruct layer-21 last token, 3,584-d",
}
REP_ORDER = ["bow", "gte", "gte_l21", "reader_l21"]
MIN_STRATUM = 20


def load_rep(name: str, panel) -> np.ndarray:
    """Float32 (panel.n, d), aligned to panel rows. Release-indexed files get the keep mask;
    panel-indexed files are taken as is. Decided per file from its own row count."""
    a = np.load(REP_FILES[name], mmap_mode="r")
    if panel.keep is not None and a.shape[0] == len(panel.keep):
        rows = np.flatnonzero(panel.keep)
    else:
        assert a.shape[0] == panel.n, (name, a.shape, panel.n)
        rows = np.arange(panel.n)
    return np.asarray(a[rows], dtype=np.float32)


def model_names(panel):
    names = [str(m).split("/")[-1] for m in panel.models]
    return names, {n: i for i, n in enumerate(names)}


def solved_matrix(panel) -> np.ndarray:
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    return r >= SUCCESS_THRESHOLD


def tier_columns(midx):
    return {t: [midx[m] for m in ms] for t, ms in TIERS.items()}


def fit_scores(xtr, ytr, xte):
    """Ridge on the binary target, standardised on training rows; returns test scores and the chosen alpha."""
    sc = StandardScaler().fit(xtr)
    m = RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(xtr), ytr.astype(np.float64))
    return m.predict(sc.transform(xte)), float(m.alpha_)


def strat_auc(y, s, strata, min_rows=MIN_STRATUM):
    """exp30's convention: AUC inside each stratum, weighted by stratum size; strata under
    min_rows or with a single class are skipped. Returns (pooled, equal-weight, per-stratum)."""
    per, num, den = {}, 0.0, 0.0
    for k in np.unique(strata):
        m = strata == k
        if m.sum() < min_rows or y[m].min() == y[m].max():
            per[int(k)] = None
            continue
        a = float(roc_auc_score(y[m], s[m]))
        per[int(k)] = a
        num += a * m.sum()
        den += m.sum()
    vals = [v for v in per.values() if v is not None]
    return (float(num / den) if den else float("nan"),
            float(np.mean(vals)) if vals else float("nan"), per)


def family_auc(y, s, fam):
    return strat_auc(y, s, fam)


def family_difficulty_auc(y, s, fam, n_solved):
    return strat_auc(y, s, fam.astype(np.int64) * 100 + n_solved.astype(np.int64))


def bootstrap_family_auc(y, s, fam, n=1000, seed=0):
    """Family-stratified resampling of test queries; 95% intervals for pooled and equal-weight AUC."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(fam == g) for g in np.unique(fam)]
    pooled, eqw = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        p, e, _ = family_auc(y[idx], s[idx], fam[idx])
        pooled.append(p)
        eqw.append(e)
    return (np.nanpercentile(pooled, [2.5, 97.5]).tolist(),
            np.nanpercentile(eqw, [2.5, 97.5]).tolist())


def permutation_null(xtr, ytr, famtr, xte, yte, famte, n=50, seed=0):
    """Labels permuted WITHIN family on the training rows, probe refitted, scored on the untouched
    test rows. This is the null that a label carries no text signal beyond the family."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yp = ytr.copy()
        for g in np.unique(famtr):
            rows = np.flatnonzero(famtr == g)
            yp[rows] = ytr[rows][rng.permutation(len(rows))]
        s, _ = fit_scores(xtr, yp, xte)
        out.append(family_auc(yte, s, famte)[0])
    return out


def precision_at(y, s, frac=0.10):
    """Rank test queries by score, take the top `frac`; precision, recall, and the base rate."""
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-s)[:k]
    return float(y[top].mean()), float(y[top].sum() / max(1, y.sum())), float(y.mean())


def out_dir(exp_name: str) -> Path:
    d = RESULTS_ROOT / "oracle_gap" / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(d: Path, result: dict, md_lines: list[str]) -> None:
    (d / "result.json").write_text(json.dumps(result, indent=1, default=float))
    (d / "result.md").write_text("\n".join(md_lines) + "\n")
    print("wrote", d, flush=True)


def base(panel, extra: dict) -> dict:
    return {**panel.provenance(), "tiers": TIERS, "representations": {k: str(REP_FILES[k]) for k in REP_ORDER},
            "seeds": list(SEEDS), "ridge_alphas": [float(a) for a in RIDGE_ALPHAS],
            "min_stratum_rows": MIN_STRATUM, **extra}


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
