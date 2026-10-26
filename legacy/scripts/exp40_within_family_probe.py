"""exp40: difficulty read-out family by family, with the probe fitted WITHIN the family.
For each family and each of the 5 splits (the global stratified splits restricted to the family):
  ridge (RidgeCV) on the standardised representation, fitted on the family's training rows only,
  R² on the family's test rows against the family's training mean (= R²_beyond_family within the
  family), Spearman; a logistic probe for "solved by at least the family's median number of models"
  (AUC); the same with log token count only (length baseline).  Representations: GTE, Qwen2.5-7B-
  Instruct layer 21 last token, layer 14 last token.  Mean over splits.  Compared with the global
  probe's per-family values (exp35).  Writes results[_root]/difficulty_llm/exp40_within_family/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, lengths_features  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

OUT = RESULTS_ROOT / "difficulty_llm" / "exp40_within_family"


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); y = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
    HID = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / "qwen2.5-7b-instruct"
    def load(p):
        a = np.load(p, mmap_mode="r"); a = a[panel.keep] if panel.keep is not None and a.shape[0] == len(panel.keep) else a; return np.asarray(a, dtype=np.float32)
    L, _ = lengths_features(panel)
    reps = {"length_only": L.astype(np.float32), "gte": load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32), "qwen_L14_last": load(HID / "L14_last.npy"), "qwen_L21_last": load(HID / "L21_last.npy")}
    glob = json.load(open(RESULTS_ROOT / "difficulty_llm" / "exp35_llm_hidden" / "result.json"))["summary"]
    res = {}
    for g in range(F):
        rows = np.flatnonzero(fam == g); med = np.median(y[rows]); res[names[g]] = {"rows": int(len(rows)), "median_solved": float(med)}
        for k, x in reps.items():
            r2s, rhos, aucs = [], [], []
            for seed in SEEDS:
                tr, te = split(n, fam, seed); tr = np.intersect1d(tr, rows); te = np.intersect1d(te, rows)
                sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
                yhat = RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, y[tr]).predict(xte); floor = y[tr].mean()
                r2s.append(1 - ((y[te] - yhat) ** 2).sum() / max(((y[te] - floor) ** 2).sum(), 1e-9))
                rhos.append(spearmanr(y[te], yhat).correlation if np.std(yhat) > 0 else 0.0)
                b = (y >= med).astype(int)
                if 0 < b[tr].mean() < 1 and 0 < b[te].mean() < 1:
                    p = LogisticRegression(C=0.1, max_iter=2000).fit(xtr, b[tr]).predict_proba(xte)[:, 1]; aucs.append(roc_auc_score(b[te], p))
            res[names[g]][k] = {"r2_within": float(np.mean(r2s)), "r2_within_sd": float(np.std(r2s)), "spearman": float(np.mean(rhos)), "auc_ge_median": float(np.mean(aucs)) if aucs else None,
                                "r2_global_probe": glob.get({"gte": "gte", "qwen_L14_last": "qwen7b_L14_last", "qwen_L21_last": "qwen7b_L21_last"}.get(k, ""), {}).get("per_family", {}).get(names[g])}
        print(names[g], {k: round(v["r2_within"], 3) for k, v in res[names[g]].items() if isinstance(v, dict)}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp40_within_family", "per_family": res}, indent=1))
    md = [f"# exp40 — difficulty read-out with the probe fitted within each family ({panel.name})", "",
          "Ridge on the standardised representation, fitted on the family's training rows only; R² on the family's test rows vs the family's training mean; AUC of a logistic probe for 'solved by ≥ the family's median'; mean over 5 splits. 'global' = the per-family value of the single cross-family probe (exp35).", "",
          "family | rows | median solved | " + " | ".join(f"{k}: R² within (global) / ρ / AUC" for k in reps), "---|---|---|" + "---|" * len(reps)]
    for g in names:
        d = res[g]; cells = []
        for k in reps:
            v = d[k]; gl = f" ({v['r2_global_probe']:+.2f})" if v["r2_global_probe"] is not None else ""; au = f"{v['auc_ge_median']:.2f}" if v["auc_ge_median"] is not None else "—"
            cells.append(f"{v['r2_within']:+.3f}{gl} / {v['spearman']:.2f} / {au}")
        md.append(f"{g} | {d['rows']} | {d['median_solved']:.0f} | " + " | ".join(cells))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
