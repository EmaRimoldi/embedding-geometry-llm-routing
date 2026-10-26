"""Aggregate probe / geometry results into tables, bootstrap CIs and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ANALYSIS_ROOT, Targets, build_targets, load_panel
from .probes import make_folds
from .reps import list_reps, safe_name

PROBES = ANALYSIS_ROOT / "results/probes"
GEOMETRY = ANALYSIS_ROOT / "results/geometry"

FAMILY_OF = {"onehot": "control", "controls": "control"}


def rep_family(name: str) -> str:
    if name in FAMILY_OF:
        return FAMILY_OF[name]
    base = name.split("/", 1)[-1]
    return "hidden" if base.startswith("hidden") else "contrastive"


def load_probe_table() -> pd.DataFrame:
    rows = []
    for path in PROBES.glob("*/*/*.json"):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "error" in d:
            rows.append({"panel": path.parts[-3], "rep_dir": path.parts[-2], "unit": path.stem, "error": d["error"]})
            continue
        base = {"panel": d["panel"], "rep": d["rep"], "family": rep_family(d["rep"]),
                "text_field": d["rep"].split("@")[-1] if "@" in d["rep"] else "origin_query",
                "transform": d["transform"], "decoder": d["decoder"], "split": d["split"],
                "wall_s": d.get("wall_s", np.nan), "unit": path.stem, "rep_dir": path.parts[-2]}
        if "frontier" in d:
            base["frontier_router_area"] = d["frontier"]["router_area"]
            base["frontier_oracle_area"] = d["frontier"]["oracle_area"]
        for subset, metrics in d["subsets"].items():
            rows.append({**base, "subset": subset, **metrics})
    df = pd.DataFrame(rows)
    return df


def load_geometry_table() -> pd.DataFrame:
    rows = []
    for path in GEOMETRY.glob("*/*.json"):
        if path.name == "cka.json":
            continue
        d = json.loads(path.read_text())
        nb = d["neighbours"]
        row = {"panel": path.parts[-2], "rep": d["rep"], "family": d["family"], "source": d["source"],
               "text_field": d["text_field"], "normalized": d["normalized"],
               **{f"spec_{k}": v for k, v in d["spectrum"].items()},
               "eta2_total": d["dataset_variance"]["eta2_total"],
               "eta2_pc1": d["dataset_variance"]["eta2_per_pc_top20"][0],
               "dataset_probe_acc": d["dataset_identifiability"]["linear_probe_accuracy"],
               "dataset_majority": d["dataset_identifiability"]["majority_baseline"],
               "nmi_k64": d["cluster_alignment"]["64"]["nmi_with_dataset"],
               "purity_k64": d["cluster_alignment"]["64"]["purity_wrt_dataset"],
               "silhouette_dataset": d["cluster_alignment"]["silhouette_by_dataset"],
               "pair_within_spearman_agree": d["pairs"]["within"]["spearman_dist_agreement"],
               "pair_across_spearman_agree": d["pairs"]["across"]["spearman_dist_agreement"],
               "pair_within_spearman_absdiff": d["pairs"]["within"]["spearman_dist_absdiffdiff"],
               "knn20_agree": nb["20"]["agreement_knn"], "knn20_agree_random": nb["20"]["agreement_random_same_dataset"],
               "knn20_gain": nb["20"]["agreement_knn"] - nb["20"]["agreement_random_same_dataset"],
               "knn20_gain_disagreement": nb["20"]["agreement_knn_disagreement_subset"] - nb["20"]["agreement_random_disagreement_subset"],
               "knn20_difficulty_spearman": nb["20"]["difficulty_spearman_knn"],
               "knn20_difficulty_spearman_random": nb["20"]["difficulty_spearman_random_same_dataset"],
               "knn20_same_dataset_rate": nb["same_dataset_rate_by_k"]["20"]}
        if "norm" in d:
            row.update({f"norm_{k}": v for k, v in d["norm"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_cka(panel: str) -> pd.DataFrame | None:
    path = GEOMETRY / panel / "cka.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    names = sorted({n for key in d for n in key.split(" || ")})
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for key, v in d.items():
        a, b = key.split(" || ")
        mat.loc[a, b] = mat.loc[b, a] = v
    return mat


# ----------------------------------------------------------------------------
# Baselines per query (deterministic from the split definition) and bootstrap
# ----------------------------------------------------------------------------

def baseline_rewards(targets: Targets, split: str, seed: int = 0) -> dict[str, np.ndarray]:
    n = len(targets.difficulty)
    reward = np.nan_to_num(targets.reward, nan=0.0)
    bs = np.zeros(n)
    db = np.zeros(n)
    for tr, te in make_folds(split, targets, seed):
        train_mean = np.nanmean(targets.reward[tr], axis=0)
        j = int(np.nanargmax(train_mean))
        bs[te] = reward[te, j]
        for g in np.unique(targets.dataset_index[te]):
            rows_tr = tr[targets.dataset_index[tr] == g]
            rows_te = te[targets.dataset_index[te] == g]
            jj = int(np.nanargmax(np.nanmean(targets.reward[rows_tr], axis=0))) if len(rows_tr) else j
            db[rows_te] = reward[rows_te, jj]
    return {"best_single": bs, "domain_best": db, "oracle": targets.oracle_reward}


def per_query(panel: str, rep: str, unit: str) -> dict | None:
    path = PROBES / panel / safe_name(rep) / f"{unit}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def bootstrap_gap_closed(router: np.ndarray, base: dict[str, np.ndarray], mask: np.ndarray,
                         other: np.ndarray | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the gap-closed statistic (vs domain best) and, optionally, its
    paired difference against another router on the same queries."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    r, d, o = router[idx], base["domain_best"][idx], base["oracle"][idx]
    r2 = other[idx] if other is not None else None
    stats, diffs = [], []
    for _ in range(n_boot):
        b = rng.integers(0, len(idx), len(idx))
        gap = o[b].mean() - d[b].mean()
        g1 = (r[b].mean() - d[b].mean()) / max(gap, 1e-9)
        stats.append(g1)
        if r2 is not None:
            diffs.append(g1 - (r2[b].mean() - d[b].mean()) / max(gap, 1e-9))
    out = {"point": float((r.mean() - d.mean()) / max(o.mean() - d.mean(), 1e-9)),
           "ci_lo": float(np.percentile(stats, 2.5)), "ci_hi": float(np.percentile(stats, 97.5))}
    if diffs:
        out.update({"diff_point": float(np.mean(diffs)), "diff_ci_lo": float(np.percentile(diffs, 2.5)),
                    "diff_ci_hi": float(np.percentile(diffs, 97.5))})
    return out


def window_masks(panel: str, targets: Targets, field: str = "origin_query") -> dict[str, np.ndarray]:
    """Evaluation masks that remove the context-window confound."""
    from .data import load_panel
    pnl = load_panel(panel)
    lengths_path = ANALYSIS_ROOT / "features" / panel / (field if panel != "routerbench-0shot" else "prompt") / "lengths.npz"
    masks = {"all": np.ones(pnl.n, dtype=bool), "no_swebench": pnl.dataset_ids != "swe-bench"}
    if lengths_path.exists():
        z = np.load(lengths_path)
        masks["fits256"] = z["bert_tokens"] <= 256
        masks["fits512"] = z["bert_tokens"] <= 512
    return masks


def gap_closed_on(router: np.ndarray, covered: np.ndarray, base: dict[str, np.ndarray], mask: np.ndarray) -> float:
    mk = mask & covered
    gap = base["oracle"][mk].mean() - base["domain_best"][mk].mean()
    return float((router[mk].mean() - base["domain_best"][mk].mean()) / max(gap, 1e-9))


def window_controlled_table(panel: str, reps: list[str], units: list[str], targets: Targets) -> pd.DataFrame:
    """G (disagreement subset) for each rep/unit on all / no_swebench / fits256 / fits512."""
    masks = window_masks(panel, targets)
    rows = []
    for unit in units:
        split = unit.split("__")[-1]
        base = baseline_rewards(targets, split)
        for rep in reps:
            q = per_query(panel, rep, unit)
            if q is None:
                continue
            row = {"rep": rep, "unit": unit}
            for name, m in masks.items():
                row[f"G_{name}"] = gap_closed_on(q["router_reward"], q["covered"], base, m & targets.disagreement)
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    x = a[idx] - b[idx]
    boots = [x[rng.integers(0, len(idx), len(idx))].mean() for _ in range(n_boot)]
    return {"point": float(x.mean()), "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))}
