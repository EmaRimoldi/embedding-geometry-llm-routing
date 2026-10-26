"""exp35: is per-query difficulty readable from a frozen LLM's hidden states better than from the
contrastive embedder?  Ridge (family one-hot + representation, standardised) predicting the solve
count (number of pool models with r >= 0.5), exp18 protocol: 5 stratified 70/30 splits, R² beyond
the family mean on test rows, per-family values, both aggregations, within-family permutation null
for the best representation, paired bootstrap on split 0.  Representations: GTE (reference) and
Qwen2.5-7B-Instruct hidden states at layers 0 / 7 / 14 / 21 / 28, mean- and last-token pooled
(scripts/extract_llm_hidden.py).  The surprise scalars of the same pass are NOT used here (step 2,
after this experiment is reported).  Writes results[_suffix]/difficulty_llm/exp35_llm_hidden/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp18_recovering_difficulty import boot_r2b, metrics  # noqa: E402

EXP = "exp35_llm_hidden"; OUT = RESULTS_ROOT / "difficulty_llm" / EXP
HID = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "qwen7b_hidden"; LAYERS = (0, 7, 14, 21, 28); N_PERM = 3


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); y = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
    onehot = np.eye(F)[fam]
    reps = {"gte": load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)}
    for L in LAYERS:
        for p in ("mean", "last"):
            arr = np.load(HID / f"L{L}_{p}.npy", mmap_mode="r")
            arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr
            assert arr.shape[0] == n, (L, p, arr.shape); reps[f"qwen7b_L{L}_{p}"] = np.asarray(arr, dtype=np.float32)
    res = {k: [] for k in reps}; per_fam = {k: [] for k in reps}; floors = []
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te)
        floor = np.array([y[tr][fam[tr] == g].mean() for g in range(F)])[fam[te]]
        for k, x in reps.items():
            sc = StandardScaler().fit(x[tr]); xtr = np.column_stack([onehot[tr], sc.transform(x[tr])]); xte = np.column_stack([onehot[te], sc.transform(x[te])])
            yhat = RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, y[tr]).predict(xte)
            m = metrics(y[te], yhat, floor, fam[te], F); res[k].append(m); per_fam[k].append(m["per_family"])
            if seed == SEEDS[0]:
                res[k][-1]["ci_r2b_split0"] = boot_r2b(y[te], yhat, floor, np.random.default_rng(0))
        print(f"seed {seed}: " + ", ".join(f"{k} {res[k][-1]['r2_beyond_family']:+.3f}" for k in reps), flush=True)
    ms = lambda k, key: float(np.mean([d[key] for d in res[k]]))  # noqa: E731
    summary = {k: {"r2_beyond_family": ms(k, "r2_beyond_family"), "r2_beyond_family_equal_weight": float(np.mean([np.mean([v["r2_beyond_family"] for v in d.values() if np.isfinite(v["r2_beyond_family"])]) for d in per_fam[k]])),
                   "r2": ms(k, "r2"), "spearman": ms(k, "spearman"), "mae": ms(k, "mae"), "ci_split0": res[k][0]["ci_r2b_split0"],
                   "per_family": {names[g]: float(np.mean([d[g]["r2_beyond_family"] for d in per_fam[k] if g in d])) for g in range(F)}} for k in reps}
    best = max((k for k in reps if k != "gte"), key=lambda k: summary[k]["r2_beyond_family"])
    # within-family permutation null for the best representation and for GTE (target shuffled within family, split 0)
    null = {}
    for k in ("gte", best):
        tr, te = split(n, fam, SEEDS[0]); tr, te = np.sort(tr), np.sort(te); floor = np.array([y[tr][fam[tr] == g].mean() for g in range(F)])[fam[te]]
        x = reps[k]; sc = StandardScaler().fit(x[tr]); xtr = np.column_stack([onehot[tr], sc.transform(x[tr])]); xte = np.column_stack([onehot[te], sc.transform(x[te])]); vals = []
        for pi in range(N_PERM):
            rng = np.random.default_rng(100 + pi); yp = y.copy()
            for g in range(F):
                rows = np.flatnonzero(fam == g); yp[rows] = y[rows][rng.permutation(len(rows))]
            fl = np.array([yp[tr][fam[tr] == g].mean() for g in range(F)])[fam[te]]
            vals.append(metrics(yp[te], RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, yp[tr]).predict(xte), fl, fam[te], F)["r2_beyond_family"])
        null[k] = {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "n": N_PERM}
    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": EXP, "model": "Qwen2.5-7B-Instruct (frozen)", "layers": list(LAYERS), "target": "n_solved", "summary": summary, "best": best, "null_within_family_shuffle": null}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# {EXP} — difficulty from frozen Qwen2.5-7B-Instruct hidden states ({panel.name})", "",
          "Ridge on family one-hot + representation (standardised), target = number of models that solve the query; R²_beyond_family = 1 − SSE / SSE(family-mean floor) on test rows, mean over 5 splits; equal weight = mean of the per-family values; CI = bootstrap on split 0.", "",
          "representation | R²_beyond_family | equal weight | R² | Spearman | CI (split 0)", "---|---|---|---|---|---"]
    for k in reps:
        s_ = summary[k]; md.append(f"{k} | {s_['r2_beyond_family']:+.3f} | {s_['r2_beyond_family_equal_weight']:+.3f} | {s_['r2']:.3f} | {s_['spearman']:.3f} | [{s_['ci_split0'][0]:+.3f}, {s_['ci_split0'][1]:+.3f}]")
    md += ["", f"Within-family shuffle null (split 0, {N_PERM} draws): " + "; ".join(f"{k}: mean {v['mean']:+.3f}, max {v['max']:+.3f}" for k, v in null.items()), "",
           "## Per family (R²_beyond_family, mean over splits)", "", "representation | " + " | ".join(names), "---|" + "---|" * F]
    for k in reps:
        md.append(f"{k} | " + " | ".join(f"{summary[k]['per_family'][g]:+.3f}" for g in names))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
