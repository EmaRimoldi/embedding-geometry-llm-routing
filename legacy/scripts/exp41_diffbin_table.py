"""exp41: what the (family, predicted-difficulty bin) lookup actually does.

Split 0 only, descriptive: build the same bins as exp36 (per-family quantiles of the ridge-predicted
solve count from the frozen reader, cross-fitted on training rows), then print, for each cell and a
few lambdas, the model the lookup stores, the cell's size, its mean TRUE solve count (to check the
bins are ordered by real difficulty) and the cell's mean cost of the chosen model.  Also the family
lookup's choice, so one can see where the bins change the decision.  No new claim; it explains the
gain reported in exp36.  Writes results[_root]/difficulty_llm/exp41_diffbin_table/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from exp36_augmented_knn import crossfit  # noqa: E402

OUT = RESULTS_ROOT / "difficulty_llm" / "exp41_diffbin_table"; NBIN = 5; MIN_CELL = 50
LAMS = (0.0, 10.0, 100.0, 1000.0)


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names)
    models = [str(m) for m in panel.models]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    y = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64); onehot = np.eye(F, dtype=np.float32)[fam]
    h = np.load(ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / "qwen2.5-7b-instruct" / "L21_last.npy", mmap_mode="r")
    h = h[panel.keep] if panel.keep is not None and h.shape[0] == len(panel.keep) else h; h = np.asarray(h, dtype=np.float32)
    seed = SEEDS[0]; tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
    dtr, dte = crossfit(h, y, onehot, tr, te, fam, seed)
    dtr = np.asarray(dtr).ravel(); dte = np.asarray(dte).ravel()
    btr = np.zeros(len(tr), int); bte = np.zeros(len(te), int)
    for g in range(F):
        q = np.quantile(dtr[ftr == g], np.linspace(0, 1, NBIN + 1)[1:-1])
        btr[ftr == g] = np.searchsorted(q, dtr[ftr == g]); bte[fte == g] = np.searchsorted(q, dte[fte == g])
    rows = []
    for lam in LAMS:
        U = r - lam * c; Utr = U[tr]
        fam_best = {g: int(Utr[ftr == g].mean(0).argmax()) for g in range(F)}
        for g in range(F):
            for b in range(NBIN):
                m = (ftr == g) & (btr == b)
                if m.sum() == 0: continue
                pick = int(Utr[m].mean(0).argmax()) if m.sum() >= MIN_CELL else fam_best[g]
                rows.append({"lambda": lam, "family": names[g], "bin": b, "train_rows": int(m.sum()), "test_rows": int(((fte == g) & (bte == b)).sum()),
                             "mean_predicted_difficulty": float(dtr[m].mean()), "mean_true_solved": float(y[tr][m].mean()),
                             "cell_model": models[pick], "family_model": models[fam_best[g]], "differs": models[pick] != models[fam_best[g]],
                             "mean_cost_cell_model": float(c[tr][m, pick].mean()), "mean_cost_family_model": float(c[tr][m, fam_best[g]].mean())})
    # ordering check: does the predicted bin track the true solve count?
    order_ok = {names[g]: [float(y[tr][(ftr == g) & (btr == b)].mean()) for b in range(NBIN) if ((ftr == g) & (btr == b)).sum()] for g in range(F)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp41_diffbin_table", "split": int(seed), "n_bins": NBIN, "min_cell": MIN_CELL, "rows": rows, "true_solved_by_bin": order_ok}, indent=1))
    md = [f"# exp41 — what the (family, predicted-difficulty bin) lookup stores ({panel.name}, split 0)", "",
          f"Bins = per-family quintiles of the ridge-predicted solve count (frozen Qwen2.5-7B L21 last token, cross-fitted on training rows). Cells under {MIN_CELL} training rows fall back to the family.", "",
          "## Do the predicted bins track real difficulty? (mean true solve count per bin, training rows)", "", "family | " + " | ".join(f"bin {b}" for b in range(NBIN)), "---|" + "---|" * NBIN]
    for g, v in order_ok.items():
        md.append(f"{g} | " + " | ".join(f"{x:.2f}" for x in v))
    for lam in LAMS:
        md += ["", f"## λ = {lam:g}: model stored per cell (bold where it differs from the family lookup)", "",
               "family | family lookup picks | " + " | ".join(f"bin {b}" for b in range(NBIN)), "---|---|" + "---|" * NBIN]
        for g in names:
            sel = [x for x in rows if x["lambda"] == lam and x["family"] == g]
            if not sel: continue
            fm = sel[0]["family_model"]
            cells = []
            for b in range(NBIN):
                e = next((x for x in sel if x["bin"] == b), None)
                cells.append("—" if e is None else (f"**{e['cell_model']}**" if e["differs"] else e["cell_model"]))
            md.append(f"{g} | {fm} | " + " | ".join(cells))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
