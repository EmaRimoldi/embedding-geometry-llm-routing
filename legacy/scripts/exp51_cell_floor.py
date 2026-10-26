"""exp51: is the per-query difficulty signal anything more than the topic?

Every difficulty measurement in this study scores variance explained beyond the FAMILY mean. A
representation could therefore score well merely by recognising the (family, topic) cell and reciting
that cell's average difficulty, which would be topic signal rather than per-query signal. The
inference that currently covers this is indirect: a predictor constant within a cell cannot beat the
cell-mean predictor, and the representations beat it comfortably. This replaces the inference with a
measurement.

Floors, both computed on the training rows only:
  family floor  the family's mean solve count;
  cell floor    the (family, topic) cell's mean, falling back to the family mean below MIN_CELL rows,
                which is the same convention the routing lookups use.
Methods: ridge on the representation alone, and ridge on [cell one-hot, representation]. Scored as
R^2 beyond BOTH floors, so `r2_beyond_cell` is the quantity that cannot be earned by knowing the topic.
Reported per query and with equal weight over families, over the 5 standard splits, with a
within-cell permutation null on split 0 (the target is shuffled inside each cell, which destroys
per-query structure while leaving every cell mean intact; a topic-only predictor scores 0 there).
Writes results[_root]/difficulty_llm/exp51_cell_floor/<rep>/.
Usage: exp51_cell_floor.py --rep computed/qwen7b-l21@prompt"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, topic_str  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

MIN_CELL = 50; N_PERM = 200


def floors(y, tr, te, fam, cell, F):
    """Family-mean and cell-mean floors for the test rows, fitted on the training rows."""
    fam_mean = np.array([y[tr][fam[tr] == g].mean() if (fam[tr] == g).any() else y[tr].mean() for g in range(F)])
    f_floor = fam_mean[fam[te]]
    c_floor = f_floor.copy()
    for c in np.unique(cell):
        m_tr = cell[tr] == c
        if m_tr.sum() >= MIN_CELL:
            c_floor[cell[te] == c] = y[tr][m_tr].mean()
    return f_floor, c_floor


def r2_beyond(y_te, pred, floor):
    sse = float(((y_te - pred) ** 2).sum()); sse0 = float(((y_te - floor) ** 2).sum())
    return 1.0 - sse / max(sse0, 1e-12)


def per_family_r2(y_te, pred, floor, fam_te, F):
    out = {}
    for g in range(F):
        m = fam_te == g
        if m.sum() < 2: continue
        sse0 = float(((y_te[m] - floor[m]) ** 2).sum())
        if sse0 <= 1e-9: continue
        out[g] = 1.0 - float(((y_te[m] - pred[m]) ** 2).sum()) / sse0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--rep", default="computed/gte-qwen2-7b@prompt"); a = ap.parse_args()
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index()
    names = [str(s) for s in panel.datasets]; F = len(names)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    y = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
    _, cell = np.unique(np.array([str(t) for t in topic_str(panel)], dtype=object), return_inverse=True)
    n_cells = int(cell.max()) + 1
    onehot_cell = np.eye(n_cells, dtype=np.float32)[cell]
    x = load_rep(find_rep(PANEL, a.rep), n, panel.keep).astype(np.float32)
    tag = a.rep.split("/")[-1].split("@")[0]
    OUT = RESULTS_ROOT / "difficulty_llm" / "exp51_cell_floor" / tag
    print(f"panel {PANEL} rep {a.rep} ({x.shape[1]} dims) cells={n_cells} families={F}", flush=True)

    acc = {k: {"beyond_family": [], "beyond_cell": [], "beyond_cell_eqw": []} for k in ("ridge:rep", "ridge:cell+rep", "floor:cell")}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te)
        f_floor, c_floor = floors(y, tr, te, fam, cell, F)
        sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
        preds = {"ridge:rep": RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, y[tr]).predict(xte),
                 "ridge:cell+rep": RidgeCV(alphas=RIDGE_ALPHAS).fit(np.column_stack([onehot_cell[tr], xtr]), y[tr]).predict(np.column_stack([onehot_cell[te], xte])),
                 "floor:cell": c_floor}
        for k, p in preds.items():
            acc[k]["beyond_family"].append(r2_beyond(y[te], p, f_floor))
            acc[k]["beyond_cell"].append(r2_beyond(y[te], p, c_floor))
            pf = per_family_r2(y[te], p, c_floor, fam[te], F)
            acc[k]["beyond_cell_eqw"].append(float(np.mean(list(pf.values()))) if pf else float("nan"))
        print(f"seed {seed}: " + ", ".join(f"{k} beyond_cell {acc[k]['beyond_cell'][-1]:+.3f}" for k in preds) + f" ({time.time()-t0:.0f}s)", flush=True)

    # within-cell permutation null on split 0: shuffle the target inside each cell, leaving cell means intact
    tr, te = split(n, fam, SEEDS[0]); tr, te = np.sort(tr), np.sort(te)
    sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
    nulls = []
    for d in range(N_PERM):
        rng = np.random.default_rng(3000 + d); yp = y.copy()
        for c in range(n_cells):
            rows = np.flatnonzero(cell == c)
            if len(rows) > 1: yp[rows] = y[rows][rng.permutation(len(rows))]
        _, c_floor_p = floors(yp, tr, te, fam, cell, F)
        pred = RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, yp[tr]).predict(xte)
        nulls.append(r2_beyond(yp[te], pred, c_floor_p))
    nulls = np.array(nulls)

    ms = lambda v: (float(np.mean(v)), float(np.std(v)))  # noqa: E731
    out = {**panel.provenance(), "experiment": "exp51_cell_floor", "representation": a.rep, "dims": int(x.shape[1]),
           "n_cells": n_cells, "min_cell": MIN_CELL, "n_perm": N_PERM,
           "results": {k: {m: ms(v[m]) for m in v} for k, v in acc.items()},
           "within_cell_null_split0": {"mean": float(nulls.mean()), "p97.5": float(np.percentile(nulls, 97.5)),
                                       "observed_ridge_rep": acc["ridge:rep"]["beyond_cell"][0],
                                       "p_value": float((nulls >= acc["ridge:rep"]["beyond_cell"][0]).mean())},
           "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# exp51 — difficulty beyond the (family, topic) cell, not just beyond the family ({panel.name}, {a.rep})", "",
          f"{n_cells} cells, cell floor falls back to the family mean below {MIN_CELL} training rows. Mean ± sd over 5 splits. "
          f"`beyond_cell` is the quantity that cannot be earned by knowing the topic: a predictor constant within a cell scores 0 there.", "",
          "method | R² beyond family | R² beyond cell | R² beyond cell, equal weight", "---|---|---|---"]
    for k, v in acc.items():
        bf, bfs = ms(v["beyond_family"]); bc, bcs = ms(v["beyond_cell"]); be, bes = ms(v["beyond_cell_eqw"])
        md.append(f"{k} | {bf:+.3f} ± {bfs:.3f} | {bc:+.3f} ± {bcs:.3f} | {be:+.3f} ± {bes:.3f}")
    nz = out["within_cell_null_split0"]
    md += ["", f"Within-cell permutation null (split 0, {N_PERM} draws, target shuffled inside each cell): "
               f"mean {nz['mean']:+.4f}, 97.5th percentile {nz['p97.5']:+.4f}, observed {nz['observed_ridge_rep']:+.4f}, p = {nz['p_value']:.3f}."]
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
