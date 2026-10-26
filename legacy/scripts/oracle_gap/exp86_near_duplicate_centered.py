"""exp86 — exp83 rerun with mean-centred cosine.

exp83 searched neighbours on raw unit vectors. The reader L21 hidden state is strongly anisotropic
(37,914 of 52,000 test rows had a nearest neighbour at cosine >= 0.95), so its bands carried no
information, and emitted GTE has a common component too. Here every representation is centred on the
TRAINING mean of the split before unit-normalising, per split, so cosine measures direction relative
to the panel mean. Everything else is identical to exp83 (see its docstring). Under GTE the raw >=0.95
band was 424 MMLU + 397 Winogrande rows (Winogrande twin sentences by construction).
PROVES the same thing as exp83 with bands that are meaningful for both representations.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

EXP = "exp86_near_duplicate_centered"
BANDS = [(0.95, 1.01), (0.90, 0.95), (0.80, 0.90), (0.70, 0.80), (-1.0, 0.70)]
BAND_NAMES = [">=0.95", "[0.90,0.95)", "[0.80,0.90)", "[0.70,0.80)", "<0.70"]
SEARCH_REPS = ("gte", "reader_l21")


def unit(a): return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def nearest(xq, xt, chunk=1000):
    """index into xt and cosine of the nearest training row for each query row (unit vectors)."""
    idx = np.empty(len(xq), dtype=np.int64); cos = np.empty(len(xq), dtype=np.float32)
    for s in range(0, len(xq), chunk):
        sim = xq[s:s+chunk] @ xt.T
        j = sim.argmax(1); idx[s:s+chunk] = j; cos[s:s+chunk] = sim[np.arange(len(j)), j]
    return idx, cos


def main() -> None:
    t0 = time.time()
    panel = C.load_panel(C.PANEL); n = panel.n; fam = panel.dataset_index()
    fnames = [str(s) for s in panel.datasets]; F = len(fnames)
    names, midx = C.model_names(panel); S = C.solved_matrix(panel); M = len(names); tc = C.tier_columns(midx)
    cell_str = np.array([str(t) for t in C.topic_str(panel)], dtype=object); _, cell = np.unique(cell_str, return_inverse=True)
    g4 = S[:, tc[1][0]]; t2 = S[:, tc[2]].any(1); labelA = (~g4 & t2).astype(np.int8); validA = g4 | t2
    # tier of the oracle's model at lambda=0 (reward only; ties by lowest tier = cheapest capability rung)
    tier_of = np.zeros(M, dtype=np.int64)
    for t, cols in tc.items():
        for c in cols: tier_of[c] = t
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    best_r = r.max(1); oracle_tier = np.array([tier_of[np.flatnonzero(r[i] == best_r[i])].max() if best_r[i] > 0 else 0 for i in range(n)])

    outcomes = {names[m]: S[:, m].astype(np.int8) for m in range(M)}
    outcomes["labelA:gpt4_fails_tier2_solves"] = labelA
    result = C.base(panel, {"experiment": EXP, "bands": BAND_NAMES, "search_representations": list(SEARCH_REPS), "results": {}})

    for srep in SEARCH_REPS:
        Xraw = C.load_rep(srep, panel)
        acc = {k: {b: [0, 0] for b in BAND_NAMES} for k in outcomes}          # agree, total per band
        acc_tier = {b: [0, 0] for b in BAND_NAMES}
        ref_fam = {k: {b: [0.0, 0] for b in BAND_NAMES} for k in outcomes}    # exact same-family reference, summed
        ref_cell = {k: {b: [0.0, 0] for b in BAND_NAMES} for k in outcomes}
        band_n = {b: 0 for b in BAND_NAMES}; band_by_fam = {b: np.zeros(F, dtype=np.int64) for b in BAND_NAMES}
        for seed in C.SEEDS:
            tr, te = C.split(n, fam, seed); X = unit(Xraw - Xraw[tr].mean(0, keepdims=True))
            j, cos = nearest(X[te], X[tr]); nn = tr[j]
            # exact references from the training rows of this split
            p_fam = {k: np.array([outcomes[k][tr][fam[tr] == g].mean() if (fam[tr] == g).any() else 0.5 for g in range(F)]) for k in outcomes}
            p_cell = {}
            for k in outcomes:
                pc = np.full(cell.max() + 1, np.nan)
                for c in np.unique(cell[tr]): pc[c] = outcomes[k][tr][cell[tr] == c].mean()
                p_cell[k] = pc
            for bname, (lo, hi) in zip(BAND_NAMES, BANDS):
                m = (cos >= lo) & (cos < hi); band_n[bname] += int(m.sum())
                for g in range(F): band_by_fam[bname][g] += int((m & (fam[te] == g)).sum())
                q = te[m]; nq = nn[m]
                if len(q) == 0: continue
                for k in outcomes:
                    o = outcomes[k]
                    if k.startswith("labelA"):
                        v = validA[q]; q2 = q[v]; nq2 = nq[v]
                        if len(q2) == 0: continue
                        acc[k][bname][0] += int((o[q2] == o[nq2]).sum()); acc[k][bname][1] += len(q2)
                        pf = p_fam[k][fam[q2]]; ref_fam[k][bname][0] += float((pf**2 + (1-pf)**2).sum()); ref_fam[k][bname][1] += len(q2)
                        pc = p_cell[k][cell[q2]]; ok = ~np.isnan(pc); ref_cell[k][bname][0] += float((pc[ok]**2 + (1-pc[ok])**2).sum()); ref_cell[k][bname][1] += int(ok.sum())
                    else:
                        acc[k][bname][0] += int((o[q] == o[nq]).sum()); acc[k][bname][1] += len(q)
                        pf = p_fam[k][fam[q]]; ref_fam[k][bname][0] += float((pf**2 + (1-pf)**2).sum()); ref_fam[k][bname][1] += len(q)
                        pc = p_cell[k][cell[q]]; ok = ~np.isnan(pc); ref_cell[k][bname][0] += float((pc[ok]**2 + (1-pc[ok])**2).sum()); ref_cell[k][bname][1] += int(ok.sum())
                acc_tier[bname][0] += int((oracle_tier[q] == oracle_tier[nq]).sum()); acc_tier[bname][1] += len(q)
            print(f"{srep} seed {seed}: bands " + ", ".join(f"{b} {int(((cos>=lo)&(cos<hi)).sum())}" for b,(lo,hi) in zip(BAND_NAMES,BANDS)) + f"  ({time.time()-t0:.0f}s)", flush=True)

        R = {"band_rows_over_5_splits": band_n, "band_rows_by_family": {b: {fnames[g]: int(v[g]) for g in range(F)} for b, v in band_by_fam.items()},
             "per_model": {}, "oracle_tier_lam0": {}}
        for k in outcomes:
            R["per_model"][k] = {b: {"n": acc[k][b][1], "p_same_nearest": (acc[k][b][0] / acc[k][b][1]) if acc[k][b][1] else None,
                                     "p_same_random_family": (ref_fam[k][b][0] / ref_fam[k][b][1]) if ref_fam[k][b][1] else None,
                                     "p_same_random_cell": (ref_cell[k][b][0] / ref_cell[k][b][1]) if ref_cell[k][b][1] else None} for b in BAND_NAMES}
        for b in BAND_NAMES:
            R["oracle_tier_lam0"][b] = {"n": acc_tier[b][1], "p_same_tier_nearest": (acc_tier[b][0] / acc_tier[b][1]) if acc_tier[b][1] else None}
        result["results"][srep] = R
        del X

    md = [f"# {EXP} — near-duplicate outcome agreement, all models, no classifier ({panel.name})", "",
          "For each test query (5 splits pooled) the nearest TRAINING query by cosine. 'nearest' = P(same outcome as that neighbour); 'random family' / 'random cell' = the EXACT expected agreement with a random same-family / same-(family,topic) training query, p^2+(1-p)^2. The gain of 'nearest' over the random references is all a nearest-neighbour router could ever exploit.", ""]
    for srep in SEARCH_REPS:
        R = result["results"][srep]
        md += [f"## Neighbour search on {C.REP_LABEL[srep]}", "", "rows per band over 5 splits: " + ", ".join(f"{b}: {R['band_rows_over_5_splits'][b]}" for b in BAND_NAMES), "",
               "model | " + " | ".join(f"{b}: nearest / rand-family / rand-cell (n)" for b in BAND_NAMES), "---|" + "---|" * len(BAND_NAMES)]
        for k in outcomes:
            cells = []
            for b in BAND_NAMES:
                v = R["per_model"][k][b]
                cells.append(f"{C.f3(v['p_same_nearest'])} / {C.f3(v['p_same_random_family'])} / {C.f3(v['p_same_random_cell'])} ({v['n']})")
            md.append(f"{k} | " + " | ".join(cells))
        md += ["", "oracle's tier at λ=0, P(same tier as nearest): " + ", ".join(f"{b} {C.f3(R['oracle_tier_lam0'][b]['p_same_tier_nearest'])} (n={R['oracle_tier_lam0'][b]['n']})" for b in BAND_NAMES), ""]
    result["seconds"] = time.time() - t0
    C.save(C.out_dir(EXP), result, md)


if __name__ == "__main__":
    main()
