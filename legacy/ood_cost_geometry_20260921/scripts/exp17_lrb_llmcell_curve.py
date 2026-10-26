"""exp17 - cost/reward curve of the (family, topic) cell lookup with a topic label on every family (LLM labels)
on LLMRouterBench, at the saved-curve lambda grid restricted to the reported range (lambda <= 10), for the paper's
frontier figures. Same protocol as exp14 (five splits, MIN_CELL 50, family fallback). Per query and equal weight
over families; mean over splits of (mean cost, mean reward). Also the released-cell lookup for reference.
Writes results/exp17_lrb_llmcell_curve/. Nothing outside this folder is modified."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from rb.probes.sweep import split
from exp14_lrb_topic_labels import labels, lookup

EXP = "exp17_lrb_llmcell_curve"; PANEL = "llmrouterbench-performance-cost"
LAMS = (0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0); SEEDS = (0, 1, 2, 3, 4)


def main():
    D = C.load(PANEL); r, c, fam, fn = D["r"], D["c"], D["fam"], D["fnames"]; F, n = len(fn), len(r)
    L = labels(D); cells = {k: np.unique(v, return_inverse=True)[1] for k, v in L.items() if k in ("llm", "released")}
    curves = {k: {"per_query": [], "per_family_equal_weight": []} for k in cells}; util = {k: {} for k in cells}
    for lam in LAMS:
        U = r - lam * c; acc = {k: {"q": [], "f": [], "u": []} for k in cells}
        for seed in SEEDS:
            tr, te = split(n, fam, seed)
            for k, cl in cells.items():
                ch, _ = lookup(U, tr, te, fam, cl, F); cc, rr = c[te, ch], r[te, ch]
                acc[k]["q"].append([cc.mean(), rr.mean()]); acc[k]["f"].append([np.mean([cc[fam[te] == g].mean() for g in range(F)]), np.mean([rr[fam[te] == g].mean() for g in range(F)])]); acc[k]["u"].append(float((rr - lam * cc).mean()))
        for k in cells:
            curves[k]["per_query"].append(np.mean(acc[k]["q"], 0).tolist()); curves[k]["per_family_equal_weight"].append(np.mean(acc[k]["f"], 0).tolist()); util[k][f"{lam:g}"] = float(np.mean(acc[k]["u"]))
        print("lambda", lam, "done", flush=True)
    res = C.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMS), "curves": {f"cell_lookup_{k}": v["per_query"] for k, v in curves.items()}, "curves_macro": {f"cell_lookup_{k}": v["per_family_equal_weight"] for k, v in curves.items()}, "utility_per_query": util})
    md = [f"# {EXP} ({PANEL})", "", "Mean over 5 splits of (mean cost, mean reward) of the cell lookup at each lambda; per query and equal weight over families.", "", "label set | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
    for k in cells: md.append(f"{k} (cost e-3, reward) | " + " | ".join(f"({p[0]*1e3:.2f}, {p[1]:.3f})" for p in curves[k]["per_query"]))
    md += ["", "utility per query (check against exp14): " + "; ".join(f"{k}: " + ", ".join(f"{l}={u:.3f}" for l, u in util[k].items() if l in ('0', '0.5', '1', '2', '5', '10')) for k in cells)]
    C.save(C.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
