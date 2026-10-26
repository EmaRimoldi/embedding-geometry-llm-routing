"""exp13 - LLMRouterBench baselines at informative, rescaled lambdas (no refits; saved per-lambda curves).

Rescaling: lambda_tilde = lambda * c_ref, c_ref = mean per-query cost of the strongest single model on
the panel (RB: GPT-4 3.25e-3 USD; LRB: gpt-5 61.3e-3 USD). lambda_tilde is the utility penalty, in reward
units, of sending every query to the strongest model; it makes the two panels' trade-offs comparable
(ratio of c_ref about 19). Sources: exp12 (in-domain curves incl. the recomputed cell lookup) and exp11
(held-out curves for every method of the study and of this series). Utility at each grid lambda is
reward - lambda * cost of the saved operating point (query-weighted, and equal weight over families
where the source stores it). Tables: LRB at lambda in {0, .5, 1, 2, 5, 10, 50} with lambda_tilde and the
RB lambda with the same lambda_tilde; RB at its own headline lambdas for the side-by-side.
Writes results/exp13_lrb_rescaled_lambda/{result.json, result.md}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

EXP = "exp13_lrb_rescaled_lambda"
C_REF = {"routerbench-0shot": 0.003252, "llmrouterbench-performance-cost": 0.06134}
LRB_L = ["0", "0.5", "1", "2", "5", "10", "50"]; RB_L = ["0", "10", "50", "100", "500", "1000"]
IN_SHOW = ["oracle", "best_single", "knn100", "knn40", "kmeans64_utility_top3", "cell_lookup (recomputed)", "family_lookup", "routellm_mf_train_selected_threshold"]
OOD_SHOW = ["ours:oracle", "ours:family_ceiling", "study:best_single_train_family_balanced", "ours:knn100", "study:kmeans64_utility_top3", "study:routellm_mf", "ours:soft_prior", "ours:verbosity_gtel", "ours:verbosity_reader", "ours:cost_oracle"]


def util(curve, lams, L):
    i = lams.index(L); c, r = curve[i]; return r - float(L) * c


def load(panel):
    j12 = json.load(open(C.RESULTS / "exp12_indomain_frontier" / panel / "result.json")); j11 = json.load(open(C.RESULTS / "exp11_frontier_reproduction" / panel / "result.json"))
    lam12 = [f"{l:g}" for l in j12["lambdas"]]; lam11 = list(j11["lambdas"])
    return j12, j11, lam12, lam11


def main():
    out = {"c_ref": C_REF, "in_domain": {}, "held_out": {}}; md = [f"# {EXP}", "", "lambda_tilde = lambda x c_ref (c_ref = mean cost of the strongest single model: RB GPT-4 3.25e-3, LRB gpt-5 61.3e-3). Utilities from saved operating points, no refits. qw = query-weighted, [macro] = equal weight over families.", ""]
    for panel, LS in (("llmrouterbench-performance-cost", LRB_L), ("routerbench-0shot", RB_L)):
        j12, j11, lam12, lam11 = load(panel); cref = C_REF[panel]; other = "routerbench-0shot" if panel.startswith("llm") else "llmrouterbench-performance-cost"
        hdr = [f"λ={L} (λ̃={float(L)*cref:.3f}; {other[:3].upper()}-equiv λ≈{float(L)*cref/C_REF[other]:.0f})" for L in LS]
        # in-domain
        md += [f"## {panel}: in-domain (exp12 curves)", "", "method | " + " | ".join(hdr), "---|" + "---|" * len(LS)]; out["in_domain"][panel] = {}
        for m in IN_SHOW:
            if m not in j12["curves"]: continue
            row = {}
            for L in LS:
                uq = util(j12["curves"][m], lam12, L); um = util(j12["curves_macro"][m], lam12, L) if m in j12.get("curves_macro", {}) else float("nan"); row[L] = (uq, um)
            out["in_domain"][panel][m] = row; md.append(f"{m} | " + " | ".join(f"{row[L][0]:.3f} [{row[L][1]:.3f}]" for L in LS))
        # held-out
        md += ["", f"## {panel}: held-out family (exp11 curves)", "", "method | " + " | ".join(hdr), "---|" + "---|" * len(LS)]; out["held_out"][panel] = {}
        for m in OOD_SHOW:
            if m not in j11["frontier_qw"]: continue
            row = {}
            for L in LS:
                uq = util(j11["frontier_qw"][m], lam11, L); um = util(j11["frontier_macro"][m], lam11, L); row[L] = (uq, um)
            out["held_out"][panel][m] = row; md.append(f"{m} | " + " | ".join(f"{row[L][0]:.3f} [{row[L][1]:.3f}]" for L in LS))
        md.append("")
    # matched-pair summary
    md += ["## matched trade-offs (same λ̃): LRB λ=0.5 ↔ RB 10; LRB 5 ↔ RB 100; LRB 10 ↔ RB ≈190; LRB 50 ↔ RB ≈940", ""]
    C.save(C.RESULTS / EXP, out, md)


if __name__ == "__main__":
    main()
