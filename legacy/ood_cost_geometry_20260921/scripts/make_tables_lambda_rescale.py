"""Generate reports/tab_headline.tex and reports/tab_baselines_full.tex at the rescaled lambda grids from the saved
five-split results (read-only on results; writes only the two .tex files in reports/, which were backed up as
*.pre-lambda-20260921.tex). RB grid {0,1,10,50,100,500}; LRB grid {0,.5,1,2,5,10}. Per-query and equal-weight blocks.
Sources: exp08 basic tests (kNN, family lookup, classify lookup, MLPs, references), exp64 baselines (K-means, Avengers
upper bound, RouteLLM-MF, GraphRouter), exp20 (true/predicted released-cell lookup), exp33/exp14 (LLM-cell lookup on LRB),
exp33/exp16 (predicted LLM-cell lookup on LRB)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C

RB = C.E4 / "results"; LRB = C.E4 / "llmrouterbench" / "results" / "performance-cost"; REP = C.E4 / "reports"
G = {"RB": [0.0, 1.0, 10.0, 50.0, 100.0, 500.0], "LRB": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]}
AGG = {"qw": "per_query", "eq": "per_family_equal_weight"}
J = lambda p: json.load(open(p))


def bl(j, l, a):
    d = j["by_lambda"]; k = next((k for k in (str(float(l)), f"{l:g}") if k in d), None)
    if k is None: return None
    v = d[k]["all"][AGG[a]]["utility"]; return float(v[0] if isinstance(v, list) else v)


def e20(root, l, m, a):
    e = J(root / "outcome_locality" / "exp20_topic_recovery" / "result.json")["per_seed"]; return float(np.mean([e[s]["routing"][f"{l:g}|lookup:{m}|{AGG[a]}"] for s in e]))


def build(panel):
    root = RB if panel == "RB" else LRB; b64 = (RB / "routerbench_baselines_exp64" / "baselines") if panel == "RB" else (LRB / "baselines"); g = G[panel]; R = {}
    ref = J(root / "basic_model_tests" / "references.json")["references"]
    R["oracle"] = {a: [bl({"by_lambda": ref["oracle"]}, l, a) for l in g] for a in AGG}; R["best single model"] = {a: [bl({"by_lambda": ref["best_single"]}, l, a) for l in g] for a in AGG}
    def add(lab, p):
        if p.exists(): j = J(p); R[lab] = {a: [bl(j, l, a) for l in g] for a in AGG}
    add("family lookup", root / "basic_model_tests" / "best_model_per_family" / "best_model_per_family.json")
    for k in (100, 40, 10, 5): add(f"kNN-{k}", root / "basic_model_tests" / "knn" / f"knn{k}.json")
    add("MLP (100)", root / "basic_model_tests" / "mlp" / "mlp100.json"); add("MLP (100,100)", root / "basic_model_tests" / "mlp" / "mlp100x100.json")
    add("regularized MLP (100)", root / "basic_model_tests" / "mlp_regularised" / "mlp100.json"); add("regularized MLP (100,100)", root / "basic_model_tests" / "mlp_regularised" / "mlp100x100.json")
    add("$K$-means utility, $K=64$, top 1", b64 / "kmeans64_utility_top1" / "result.json"); add("$K$-means utility, $K=64$, top 3", b64 / "kmeans64_utility_top3" / "result.json")
    add("RouteLLM-MF, two models", b64 / "routellm_mf_train_selected_threshold" / "result.json")
    for sc, lab in (("performance_first", "performance first"), ("balance", "balance"), ("cost_first", "cost first")): add(("GraphRouter transfer, " if panel == "RB" else "GraphRouter, ") + lab, b64 / f"graphrouter_{sc}" / "result.json")
    for t in (1, 3):
        p = b64 / f"avengerspro_k64_top{t}" / "result.json"
        if p.exists():
            a_ = J(p)["by_alpha"]; R[f"Avengers-Pro, $K=64$, top {t}, upper bound"] = {"qw": [max(float(r["utility_by_lambda_per_query"][str(float(l))]) for r in a_) for l in g], "eq": [max(float(r["utility_by_lambda_equal_weight"][str(float(l))]) for r in a_) if "utility_by_lambda_equal_weight" in a_[0] else None for l in g]}
    R["family/topic lookup" + (", released labels" if panel == "LRB" else "")] = {a: [e20(root, l, "true_cell", a) for l in g] for a in AGG}
    R["predicted-cell lookup" + (", released labels" if panel == "LRB" else "")] = {a: [e20(root, l, "predicted_cell", a) for l in g] for a in AGG}
    if panel == "LRB":
        e14 = J(C.RESULTS / "exp14_lrb_topic_labels" / "result.json")["results"]
        R["family/topic lookup, all topics"] = {"qw": [e14["lookup:llm"][f"{l:g}"]["qw"][0] for l in g], "eq": [e14["lookup:llm"][f"{l:g}"]["macro"][0] for l in g]}
        p16 = C.RESULTS / "exp16_lrb_predicted_cell" / "result.json"
        if p16.exists():
            e16 = J(p16)["results"]; R["predicted-cell lookup, all topics"] = {"qw": [e16["predicted_cell"][f"{l:g}"]["qw"] for l in g], "eq": [e16["predicted_cell"][f"{l:g}"]["macro"] for l in g]}
        else:
            e33 = J(LRB / "outcome_locality" / "exp33_llm_topics" / "result.json")["A_utility"]; R["predicted-cell lookup, all topics"] = {a: [float(e33[f"{l:g}"][f"llm_predicted|{AGG[a]}"]) if f"{l:g}" in e33 else None for l in g] for a in AGG}
    return R


def fmt(x): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"${x:.3f}$" if x < 0 else f"{x:.3f}")


def block(R, order, a, g):
    lines = []
    for lab in order:
        if lab in R: lines.append(f"{lab} & " + " & ".join(fmt(v) for v in R[lab][a]) + r" \\")
    return lines


HEAD_RB = ["oracle", "family/topic lookup", "family lookup", "kNN-100", "kNN-40", "$K$-means utility, $K=64$, top 1", "best single model"]
HEAD_LRB = ["oracle", "family/topic lookup, all topics", "family lookup", "kNN-100", "$K$-means utility, $K=64$, top 3", "kNN-40", "kNN-10"]
FULL_RB = ["oracle", "family/topic lookup", "predicted-cell lookup", "family lookup", "kNN-100", "kNN-40", "Avengers-Pro, $K=64$, top 1, upper bound", "Avengers-Pro, $K=64$, top 3, upper bound", "$K$-means utility, $K=64$, top 1", "$K$-means utility, $K=64$, top 3", "best single model", "regularized MLP (100,100)", "kNN-10", "regularized MLP (100)", "kNN-5", "MLP (100,100)", "MLP (100)", "GraphRouter transfer, balance", "RouteLLM-MF, two models", "GraphRouter transfer, cost first", "GraphRouter transfer, performance first"]
FULL_LRB = ["oracle", "family/topic lookup, all topics", "family lookup", "family/topic lookup, released labels", "predicted-cell lookup, all topics", "kNN-100", "Avengers-Pro, $K=64$, top 3, upper bound", "$K$-means utility, $K=64$, top 3", "Avengers-Pro, $K=64$, top 1, upper bound", "kNN-40", "$K$-means utility, $K=64$, top 1", "kNN-10", "kNN-5", "RouteLLM-MF, two models", "best single model", "MLP (100,100)", "GraphRouter, cost first", "MLP (100)", "GraphRouter, balance", "GraphRouter, performance first"]


def table(label, caption, rb, lrb, order_rb, order_lrb, colsep="5pt"):
    L = [r"\begin{table}[H]", r"\centering", r"\scriptsize", rf"\setlength{{\tabcolsep}}{{{colsep}}}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule"]
    for name, R, order, g in (("RouterBench", rb, order_rb, G["RB"]), ("LLMRouterBench", lrb, order_lrb, G["LRB"])):
        for a, alab in (("qw", "per query"), ("eq", "equal weight over families")):
            L += [rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{name}, {alab}}}}} \\", "router & " + " & ".join(rf"$\lam={l:g}$" for l in g) + r" \\", r"\midrule"] + block(R, order, a, g) + [r"\midrule"]
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]; return "\n".join(L) + "\n"


def main():
    rb, lrb = build("RB"), build("LRB")
    cap_h = (r"Held-out routing utility, mean over five family-stratified splits, across the cost sweep. RouterBench is read at $\lam\in\{0,1,10,50,100,500\}$; LLMRouterBench, whose per-query costs are about 19 times larger, at the matching trade-offs $\lam\in\{0,0.5,1,2,5,10\}$ ($\lam\cdot c_{\text{strongest}}$ equal across the two benchmarks). Each benchmark has a per-query block and an equal-weight-over-families block. The oracle is an outcome-informed ceiling. The cell-lookup rows use true labels as analysis references; predicted-cell results and the full baseline comparison appear in Table~\ref{tab:baselines-full}. The all-topics lookup uses outcome-free labels for every LLMRouterBench family; its values at $\lam\in\{0.5,2,5\}$ are a recomputation under the same protocol (identical at the shared $\lam$).")
    cap_f = (r"Full held-out routing comparison, mean over five family-stratified splits, across the cost sweep (RouterBench $\lam\in\{0,1,10,50,100,500\}$; LLMRouterBench $\lam\in\{0,0.5,1,2,5,10\}$, the matching trade-offs). Per-query and equal-weight-over-families blocks per benchmark. Cell lookup rows use true labels unless marked predicted; the predicted labels come from a linear classifier trained on each split's embeddings. LLMRouterBench's all-topics labels are outcome-free labels for every family; the all-topics rows at $\lam\in\{0.5,2,5\}$ are recomputed under the same protocol. Avengers-Pro reports the best point of its test-selected $\alpha$ frontier as an upper bound, not a deployable router.")
    (REP / "tab_headline.tex").write_text(table("tab:headline", cap_h, rb, lrb, HEAD_RB, HEAD_LRB)); (REP / "tab_baselines_full.tex").write_text(table("tab:baselines-full", cap_f, rb, lrb, FULL_RB, FULL_LRB, "4pt"))
    print("wrote tab_headline.tex and tab_baselines_full.tex"); print((REP / "tab_headline.tex").read_text())


if __name__ == "__main__":
    main()
