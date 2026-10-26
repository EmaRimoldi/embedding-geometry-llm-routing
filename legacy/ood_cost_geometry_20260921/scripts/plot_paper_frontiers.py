"""Paper frontier figures from saved curves (exp12; LRB all-topics cell lookup from exp17). No recomputation.
Writes reports/frontier_<panel>_<agg>.pdf for agg in (pq, eq). Reported lambda ranges only: RB <= 500, LRB <= 10.
The y-axis is clipped to the routers' range: the oracle (flat, far above) is a text note; a router point below the
axis (RouteLLM-MF on RouterBench at lambda=500) is marked by an arrow with its value."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"; OUT = HERE.parent.parent / "reports"
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "legend.fontsize": 7.5, "pdf.fonttype": 42})
PANELS = {"routerbench-0shot": ("RouterBench", 500.0, [0.05, 0.1, 0.2, 0.5, 1, 2, 5]), "llmrouterbench-performance-cost": ("LLMRouterBench", 10.0, [2, 5, 10, 20, 50, 100]), "sprout": ("SPROUT", 1000.0, [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])}
import os; SPROUT_EXP = "exp21_sprout_frontier" + ("_v2" if os.environ.get("SPROUT_TOPICS") == "v2" else "")
STYLE = [  # (key, label, colour, marker, linestyle)
    ("cell", "(family, topic) cell lookup", "#d62728", "o", "-"),
    ("knn100", "kNN-100", "#1f77b4", "s", "--"),
    ("kmeans64_utility_top3", "$K$-means-64 utility router", "#2ca02c", "D", "-."),
    ("routellm_mf_train_selected_threshold", "RouteLLM-MF", "#ff7f0e", "^", "-"),
    ("best_single", "best single model", "#7f7f7f", "x", ":"),
]
for panel, (title, lmax, ticks) in PANELS.items():
    j = json.load(open(RES / SPROUT_EXP / "result.json")) if panel == "sprout" else json.load(open(RES / "exp12_indomain_frontier" / panel / "result.json")); lams = np.array(j["lambdas"]); keep = lams <= lmax
    curves = {"pq": {m: np.array(v)[keep] for m, v in j["curves"].items() if len(v) == len(lams)}, "eq": {m: np.array(v)[keep] for m, v in j["curves_macro"].items() if len(v) == len(lams)}}
    if panel == "sprout":
        curves["pq"]["cell"] = curves["pq"]["cell_lookup"]; curves["eq"]["cell"] = curves["eq"]["cell_lookup"]; curves["pq"]["kmeans64_utility_top3"] = curves["pq"]["kmeans64_top1"]; curves["eq"]["kmeans64_utility_top3"] = curves["eq"]["kmeans64_top1"]
    elif panel.startswith("llmrouterbench"):
        e = json.load(open(RES / "exp17_lrb_llmcell_curve" / "result.json")); assert e["lambdas"] == [float(l) for l in lams[keep]]
        curves["pq"]["cell"] = np.array(e["curves"]["cell_lookup_llm"]); curves["eq"]["cell"] = np.array(e["curves_macro"]["cell_lookup_llm"])
    else:
        curves["pq"]["cell"] = curves["pq"]["cell_lookup (recomputed)"]; curves["eq"]["cell"] = curves["eq"]["cell_lookup (recomputed)"]
    style = [t for t in STYLE if not ((panel.startswith("router") or panel == "sprout") and t[0] == "routellm_mf_train_selected_threshold")]   # RouteLLM-MF is far below on RouterBench: omitted there, stated in the appendix
    for agg, aggname in (("pq", "per query"), ("eq", "equal weight over families")):
        fig, ax = plt.subplots(figsize=(4.0, 3.7))
        allr = np.concatenate([curves[agg][k][:, 1] for k, *_ in style if k != "routellm_mf_train_selected_threshold"])
        lo, hi = allr.min() - 0.012, allr.max() + 0.012; hi = hi + 0.055 * (hi - lo)   # headroom for the oracle note
        off = [(key, lab, col, x * 1e3, y, l) for key, lab, col, *_ in style for x, y, l in zip(curves[agg][key][:, 0], curves[agg][key][:, 1], lams[keep]) if y < lo]
        if off: lo = lo - (0.06 + 0.07 * len(off)) * (hi - lo)   # a band under the curves for the off-chart labels
        xs = np.concatenate([curves[agg][k][:, 0] for k, *_ in style]) * 1e3; xmid = np.sqrt(xs.min() * xs.max())
        for key, lab, col, mk, ls in style:
            p = curves[agg][key]; big = key in ("cell", "knn100", "kmeans64_utility_top3", "routellm_mf_train_selected_threshold")
            ax.plot(p[:, 0] * 1e3, p[:, 1], marker=mk, ms=5.5 if big else 5, lw=1.7 if big else 1.1, ls=ls, color=col, label=lab, zorder=4 if key == "cell" else 3, markeredgecolor="white" if big else col, markeredgewidth=0.7 if big else 1.2, clip_on=True)
        for k, (key, lab, col, x, y, l) in enumerate(sorted(off, key=lambda t: t[3])):
            ax.annotate(f"{lab} {y:.2f} at $\\lambda$={l:g} (below the axis)", xy=(x, lo), xytext=(0.03, 0.025 + 0.062 * k), textcoords="axes fraction", fontsize=6.8, color=col, ha="left", va="bottom", arrowprops=dict(arrowstyle="->", color=col, lw=0.9, shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0.0"), annotation_clip=False)
        o = curves[agg]["oracle"][:, 1]; ax.text(0.02, 0.97, f"oracle ceiling: {o.min():.2f}--{o.max():.2f} (above the axis)", transform=ax.transAxes, fontsize=7, va="top", ha="left", color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.5))
        ax.set_ylim(lo, hi); ax.set_xscale("log"); ax.set_xlabel("mean cost per query ($10^{-3}$ USD, log scale)"); ax.set_ylabel("mean reward"); ax.set_title(f"{title}, {aggname}"); ax.grid(alpha=.3, which="both", lw=.4)
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, handlelength=2.4, columnspacing=1.0)
        fig.tight_layout(); out = OUT / f"frontier_{'rb' if panel.startswith('router') else 'sprout' if panel == 'sprout' else 'lrb'}_{agg}.pdf"; fig.savefig(out, bbox_inches="tight"); print("wrote", out.name, f"ylim {lo:.3f}-{hi:.3f}")
