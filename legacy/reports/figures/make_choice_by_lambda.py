#!/usr/bin/env python
"""Fig. C1: share of routed queries per model as a function of lambda, family lookup vs
kNN-100 (stacked bars), from exp28 result.json.  RouterBench only (the LLMRouterBench
analogue exists only as prose)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
from _style import apply_style, savefig, PALETTE                                    # noqa: E402
LAMS = ["0", "1", "10", "50", "100", "500", "1000", "10000"]


def main() -> None:
    apply_style(); res = json.load(open(ROOT / "results" / "outcome_locality" / "exp28_choice_by_lambda" / "result.json"))
    models = [m.split("/")[-1] for m in res["models"]]; cols = [PALETTE[j % len(PALETTE)] for j in range(len(models))]   # PALETTE already has 13 colours
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    for ax, router in zip(axes, ("family_lookup", "knn100")):
        lams = [l for l in LAMS if l in res["share"][router]]; bottom = np.zeros(len(lams))
        for j, m in enumerate(models):
            vals = np.array([res["share"][router][l][j] for l in lams])
            ax.bar(range(len(lams)), vals, bottom=bottom, color=cols[j], label=m, width=0.8, lw=0); bottom += vals
        ax.set_xticks(range(len(lams))); ax.set_xticklabels(lams, rotation=45); ax.set_xlabel("λ"); ax.set_title(router.replace("_", " "))
    axes[0].set_ylabel("share of routed queries"); axes[1].legend(frameon=False, fontsize=5.5, bbox_to_anchor=(1.01, 1), loc="upper left")
    agr = res.get("agreement", {}); txt = ", ".join(f"λ={l}: {agr[l]['knn100_vs_cell']:.2f}" for l in ("0", "100", "1000") if l in agr)
    fig.text(0.5, -0.05, "kNN-100 agrees with the cell lookup on " + txt + " of queries", ha="center", fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "choice_by_lambda_rb"); plt.close(fig)


if __name__ == "__main__":
    main()
