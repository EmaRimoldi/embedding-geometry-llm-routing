#!/usr/bin/env python
"""Fig. G1: leave-one-family-out utility at lambda=100 per held-out family (exp32):
kNN-100, classify-cell lookup, best single model chosen on the training families, and the
in-domain ceiling. Any method missing for a family plots as NaN (a gap), never crashes
(ruling 5)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
from _style import apply_style, savefig, PALETTE                                    # noqa: E402
METHODS = [("knn100", "kNN-100"), ("classify_cell_lookup", "classify → cell lookup"), ("kmeans64", "K-means-64"), ("best_single_train", "best single (train families)"), ("true_family_lookup_in_domain", "in-domain ceiling")]


def main() -> None:
    apply_style(); res = json.load(open(ROOT / "results" / "generalization" / "exp32_lofo" / "result.json"))
    pf = res["per_family"]["prompt"]; fams = list(pf); lam = "100"
    fig, ax = plt.subplots(figsize=(5.5, 2.2)); w = 0.8 / len(METHODS)
    for i, (m, lab) in enumerate(METHODS):
        vals = [pf[f].get(lam, {}).get(m, np.nan) for f in fams]
        ax.bar(np.arange(len(fams)) + (i - len(METHODS) / 2 + 0.5) * w, vals, width=w, color=PALETTE[i], label=lab, lw=0)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=30, ha="right"); ax.set_ylabel("held-out utility, λ = 100")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    savefig(fig, "lofo_rb"); plt.close(fig)


if __name__ == "__main__":
    main()
