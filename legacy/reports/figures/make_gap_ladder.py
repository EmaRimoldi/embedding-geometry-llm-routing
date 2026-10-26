#!/usr/bin/env python
"""Fig. D1: oracle-minus-cell-lookup gap versus lambda, and the share of that gap closed
when the cheapest, the two cheapest, ... five cheapest models are removed from the oracle's
action set (exp30 section A). Per query on top; per-family-equal-weight on a second row of
panels when that sibling key is present (ruling 4).

Each ablation curve is the utility actually removed, ``oracle_full - oracle_minus_k``, plotted
directly. It used to be computed as (share of the gap closed) x (the gap), i.e. the same
quantity multiplied and divided by ``oracle_full - cell_lookup``; that round trip only added a
1e-9 floor and a division, so it is gone.

The `oracle_no_mistral` key removes the panel's CHEAP_MODEL role, not literally Mistral on
every panel: Mistral-7B on RouterBench, Qwen3-235B-A22B-2507 on LLMRouterBench (lowest median
cost; E4/DECISIONS.md L8, src/rb/panelctx.py CHEAP_MODEL). `oracle_minus_k_cheapest` removes
the k cheapest models by MEAN cost in `cheapest_order`. Legend labels are built per panel so
they never say "mistral" on LLMRouterBench (fix round 1)."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
from _style import apply_style, savefig, PALETTE                                    # noqa: E402
PANELS = {"RouterBench": ROOT / "results" / "oracle_gap" / "exp30_unrecoverable_models" / "result.json", "LLMRouterBench": ROOT / "llmrouterbench" / "results" / "performance-cost" / "oracle_gap" / "exp30_unrecoverable_models" / "result.json"}
CHEAP_MODEL_LABEL = {"RouterBench": "Mistral-7B", "LLMRouterBench": "Qwen3-235B-A22B-2507"}
AGG_LABEL = {"per_query": "per query", "per_family_equal_weight": "equal family"}


def label_for(key: str, pn: str) -> str:
    if key == "oracle_no_mistral":
        return f"without the cheap-model role ({CHEAP_MODEL_LABEL[pn]})"
    m = re.fullmatch(r"oracle_minus_(\d+)_cheapest", key)
    if m:
        return f"without the {m.group(1)} cheapest (mean cost)"
    return key  # unexpected key: fall back to the raw name rather than crash


def plot_agg(ax, A, lams, agg, pn):
    d0 = A[lams[0]][agg]
    gap = np.array([A[l][agg]["oracle_full"] - A[l][agg]["cell_lookup"] for l in lams])
    ax.plot(range(len(lams)), gap, marker="o", ms=3, color="k", label="oracle − cell lookup (the gap)")
    keys = [k for k in d0 if k.startswith("oracle_no_") or k.startswith("oracle_minus_")]
    for i, k in enumerate(keys):
        removed = np.array([A[l][agg]["oracle_full"] - A[l][agg][k] for l in lams])
        ax.plot(range(len(lams)), removed, marker=".", ms=2.5, lw=0.8, color=PALETTE[i % len(PALETTE)], label=label_for(k, pn))
    ax.set_xticks(range(len(lams))); ax.set_xticklabels(lams)


def main() -> None:
    apply_style()
    loaded = {}
    for pn, p in PANELS.items():
        if not p.exists():
            print(f"skipping {pn}: {p} not found")
            continue
        loaded[pn] = json.load(open(p))["A_oracle_ablation"]

    for pn, d in loaded.items():
        l0 = sorted(d, key=float)[0]
        print(f"[{pn}] A_oracle_ablation['{l0}']['per_query'].keys() = {list(d[l0]['per_query'].keys())}")
    has_efw = all("per_family_equal_weight" in d[sorted(d, key=float)[0]] for d in loaded.values())
    aggs = ["per_query", "per_family_equal_weight"] if has_efw else ["per_query"]
    print("aggregation rows:", aggs)

    nrows, ncols = len(aggs), len(PANELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 2.3 * nrows), squeeze=False)
    for r, agg in enumerate(aggs):
        for c, pn in enumerate(PANELS):
            ax = axes[r][c]
            if pn not in loaded:
                ax.set_title(pn + " (missing)"); continue
            A = loaded[pn]; lams = sorted(A, key=float)
            plot_agg(ax, A, lams, agg, pn)
            ax.set_xlabel("λ")
            ax.set_title(pn if r == 0 else "")
        axes[r][0].set_ylabel(f"utility gap and the part removed\n({AGG_LABEL[agg]})")
    # each column gets its own legend (labels differ by panel, e.g. the cheap-model name),
    # placed inside the top panel of that column rather than one shared figure legend
    for c in range(ncols):
        axes[0][c].legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=4.8,
                           loc="upper left", handlelength=1.3, labelspacing=0.3, borderpad=0.4)
    fig.tight_layout()
    savefig(fig, "gap_ladder"); plt.close(fig)


if __name__ == "__main__":
    main()
