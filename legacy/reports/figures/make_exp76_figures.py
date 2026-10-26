#!/usr/bin/env python
"""Figures for exp76 (lookup-plus-reasoning oracle), one panel per run.

Reads every ``summary.json`` written by ``scripts/exp76_oracle_reasoning_analyze.py`` under
``RESULTS_ROOT/oracle_reasoning/exp76_lookup_reasoning/<tag>/``, where ``<tag>`` is
``<mode>_lam<lambda>[_lofo][_smoke]``, for the panel named by ``E4_PANEL``.  Two figures:

* ``exp76_bars_<rb|lrb>.pdf`` — grouped bars of mean utility per tag, per query (left) and
  per-family equal weight (right), for the four arms ``utility_chosen`` (the model's own
  lookup+reasoning decision), ``utility_knn`` (the kNN-k counterfactual over the retrieved
  rows, NOT the paper's kNN-100 router), ``utility_cell`` and ``utility_oracle``.
* ``exp76_sources_<rb|lrb>.pdf`` — stacked bars of ``source_share`` (lookup vs reasoning)
  with ``accuracy_by_source`` on a second axis.

No live run exists yet, so the expected behaviour today is one printed line and exit 0.
Usage: E4_PANEL=<panel> python make_exp76_figures.py [--smoke-tags]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
from _style import apply_style, savefig, PALETTE                                  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT                                       # noqa: E402

SHORT = {"routerbench-0shot": "rb", "llmrouterbench-performance-cost": "lrb"}
ARMS = [("utility_chosen", "lookup + reasoning"), ("utility_knn", "kNN-k counterfactual"),
        ("utility_cell", "cell lookup"), ("utility_oracle", "oracle")]
AGGS = [("per_query", "per query"), ("equal_weight", "equal family")]


def load_summaries(keep_smoke: bool) -> dict[str, dict]:
    """{tag: summary dict} for every tag directory that has a summary.json."""
    base = RESULTS_ROOT / "oracle_reasoning" / "exp76_lookup_reasoning"
    if not base.is_dir():
        return {}
    out = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or (d.name.endswith("_smoke") and not keep_smoke):
            continue
        f = d / "summary.json"
        if not f.exists():
            continue
        try:
            out[d.name] = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"warning: skipping {f} ({e})")
    return out


def bars_figure(sums: dict[str, dict], tag: str) -> None:
    tags = list(sums)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
    xs = np.arange(len(tags)); w = 0.8 / len(ARMS)
    for ax, (agg, agg_label) in zip(axes, AGGS):
        for k, (arm, arm_label) in enumerate(ARMS):
            vals = [sums[t].get("utility", {}).get(arm, {}).get(agg, np.nan) for t in tags]
            ax.bar(xs + (k - (len(ARMS) - 1) / 2) * w, vals, width=w, color=PALETTE[k],
                   label=arm_label, lw=0)
        ax.set_xticks(xs); ax.set_xticklabels(tags, rotation=30, ha="right")
        ax.set_title(agg_label); ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("mean utility (reward − λ·cost)")
    axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.text(0.5, -0.06, "kNN-k is the counterfactual over the rows retrieved into the prompt, "
                         "not the kNN-100 router of the main text", ha="center", fontsize=6)
    fig.tight_layout()
    savefig(fig, f"exp76_bars_{tag}"); plt.close(fig)


def sources_figure(sums: dict[str, dict], tag: str) -> None:
    tags = list(sums)
    sources = sorted({s for t in tags for s in sums[t].get("source_share", {})}) or ["lookup", "reasoning"]
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    xs = np.arange(len(tags)); bottom = np.zeros(len(tags))
    for k, src in enumerate(sources):
        vals = np.array([float(sums[t].get("source_share", {}).get(src, 0.0)) for t in tags])
        ax.bar(xs, vals, bottom=bottom, width=0.6, color=PALETTE[k], label=f"share: {src}", lw=0)
        bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("share of decisions"); ax.set_ylim(0, 1)
    ax2 = ax.twinx(); ax2.set_ylabel("accuracy of that source"); ax2.set_ylim(0, 1)
    ax2.spines["right"].set_visible(True)
    for k, src in enumerate(sources):
        acc = [sums[t].get("accuracy_by_source", {}).get(src, np.nan) for t in tags]
        ax2.plot(xs, acc, marker="o", ms=3.5, lw=0.9, ls="--", color=PALETTE[k],
                 markeredgecolor="k", markeredgewidth=0.4, label=f"accuracy: {src}")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, bbox_to_anchor=(1.12, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, f"exp76_sources_{tag}"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-tags", action="store_true", help="also plot the *_smoke tag directories")
    a = ap.parse_args()
    sums = load_summaries(a.smoke_tags)
    if not sums:
        print(f"no exp76 summaries under {RESULTS_ROOT / 'oracle_reasoning' / 'exp76_lookup_reasoning'} yet")
        return
    apply_style()
    tag = SHORT.get(PANEL, PANEL)
    print(f"[exp76-fig] {PANEL}: {len(sums)} tag(s): {', '.join(sums)}")
    bars_figure(sums, tag)
    sources_figure(sums, tag)


if __name__ == "__main__":
    main()
