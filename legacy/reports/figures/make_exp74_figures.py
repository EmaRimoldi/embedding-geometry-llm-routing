#!/usr/bin/env python
"""Figures for exp74 (pathological pairs). Reads results/ only; the PCA panel re-loads the
two embedding matrices (base and --aug) for the plotted rows without touching outcomes.
Usage: E4_PANEL=<panel> python make_exp74_figures.py [--aug reader_l21] [--smoke]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from _style import apply_style, savefig, PALETTE                                   # noqa: E402
from rb.data import load_panel                                                    # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT                                       # noqa: E402
from rb.reps import find_rep, load_rep                                            # noqa: E402
from rb import pathological as P                                                  # noqa: E402
from exp74_pathological_pairs import load_space                                   # noqa: E402

SHORT = {"routerbench-0shot": "rb", "llmrouterbench-performance-cost": "lrb"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--aug", default="reader_l21"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-segments", type=int, default=200); a = ap.parse_args(); apply_style()
    d = RESULTS_ROOT / "pathological_pairs" / ("exp74_pairs_smoke" if a.smoke else "exp74_pairs")
    pairs = pd.read_csv(d / "pairs.csv"); dist = np.load(d / "distances.npz"); met = json.load(open(d / "metrics.json")) if (d / "metrics.json").exists() else json.load(open(d / "result.json"))["metrics"]
    tag = SHORT[PANEL] + ("_smoke" if a.smoke else "")
    spaces = [s for s in dist["spaces"].tolist()]
    # --- Fig F1: normalised distance distributions, three groups, base vs each space
    fig, axes = plt.subplots(1, len(spaces), figsize=(1.9 * len(spaces) + 0.6, 1.9), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), spaces):
        for grp, col, lab in ((np.ones(len(pairs), bool), PALETTE[7], "all candidates"), (pairs.pathological.values, PALETTE[1], "pathological"), (pairs.benign.values, PALETTE[0], "benign")):
            ax.hist(np.log10(np.maximum(dist[f"{s}__dnorm"][grp], 1e-4)), bins=40, histtype="step", color=col, label=lab, density=True)
            ax.hist(np.log10(np.maximum(dist["base__dnorm"][grp], 1e-4)), bins=40, histtype="step", color=col, ls=":", density=True)
        ax.set_title(s); ax.set_xlabel("log10 normalised distance")
    np.atleast_1d(axes)[0].set_ylabel("density"); np.atleast_1d(axes)[0].legend(frameon=False)
    fig.text(0.5, -0.05, "solid: augmented space; dotted: base space", ha="center", fontsize=7)
    savefig(fig, f"exp74_distances_{tag}"); plt.close(fig)
    # --- Fig F2: log-log base vs augmented normalised distance with y=x, plus histogram of m
    fig, axes = plt.subplots(2, len(spaces), figsize=(1.9 * len(spaces) + 0.6, 3.6))
    axes = np.array(axes).reshape(2, -1)
    for k, s in enumerate(spaces):
        ax = axes[0, k]; m = dist[f"{s}__m"]
        for grp, col, lab in ((pairs.benign.values, PALETTE[0], "benign"), (pairs.pathological.values, PALETTE[1], "pathological")):
            ax.scatter(dist["base__dnorm"][grp], dist[f"{s}__dnorm"][grp], s=2, alpha=0.3, color=col, label=lab, rasterized=True)
        lim = [1e-3, 3]; ax.plot(lim, lim, color="k", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("base (normalised)"); ax.set_ylabel(f"{s} (normalised)"); ax.set_title(s)
        ax = axes[1, k]
        for grp, col, lab in ((pairs.benign.values, PALETTE[0], "benign"), (pairs.pathological.values, PALETTE[1], "pathological")):
            ax.hist(np.log10(m[grp]), bins=40, histtype="step", color=col, label=lab, density=True)
        r = met[s]["ratio_path_over_benign"]; ax.set_title(f"median ratio {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]"); ax.set_xlabel("log10 m"); ax.axvline(0, color="k", lw=0.6)
    axes[0, 0].legend(frameon=False, markerscale=4); savefig(fig, f"exp74_loglog_{tag}"); plt.close(fig)
    # --- Fig 3 (main): PCA base vs aug, pathological pairs as segments, colour = argmax model at lambda=100
    panel = load_panel(PANEL)
    rows = np.unique(np.concatenate([pairs.i.values, pairs.j.values]))
    sub = np.sort(np.unique(np.concatenate([rows, np.random.default_rng(0).choice(panel.n, size=min(6000, panel.n), replace=False)])))
    if a.aug == "big40k":
        print("[exp74-fig] note: --aug big40k loads the 5.6 GB 40k matrix; run this on the cpu_big "
              "partition (slurm/cpu_big.sbatch), not on a login node", flush=True)
    pos = {int(r): k for k, r in enumerate(sub)}
    lab = P.argmax_utility(np.nan_to_num(panel.rewards, nan=0.0), np.nan_to_num(panel.costs, nan=0.0, posinf=0.0),
                           100.0, available=np.asarray(panel.available, dtype=bool))[sub]
    sel = pairs[pairs.pathological].sample(n=min(a.n_segments, int(pairs.pathological.sum())), random_state=0)
    # The two embedding matrices are never held at the same time: base is projected and freed
    # before aug is loaded (aug is the 5.6 GB 40k block space in the --aug big40k case).
    base = P.unit_rows(load_rep(find_rep(PANEL, "computed/gte-qwen2-7b@prompt"), panel.n, panel.keep))[sub]
    z_base = PCA(2, random_state=0).fit_transform(base); del base
    aug = load_space(a.aug, panel)[sub]
    z_aug = PCA(2, random_state=0).fit_transform(aug); del aug
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))
    for ax, z, title in ((axes[0], z_base, "emitted GTE"), (axes[1], z_aug, a.aug)):
        for m in np.unique(lab):
            g = lab == m; col = PALETTE[m] if m < len(PALETTE) else PALETTE[m % len(PALETTE)]
            ax.scatter(z[g, 0], z[g, 1], s=1.5, alpha=0.25, color=col, label=panel.models[m], rasterized=True)
        for i, j in zip(sel.i.values, sel.j.values):
            ax.plot([z[pos[i], 0], z[pos[j], 0]], [z[pos[i], 1], z[pos[j], 1]], color="k", lw=0.4, alpha=0.7)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=6, markerscale=6, bbox_to_anchor=(0.5, -0.02))
    savefig(fig, f"exp74_projection_{tag}"); plt.close(fig)


if __name__ == "__main__":
    main()
