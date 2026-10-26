#!/usr/bin/env python
"""Fig. 4: separation gain (x) versus routing gain (y) per (panel, family, space), one
panel per lambda in {0, 100}; the (family, predicted argmax) router from exp44 is drawn as
a horizontal reference line because it changes the head, not the space."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
from _style import apply_style, savefig, PALETTE                                    # noqa: E402
ROOTS = {"RouterBench": ROOT / "results" / "pathological_pairs", "LLMRouterBench": ROOT / "llmrouterbench" / "results" / "performance-cost" / "pathological_pairs"}
MARK = {"RouterBench": "o", "LLMRouterBench": "s"}; COL = {"gte_l21": PALETTE[0], "reader_l21": PALETTE[1], "big40k": PALETTE[2]}
LS = {"RouterBench": "--", "LLMRouterBench": "-."}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); apply_style()
    exp = "exp75_sep_vs_routing" + ("_smoke" if a.smoke else "")
    frames, refs = [], {}
    for pn, rt in ROOTS.items():
        if (rt / exp / "table.csv").exists():
            df = pd.read_csv(rt / exp / "table.csv"); df["panel_name"] = pn; frames.append(df)
            try:
                refs[pn] = json.load(open(rt / exp / "pooled.json"))["references"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
                print(f"warning: {pn} pooled.json missing or unreadable ({exc}); plotting without its exp44 reference line")
    if not frames:
        raise SystemExit("no exp75 outputs found")
    df = pd.concat(frames)
    n_before = len(df)
    df = df.dropna(subset=["sep_gain", "routing_gain"])
    print(f"dropped {n_before - len(df)} of {n_before} rows with NaN sep_gain/routing_gain")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=False)
    for ax, lam in zip(axes, (0.0, 100.0)):
        d = df[df["lambda"] == lam]
        for (pn, sp), g in d.groupby(["panel_name", "space"]):
            # routing_gain is the mean over 5 seeds of the per-seed family-mean differences;
            # gain_sd is the population std over those same 5 differences, so it is a
            # symmetric error bar centred on the plotted point. The split-0-only bootstrap
            # (gain_lo/gain_hi) is centred on a different estimate (gain_split0) and is not
            # used here -- see exp75_separation_vs_routing.py "Fix round 2".
            ax.errorbar(g.sep_gain, g.routing_gain, yerr=g.gain_sd, fmt=MARK[pn], ms=3.5, color=COL.get(sp, "k"), ecolor=COL.get(sp, "k"), elinewidth=0.5, alpha=0.85, label=f"{pn} / {sp}")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        x0, x1 = ax.get_xlim(); x_off = x0 + 0.02 * (x1 - x0)
        for pn, r in refs.items():
            if "exp44" in r and str(lam) in r["exp44"]:
                e = r["exp44"][str(lam)]; gain = e["family+predicted_argmax"]["per_query"] - e["family+topic"]["per_query"]
                ax.axhline(gain, color="k", ls=LS[pn], lw=0.7)
                ax.text(x_off, gain, f"{pn}: argmax key", fontsize=6, va="bottom", ha="left")
        ax.set_xlabel("separation gain (10-NN purity of the λ=100\nbest-model label, aug − base)")
        ax.set_ylabel("routing gain (kNN-100 utility, aug − base)"); ax.set_title(f"λ = {lam:g}")
    axes[0].legend(frameon=False, fontsize=6, ncol=1)
    fig.tight_layout()
    fig.text(0.5, -0.02, "x: 10-NN purity of the λ=100 best-model (utility-argmax) label, augmented minus base; "
                         "error bars: ± SD over the 5 splits", ha="center", va="top", fontsize=6)
    savefig(fig, "exp75_sep_vs_routing" + ("_smoke" if a.smoke else "")); plt.close(fig)


if __name__ == "__main__":
    main()
