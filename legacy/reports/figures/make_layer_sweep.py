#!/usr/bin/env python
"""Fig. 2: R² beyond the family mean versus layer depth for every frozen reader, both panels,
last-token pooling for decoders and the encoder-native pooling for encoders; the emitted GTE
reference is a horizontal line.  Shows the whole depth curve, not the selected maximum."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(HERE))
from _style import apply_style, savefig, PALETTE                                    # noqa: E402
PANELS = {"RouterBench": ROOT / "results" / "difficulty_llm" / "exp37_layer_sweep", "LLMRouterBench": ROOT / "llmrouterbench" / "results" / "performance-cost" / "difficulty_llm" / "exp37_layer_sweep"}
READERS = ["gte-qwen2-7b", "qwen2.5-7b-instruct", "qwen2.5-coder-1.5b-instruct", "qwen3-embedding-0.6b", "e5-large-v2", "bge-m3", "minilm-l12"]
PREF = {"gte-qwen2-7b": "last", "qwen2.5-7b-instruct": "last", "qwen2.5-coder-1.5b-instruct": "last", "qwen3-embedding-0.6b": "last", "e5-large-v2": "mean", "bge-m3": "mean", "minilm-l12": "mean"}
FALLBACK = ("mean", "cls", "last")  # ruling 3: fallback order when preferred pooling is empty


def curve(res, pool):
    pts = []
    for k, v in res["layers"].items():
        m = re.fullmatch(r"L(\d+)_(\w+)", k)
        if m and m.group(2) == pool:
            pts.append((int(m.group(1)), v["r2_beyond_family"], v.get("r2_beyond_family_sd", 0.0)))
    pts.sort(); return np.array(pts)


def main() -> None:
    apply_style(); fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=False)
    for ax, (pn, root) in zip(axes, PANELS.items()):
        for i, rd in enumerate(READERS):
            p = root / rd / "result.json"
            if not p.exists():
                print(f"[{pn}] skipping {rd}: no result.json at {p}")
                continue
            res = json.load(open(p))
            pool = PREF[rd]; c = curve(res, pool)
            if len(c) == 0:                       # ruling 3: fall back through mean -> cls -> last
                for fb in FALLBACK:
                    if fb == pool:
                        continue
                    c = curve(res, fb)
                    if len(c):
                        pool = fb
                        break
            if len(c) == 0:
                avail = sorted(set(k.split("_", 1)[1] for k in res["layers"]))
                print(f"[{pn}] skipping {rd}: no usable pooling found; available poolings: {avail}")
                continue
            print(f"[{pn}] {rd}: using pooling '{pool}'")
            depth = c[:, 0] / c[:, 0].max()
            ax.plot(depth, c[:, 1], marker=".", ms=2.5, lw=0.9, color=PALETTE[i % len(PALETTE)], label=f"{rd} ({pool})")
            ax.fill_between(depth, c[:, 1] - c[:, 2], c[:, 1] + c[:, 2], color=PALETTE[i % len(PALETTE)], alpha=0.12, lw=0)
            if rd == "gte-qwen2-7b":
                ax.axhline(res["gte_reference"]["r2_beyond_family"], color=PALETTE[0], ls="--", lw=0.8)
        ax.set_title(pn); ax.set_xlabel("relative depth (layer / last layer)"); ax.axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("R² beyond family mean (solve count)"); axes[1].legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(0.5, -0.03, "dashed: emitted GTE reference; band: ± sd over five splits", ha="center", fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "layer_sweep"); plt.close(fig)


if __name__ == "__main__":
    main()
