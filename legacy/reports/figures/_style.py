"""Shared matplotlib style for every paper figure. Read-only helpers, no data access."""
from __future__ import annotations
import os, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parents[1]
OL_FIGS = Path(os.environ.get("OL_FIGURES", Path.home() / "paper/figures"))
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
           "#999999", "#66CCEE", "#AA4499", "#DDCC77", "#332288"]


def apply_style() -> None:
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.prop_cycle": matplotlib.cycler(color=PALETTE)})


def savefig(fig, stem: str) -> Path:
    out = REPORTS / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    png = REPORTS / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", dpi=110)
    if OL_FIGS.parent.exists():
        OL_FIGS.mkdir(exist_ok=True); shutil.copy(out, OL_FIGS / out.name); print("copied", OL_FIGS / out.name)
    print("wrote", out, flush=True)
    print("wrote", png, flush=True)
    return out
