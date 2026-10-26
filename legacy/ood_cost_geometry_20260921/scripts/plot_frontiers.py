"""Re-draw frontier figures from saved result.json files (no recomputation). Distinct colour+marker per method,
legend outside, highlighted methods drawn thicker. Usage: plot_frontiers.py <exp_dir_name> <panel>"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
exp, panel = sys.argv[1], sys.argv[2]; j = json.load(open(RES / exp / panel / "result.json"))
SELECT = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None   # exact method names to draw
OUTNAME = sys.argv[4] if len(sys.argv) > 4 else "frontier.png"
XSCALE = sys.argv[5] if len(sys.argv) > 5 else "log"
HIGHLIGHT = ("cell_lookup", "knn100", "family_lookup", "best_single", "oracle", "soft_prior", "verbosity_reader", "family_ceiling")
if exp.startswith("exp12"):
    panels = [("curves", "query-weighted"), (None, "equal weight over families")]
    curves_q = {m: np.array(v) for m, v in j["curves"].items()}
    # macro curves are not stored in exp12 json except via aiq; rebuild from per_family_equal_weight if present
    curves_m = {m: np.array(v) for m, v in j.get("curves_macro", {}).items()}
else:
    curves_q = {m: np.array(v) for m, v in j["frontier_qw"].items()}; curves_m = {m: np.array(v) for m, v in j["frontier_macro"].items()}
sets = [(curves_q, "query-weighted")] + ([(curves_m, "equal weight over families")] if curves_m else [])
cmap = plt.get_cmap("tab20"); markers = "osD^v<>Pp*hXd|_+"
fig, axs = plt.subplots(1, len(sets), figsize=(7.5 * len(sets) + 2.5, 5.2), squeeze=False)
names = sorted(curves_q, key=lambda m: (not any(h in m for h in HIGHLIGHT), m))
if SELECT: names = [m for m in SELECT if m in curves_q]
for ax, (curves, title) in zip(axs[0], sets):
    for i, m in enumerate(names):
        if m not in curves: continue
        p = curves[m]; hl = any(h in m for h in HIGHLIGHT)
        ax.plot(p[:, 0] * 1e3, p[:, 1], marker=markers[i % len(markers)], ms=5 if (hl or SELECT) else 3.5, lw=2.2 if (hl or SELECT) else 0.9, alpha=1.0, color=(plt.get_cmap("tab10")(i % 10) if SELECT else cmap(i % 20)), label=m)
    ax.set_xscale(XSCALE); ax.set_xlabel(f"mean cost per query (e-3 USD, {XSCALE})"); ax.set_ylabel("mean reward"); ax.set_title(f"{panel}: {exp.split('_')[0]} {title}"); ax.grid(alpha=.3)
axs[0][-1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False); fig.tight_layout()
out = RES / exp / panel / OUTNAME; fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
