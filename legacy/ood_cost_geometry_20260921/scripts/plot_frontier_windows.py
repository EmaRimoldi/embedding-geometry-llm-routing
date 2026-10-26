"""Frontier figure with the lambda reading windows marked (no recomputation). Usage: plot_frontier_windows.py <exp> <panel> <methods,comma>"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
exp, panel, sel = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
j = json.load(open(RES / exp / panel / "result.json"))
lams = [float(l) for l in j["lambdas"]]; curves = j["curves"] if "curves" in j else j["frontier_qw"]
lrb = panel.startswith("llm")
old = [50.0, 100.0, 1000.0] if lrb else [10.0, 50.0, 100.0, 1000.0]; new = [0.5, 2.0, 5.0, 10.0] if lrb else [10.0, 50.0, 100.0, 1000.0]
fig, ax = plt.subplots(figsize=(9.5, 5.4)); cmap = plt.get_cmap("tab10")
for i, m in enumerate(sel):
    if m not in curves: continue
    p = np.array(curves[m]); ax.plot(p[:, 0] * 1e3, p[:, 1], "-", lw=1.6, color=cmap(i), label=m, alpha=0.9)
    for L in new:
        if L in lams: k = lams.index(L); ax.plot(p[k, 0] * 1e3, p[k, 1], "o", ms=8, color=cmap(i), mec="black")
    if lrb:
        for L in old:
            if L in lams: k = lams.index(L); ax.plot(p[k, 0] * 1e3, p[k, 1], "x", ms=9, color=cmap(i), mew=2)
ref = [m for m in sel if m in curves][0]; p = np.array(curves[ref])
for L in new:
    if L in lams: k = lams.index(L); ax.annotate(f"λ={L:g}", (p[k, 0] * 1e3, p[k, 1]), textcoords="offset points", xytext=(6, 8), fontsize=8)
title = f"{panel}: {'held-out family' if exp.startswith('exp11') else 'in-domain'} — circles: " + ("new LRB reading window λ∈{0.5,2,5,10} (≈RB 10–190); crosses: old window λ∈{50,100,1000}" if lrb else "RB reading window λ∈{10,50,100,1000}")
ax.set_xscale("log"); ax.set_xlabel("mean cost per query (e-3 USD, log)"); ax.set_ylabel("mean reward"); ax.set_title(title, fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
out = RES / exp / panel / "frontier_windows.png"; fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)
