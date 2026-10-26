"""Do exp51's 'cells' actually respect family boundaries, and how saturated is each family within cells?

Replicates exp51_cell_floor.py's own cell construction verbatim, then:
  (1) checks whether any cell spans more than one family (if so, the cell floor
      is NOT a within-family floor and the '(family, topic)' label is wrong);
  (2) reports, per family, the variance of the solve count around the family mean
      and around its own cell means, which is the denominator that makes
      per-family R2 beyond the cell floor explode.
Loads the outcome table only. No embeddings, no fitting.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, topic_str  # noqa: E402

MIN_CELL = 50

panel = load_panel(PANEL)
n = panel.n
fam = panel.dataset_index()
names = [str(s) for s in panel.datasets]
F = len(names)
r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
y = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
labels, cell = np.unique(np.array([str(t) for t in topic_str(panel)], dtype=object), return_inverse=True)
n_cells = int(cell.max()) + 1

print("=" * 78)
print("panel:", PANEL, " rows:", n, " families:", F, " cells:", n_cells)
print("sample topic labels:", [str(x) for x in labels[:6]])
print()

print("--- (1) does any cell span more than one family?")
bad = 0
for c in range(n_cells):
    m = cell == c
    fams_here = sorted(set(fam[m].tolist()))
    if len(fams_here) > 1:
        bad += 1
        if bad <= 10:
            print("    CELL %-3d label=%-34s rows=%6d spans %d families: %s"
                  % (c, str(labels[c])[:34], int(m.sum()), len(fams_here),
                     ", ".join("%s(%d)" % (names[g], int((m & (fam == g)).sum())) for g in fams_here)))
print("    cells spanning >1 family: %d of %d" % (bad, n_cells))
if bad == 0:
    print("    OK: every cell lies inside a single family, so the cell floor is a within-family floor.")
print()

print("--- cells per family")
for g in range(F):
    m = fam == g
    cs = sorted(set(cell[m].tolist()))
    big = [c for c in cs if int(((cell == c) & m).sum()) >= MIN_CELL]
    print("    %-20s rows=%6d  cells=%3d  cells with >=%d rows: %d"
          % (names[g], int(m.sum()), len(cs), MIN_CELL, len(big)))
print()

print("--- (2) target spread per family: around the family mean vs around its cell means")
print("    %-20s %7s %8s %10s %10s %10s" % ("family", "n", "mean", "var(fam)", "var(cell)", "cell/fam"))
for g in range(F):
    m = fam == g
    yy = y[m]
    v_fam = float(yy.var())
    resid = yy.copy()
    for c in set(cell[m].tolist()):
        mm = (cell == c)[m]
        if mm.sum() > 0:
            resid[mm] = yy[mm] - yy[mm].mean()
    v_cell = float((resid ** 2).mean())
    ratio = v_cell / v_fam if v_fam > 1e-12 else float("nan")
    print("    %-20s %7d %8.3f %10.4f %10.4f %10.3f"
          % (names[g], int(m.sum()), float(yy.mean()), v_fam, v_cell, ratio))
print()
print("    var(cell) is the denominator of per-family R2 beyond the CELL floor.")
print("    A family with small var(cell) makes that R2 unstable at any family size.")
