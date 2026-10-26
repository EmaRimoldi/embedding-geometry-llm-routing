"""Align raw embedding arrays with the frozen panel rows read by code/common.py:load().

RouterBench arrays extracted over the audited RouterBench cohort are already in panel order (rows: none).
SPROUT arrays follow the SPROUT source panel and are indexed with gte_rows (GTE) or panel_rows (Qwen reader),
exactly as in the loader used for the paper.
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--panel', required=True, choices=['routerbench-0shot', 'sprout-sixfam'])
ap.add_argument('--rep', required=True, choices=['gte', 'reader'])
ap.add_argument('--raw', required=True, help='raw embedding array (.npy)')
ap.add_argument('--rows', default='auto', choices=['auto', 'none', 'gte_rows', 'panel_rows'])
ap.add_argument('--root', default=None, help='archive root (default: parent of tools/)')
a = ap.parse_args()
root = Path(a.root) if a.root else Path(__file__).resolve().parents[1]
z = np.load(root / 'astra_definitive_paper_20260923_v1' / 'inputs' / a.panel / 'panel.npz', allow_pickle=False)
rows = a.rows
if rows == 'auto':
    rows = 'none' if a.panel == 'routerbench-0shot' else ('gte_rows' if a.rep == 'gte' else 'panel_rows')
x = np.load(a.raw, mmap_mode='r')
x = np.asarray(x if rows == 'none' else x[z[rows]])
assert len(x) == len(z['r']), (len(x), len(z['r']))
out = root / 'embeddings' / a.panel / f'{a.rep}.npy'
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, x)
print('wrote', out, x.shape, x.dtype)
