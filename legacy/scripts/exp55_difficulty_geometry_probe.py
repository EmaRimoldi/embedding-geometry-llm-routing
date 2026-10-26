"""Read-only diagnostic of difficulty geometry in frozen panel embeddings.

Within each released family/topic cell, compare cosine for pairs in the same
outcome tail against pairs in opposite tails. The tail assignment uses only
within-cell outcomes; no embedding is used to select endpoints. Pair scores
are evaluated per cell to prevent topic centroids from inflating the result.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rb.data import SUCCESS_THRESHOLD, load_panel
from rb.panelctx import PANEL, REP, topic_str
from rb.reps import find_rep, load_rep


def one_cell(x, y, ids, rng, pairs=2000):
    lo, hi = np.quantile(y[ids], [.25, .75])
    low = ids[y[ids] <= lo]
    high = ids[y[ids] >= hi]
    if len(low) < 15 or len(high) < 15 or lo == hi:
        return None
    # Strictly disjoint groups when a quantile lies in a mass of ties.
    low = low[y[low] < hi]
    high = high[y[high] > lo]
    if len(low) < 15 or len(high) < 15:
        return None
    a = rng.choice(low, pairs); b = rng.choice(high, pairs)
    def distinct(group, count):
        first = rng.choice(group, count)
        second = rng.choice(group, count)
        bad = first == second
        while bad.any():
            second[bad] = rng.choice(group, int(bad.sum()))
            bad = first == second
        return np.column_stack([first, second])
    same_l = distinct(low, pairs // 2); same_h = distinct(high, pairs // 2)
    cross = np.einsum('ij,ij->i', x[a], x[b])
    same = np.r_[np.einsum('ij,ij->i', x[same_l[:,0]], x[same_l[:,1]]),
                 np.einsum('ij,ij->i', x[same_h[:,0]], x[same_h[:,1]])]
    auc = roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(cross))], np.r_[same, cross])
    return dict(n=int(len(ids)), n_low=int(len(low)), n_high=int(len(high)),
                q25=float(lo), q75=float(hi), auc=float(auc),
                same_cos=float(same.mean()), opposite_cos=float(cross.mean()),
                delta_cos=float(same.mean()-cross.mean()))


def main():
    p = load_panel(PANEL)
    fam = p.dataset_index()
    names, cell = np.unique(np.asarray(topic_str(p), dtype=str), return_inverse=True)
    x = load_rep(find_rep(PANEL, REP), p.n, p.keep).astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    rewards = np.nan_to_num(p.rewards, nan=0.0)
    proxies = {"solve_fraction": (rewards >= SUCCESS_THRESHOLD).mean(1),
               "mean_reward": rewards.mean(1)}
    out = {"panel": PANEL, "rep": REP, "seed": 6071, "pairs_per_cell_type": 2000,
           "method": "within-cell bottom/top outcome quartiles; cosine AUC of same-tail vs opposite-tail random pairs",
           "results": {}}
    for proxy, y in proxies.items():
        rng = np.random.default_rng(6071)
        rows = {}
        for c, name in enumerate(names):
            ids = np.flatnonzero(cell == c)
            if len(ids) < 60: continue
            res = one_cell(x, y, ids, rng)
            if res is not None:
                res["family"] = p.datasets[int(fam[ids[0]])]
                rows[str(name)] = res
        byfam = {}
        for f in p.datasets:
            ff = [v for v in rows.values() if v["family"] == f]
            if ff:
                byfam[f] = {"cells":len(ff), "queries":sum(v["n"] for v in ff),
                            "auc_equal_cell":float(np.mean([v["auc"] for v in ff])),
                            "auc_query_weighted":float(np.average([v["auc"] for v in ff], weights=[v["n"] for v in ff])),
                            "delta_cos_equal_cell":float(np.mean([v["delta_cos"] for v in ff]))}
        allrows = list(rows.values())
        out["results"][proxy] = {"cells":rows, "per_family":byfam,
             "auc_equal_cell":float(np.mean([v["auc"] for v in allrows])),
             "auc_query_weighted":float(np.average([v["auc"] for v in allrows], weights=[v["n"] for v in allrows])),
             "auc_equal_family":float(np.mean([v["auc_equal_cell"] for v in byfam.values()]))}
    dest = ROOT / "results" / "difficulty_geometry" / (PANEL + "_quartile_pairs_v2.json")
    if dest.exists(): raise FileExistsError(dest)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k:{m:v[m] for m in ("auc_equal_cell","auc_query_weighted","auc_equal_family","per_family")}
                      for k,v in out["results"].items()}, indent=2), flush=True)
    print("SAVED", dest, flush=True)

if __name__ == "__main__": main()
