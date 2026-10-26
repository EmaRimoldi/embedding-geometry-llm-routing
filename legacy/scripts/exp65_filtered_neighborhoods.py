"""Conditional cosine neighborhoods versus an exact same-cell random reference.

New independent analysis; reads frozen panels and embeddings, writes only an
experiment-specific directory. Split definitions and cell labels match exp51.
The third outcome is a routing-relevant vector profile, not an intrinsic scalar
difficulty rating. No difficulty label is used to select neighbours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rb.data import SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, topic_str, _rb_record_prefix, _HS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

KS = (5, 10, 40, 100)
METRICS = ("solve_fraction", "mean_reward", "reward_profile")
BOOT = 1000


def safe_topic_str(panel):
    if not panel.name.startswith("routerbench"):
        return topic_str(panel)
    pre=_rb_record_prefix(panel)
    src=np.array(["wikihow" if _HS.search(str(t)) else "activitynet" for t in panel.prompts])
    hs=np.char.add("hellaswag/",src)
    return np.where(panel.dataset_ids=="mmlu",pre,np.where(panel.dataset_ids=="hellaswag",hs,panel.dataset_ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="difficulty_l21/exp65_conditional")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--rep", choices=("emitted","l21"), required=True)
    args = ap.parse_args()
    out = RESULTS_ROOT / "difficulty_filtered" / args.rep
    if out.exists():
        raise FileExistsError(f"Will not overwrite an existing experiment: {out}")
    t0 = time.time()
    p = load_panel(PANEL)
    r = np.nan_to_num(p.rewards, nan=0.0).astype(np.float32)
    vals = [(r >= SUCCESS_THRESHOLD).mean(1).astype(np.float32), r.mean(1), r]
    names, cell = np.unique(np.asarray(safe_topic_str(p), dtype=str), return_inverse=True)
    fam = p.dataset_index()
    rep_path = ROOT / "features" / PANEL / "prompt" / "hidden" / "gte-qwen2-7b" / "L21_last.npy"
    if args.rep == "emitted":
        rep_path = Path(find_rep(PANEL, REP).path)
        x = load_rep(find_rep(PANEL, REP), p.n, p.keep).copy()
    else:
        raw = np.load(rep_path, mmap_mode="r")
        x = np.asarray(raw[p.keep] if p.keep is not None and len(raw) == len(p.keep) else raw, dtype=np.float32).copy()
    assert x.shape[0] == p.n
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    lengths = np.load(ROOT / "features" / PANEL / "prompt" / "lengths.npz")["gte_tokens"]
    lengths = lengths[p.keep] if p.keep is not None and len(lengths) == len(p.keep) else lengths
    valid = lengths < 8000
    rows = []
    for seed in SEEDS[:args.splits]:
        tr, te = split(p.n, fam, seed)
        tr, te = np.sort(tr[valid[tr]]), np.sort(te[valid[te]])
        print(f"{PANEL} seed={seed} train={len(tr)} test={len(te)}", flush=True)
        for c in np.unique(cell[te]):
            a = tr[cell[tr] == c]
            b = te[cell[te] == c]
            if len(a) < 2:  # no meaningful within-cell comparison
                continue
            kmax = min(max(KS), len(a))
            for start in range(0, len(b), 192):
                q = b[start:start + 192]
                with threadpool_limits(limits=8):
                    sim = x[q] @ x[a].T
                nearest = np.argpartition(-sim, kmax - 1, axis=1)[:, :kmax]
                nearest = np.take_along_axis(nearest, np.argsort(-np.take_along_axis(sim, nearest, axis=1), axis=1), axis=1)
                near_ids = a[nearest]
                for vi, value in enumerate(vals):
                    if vi < 2:
                        dn = np.abs(value[q, None] - value[near_ids])
                        ref = np.abs(value[q, None] - value[a]).mean(1)
                    else:
                        dn = np.abs(value[q, None, :] - value[near_ids]).mean(2)
                        ref = np.abs(value[q, None, :] - value[a]).mean((1, 2))
                    for k in KS:
                        kk = min(k, kmax)
                        near = dn[:, :kk].mean(1)
                        for i, qid in enumerate(q):
                            rows.append((seed, qid, fam[qid], c, vi, k, kk, float(near[i]), float(ref[i])))
            print(f"  {names[c]}: {len(b)} held-out, {len(a)} train ({time.time()-t0:.0f}s)", flush=True)

    dtype = [("seed", "i4"), ("query", "i4"), ("family", "i4"), ("cell", "i4"),
             ("proxy", "i4"), ("k", "i4"), ("effective_k", "i4"), ("near", "f4"), ("random", "f4")]
    arr = np.array(rows, dtype=dtype)
    out.mkdir(parents=True)
    np.save(out / "query_metrics.npy", arr)
    report = {**p.provenance(), "experiment": "exp65_l21_conditional", "rep": args.rep, "token_filter": "gte_tokens < 8000; both train and held-out rows", "valid_rows": int(valid.sum()),
              "rep_sha256": hashlib.sha256(rep_path.read_bytes()).hexdigest(),
              "proxies": list(METRICS), "ks": list(KS), "seeds": list(SEEDS[:args.splits]),
              "reference": "mean absolute difference over all training rows of the same family/topic cell",
              "cells": {str(n): int((cell == i).sum()) for i, n in enumerate(names)}, "families_order": p.datasets,
              "bootstrap": "1000 resamples of unique queries, stratified within family; mean held-out appearances per query before resampling",
              "results": {}}
    rng = np.random.default_rng(521)
    for vi, metric in enumerate(METRICS):
        report["results"][metric] = {}
        for k in KS:
            s = arr[(arr["proxy"] == vi) & (arr["k"] == k)]
            count = np.bincount(s["query"], minlength=p.n)
            ids = np.flatnonzero(count)
            per_near = np.bincount(s["query"], weights=s["near"], minlength=p.n)[ids] / count[ids]
            per_rand = np.bincount(s["query"], weights=s["random"], minlength=p.n)[ids] / count[ids]
            family_ids = fam[ids]
            groups = [np.flatnonzero(family_ids == f) for f in range(len(p.datasets))]
            def effect(ix):
                rn, rr = per_near[ix].mean(), per_rand[ix].mean()
                return float(1 - rn / rr) if rr > 1e-12 else float("nan")
            pooled = effect(np.arange(len(ids)))
            eqw = float(np.nanmean([effect(g) for g in groups if len(g)]))
            boot_p = np.empty(BOOT); boot_e = np.empty(BOOT)
            for z in range(BOOT):
                bg = [rng.choice(g, len(g), replace=True) for g in groups if len(g)]
                boot_p[z] = effect(np.concatenate(bg))
                boot_e[z] = np.nanmean([effect(g) for g in bg])
            report["results"][metric][str(k)] = {
                "per_query": {"random": float(per_rand.mean()), "nearest": float(per_near.mean()), "relative_reduction": pooled,
                              "ci95": np.quantile(boot_p, [.025, .975]).tolist()},
                "equal_family": {"relative_reduction": eqw, "ci95": np.quantile(boot_e, [.025, .975]).tolist()},
                "per_family": {p.datasets[f]: {"n": len(g), "random": float(per_rand[g].mean()), "nearest": float(per_near[g].mean()),
                                              "relative_reduction": effect(g)} for f, g in enumerate(groups)},
                "per_split": [{"seed": int(seed), "relative_reduction": float(1 - z["near"].mean() / z["random"].mean())}
                              for seed in SEEDS[:args.splits] if len(z := s[s["seed"] == seed])],
                "queries": len(ids), "short_k_rows": int((s["effective_k"] < k).sum()),
            }
    report["seconds"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print(f"FINISHED {out} ({report['seconds']:.0f}s)", flush=True)
    for metric, byk in report["results"].items():
        print(metric, [(k, round(v["per_query"]["relative_reduction"], 3),
                         round(v["equal_family"]["relative_reduction"], 3)) for k, v in byk.items()], flush=True)


if __name__ == "__main__":
    main()
