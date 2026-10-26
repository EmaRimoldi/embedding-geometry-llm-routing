#!/usr/bin/env python
"""exp74 — pathological pairs: do augmented representations separate near-identical
queries with different outcome profiles more than they separate everything else?

Base space: unit gte-qwen2-7b (emitted).  Augmented: GTE-own layer 21, Qwen2.5-7B reader
layer 21, the 40k block space, and the hashed bag-of-words lexical control.  Pairs come
from the base space only (10 nearest same-family rows at cosine >= --cos-min); a pair is
pathological if the utility argmax at lambda=100 differs or the solver-set Jaccard is <= 0.5,
benign if the argmax agrees and Jaccard >= 0.8 (the argmax respects the panel's availability
mask, so a model the panel never ran on a row can never be that row's best).  The headline
pathological/benign ratio interval is a query-level cluster bootstrap (pairs weighted by the
multiplicities of their two endpoint rows); the pair-level interval is kept in metrics.json as
``ratio_path_over_benign``.  Distances in every space are divided by
that space's median cosine distance over random same-family pairs; m = normalised
augmented distance / normalised base distance.  Purity/ARI/NMI/silhouette of the argmax
label are computed within family.  No outcome enters any embedding.  --smoke restricts to
3,000 stratified rows and skips big40k unless --smoke-big40k.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rb.data import SUCCESS_THRESHOLD, load_panel                         # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, topic_str                    # noqa: E402
from rb.reps import find_rep, load_rep                                    # noqa: E402
from rb import pathological as P                                          # noqa: E402

EXP = "exp74_pairs"
LAMS = (0.0, 100.0, 1000.0)
SPACES = {  # name -> (registry rep name or None, direct path relative to features/<panel>/prompt/)
    "gte_l21": (None, "hidden/gte-qwen2-7b/L21_last.npy"),
    "reader_l21": ("computed/qwen7b-l21@prompt", None),
    "big40k": ("computed/big40k@prompt", None),
    "hash_bow": ("computed/hash-bow@prompt", None),
}


def load_space(name: str, panel) -> np.ndarray:
    rep, direct = SPACES[name]
    if rep is not None:
        x = load_rep(find_rep(PANEL, rep), panel.n, panel.keep)
    else:
        raw = np.load(ROOT / "features" / PANEL / "prompt" / direct, mmap_mode="r")
        x = np.asarray(raw[panel.keep] if panel.keep is not None and len(raw) == len(panel.keep) else raw, dtype=np.float32)
    assert x.shape[0] == panel.n, (name, x.shape)
    return P.unit_rows(x)


def block_slices(panel_name: str) -> dict[str, slice]:
    """Block layout of the big40k matrix: side["blocks"] is a list of {"block": name,
    "dim": width} dicts in stored order.  If the sidecar carries a "dims" field (the
    matrix's total column width), assert the summed block dims equal it; otherwise skip
    the assert (neither RouterBench nor LLMRouterBench sidecars carry a "shape" field,
    but both carry "dims": 40264 / 40263, observed 2026-09-22)."""
    side = json.load(open(ROOT / "features" / panel_name / "prompt" / "big40k.npy.json"))
    out, off = {}, 0
    for b in side["blocks"]:
        out[b["block"]] = slice(off, off + int(b["dim"])); off += int(b["dim"])
    width = side.get("dims")
    if width is not None:
        assert off == int(width), (panel_name, off, width)
    else:
        print(f"[exp74] block_slices({panel_name}): sidecar has no 'dims' field, skipping width assert (summed dims={off})", flush=True)
    return out


def curated_pairs(panel_name: str, spaces: list[str]) -> dict:
    """The 21 authored minimal pairs projected into big40k (hard_twins/big40k_blocks); slice the
    gte:output, gte_own:L21_last and qwen7b:L21_last blocks out of the same vector."""
    d = ROOT / "results" / "hard_twins"
    coll = json.load(open(d / "collection.json"))["items"]
    v = np.load(d / "big40k_blocks" / f"big40k__{panel_name}__raw.npy").astype(np.float32)   # 42 × 40k
    sl = block_slices(panel_name)
    pick = {"base": sl["gte:output"], "gte_l21": sl["gte_own:L21_last"], "reader_l21": sl["qwen7b:L21_last"], "big40k": slice(0, v.shape[1])}
    by_pair = {}
    for k, it in enumerate(coll):
        by_pair.setdefault(it["pair"], {})[it["arm"]] = k
    pairs = np.array([[a["easy"], a["hard"]] for a in by_pair.values() if {"easy", "hard"} <= set(a)])
    out = {"n_pairs": int(len(pairs))}
    ub = P.unit_rows(v[:, pick["base"]]); db = P.pair_cosine_distance(ub, pairs)
    out["base"] = {"twin_cos": float(1 - db.mean())}
    for s in spaces:
        if s not in pick:
            continue
        us = P.unit_rows(v[:, pick[s]]); ds = P.pair_cosine_distance(us, pairs)
        out[s] = {"twin_cos": float(1 - ds.mean()), "median_raw_ratio": float(np.median(ds / np.maximum(db, 1e-6)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--cos-min", type=float, default=0.90)
    ap.add_argument("--spaces", default="gte_l21,reader_l21,big40k,hash_bow")
    ap.add_argument("--n-norm-pairs", type=int, default=200_000); ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--smoke-big40k", action="store_true")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(74)
    out = Path(a.out) if a.out else RESULTS_ROOT / "pathological_pairs" / (EXP + ("_smoke" if a.smoke else ""))
    if out.exists() and not a.force:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(PANEL); fam = panel.dataset_index(); cells = topic_str(panel)
    rows = np.arange(panel.n)
    if a.smoke:
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(fam == f), size=min(3000 // len(np.unique(fam)), (fam == f).sum()), replace=False) for f in np.unique(fam)]))
    spaces = [s for s in a.spaces.split(",") if s]
    if a.smoke and not a.smoke_big40k:
        spaces = [s for s in spaces if s != "big40k"]

    # Outside --smoke, rows is exactly arange(panel.n): the fancy index would copy every
    # matrix (5.6 GB for big40k) for nothing, so take() is the identity there.
    take = (lambda arr: arr[rows]) if a.smoke else (lambda arr: arr)
    r = take(np.nan_to_num(panel.rewards, nan=0.0).astype(np.float64))
    c = take(np.nan_to_num(panel.costs, nan=0.0, posinf=0.0).astype(np.float64))
    avail = take(np.asarray(panel.available, dtype=bool))
    fam_r = take(fam); cells_r = take(cells)
    base = take(P.unit_rows(load_rep(find_rep(PANEL, "computed/gte-qwen2-7b@prompt"), panel.n, panel.keep)))

    print(f"[exp74] panel={PANEL} rows={len(rows)} spaces={spaces}", flush=True)
    with threadpool_limits(limits=8):
        pairs, cosb = P.candidate_pairs(base, fam_r, k=a.k, cos_min=a.cos_min)
    print(f"[exp74] candidate pairs: {len(pairs)}", flush=True)
    am = {lam: P.argmax_utility(r, c, lam, available=avail) for lam in LAMS}
    S = P.solver_sets(r, SUCCESS_THRESHOLD); jac = P.jaccard_pairs(S, pairs)
    flags = P.classify_pairs(pairs, am, jac)
    for k, v in flags.items():
        print(f"[exp74] {k}: {int(v.sum())}", flush=True)

    norm_pairs = P.random_pairs(fam_r, a.n_norm_pairs if not a.smoke else 20_000, rng, same_family=True)
    all_pairs = P.random_pairs(fam_r, a.n_norm_pairs if not a.smoke else 20_000, rng, same_family=False)
    d_base = P.pair_cosine_distance(base, pairs)
    med_base = float(np.median(P.pair_cosine_distance(base, norm_pairs)))
    med_base_all = float(np.median(P.pair_cosine_distance(base, all_pairs)))
    dist = {"base__d": d_base, "base__dnorm": d_base / med_base}
    metrics = {"base": {"med_same_family": med_base, "med_all": med_base_all}}
    label = am[100.0]
    metrics["base"]["purity"] = {}; metrics["base"]["clustering"] = {}
    for f in tqdm(np.unique(fam_r), desc="base purity"):
        sel = fam_r == f
        metrics["base"]["purity"][panel.datasets[f]] = P.knn_label_purity(base[sel], label[sel], np.zeros(sel.sum(), int), k=a.k)
        metrics["base"]["clustering"][panel.datasets[f]] = P.clustering_scores(base[sel], label[sel])

    for s in spaces:
        ts = time.time(); x = take(load_space(s, panel))
        d = P.pair_cosine_distance(x, pairs)
        med = float(np.median(P.pair_cosine_distance(x, norm_pairs))); med_all = float(np.median(P.pair_cosine_distance(x, all_pairs)))
        m = P.multiplier(d_base, d, med_base, med)
        dist[f"{s}__d"] = d; dist[f"{s}__dnorm"] = d / med; dist[f"{s}__m"] = m
        pa, be = flags["pathological"], flags["benign"]
        ok = bool(pa.any() and be.any())
        ratio = P.bootstrap_ratio_of_medians(m[pa], m[be], n_boot=a.n_boot, seed=0) if ok else (float("nan"),) * 3
        ratio_q = (P.bootstrap_ratio_of_medians_by_query(m[pa], pairs[pa], m[be], pairs[be], len(rows),
                                                         n_boot=a.n_boot, seed=0) if ok else (float("nan"),) * 3)
        ms = {"med_same_family": med, "med_all": med_all,
              "pathological": {"n": int(pa.sum()), "median_m": float(np.median(m[pa])) if pa.any() else float("nan")},
              "benign": {"n": int(be.sum()), "median_m": float(np.median(m[be])) if be.any() else float("nan")},
              "ratio_path_over_benign": list(ratio), "ratio_path_over_benign_by_query": list(ratio_q),
              "purity": {}, "clustering": {}}
        for f in tqdm(np.unique(fam_r), desc=f"{s} purity"):
            sel = fam_r == f
            ms["purity"][panel.datasets[f]] = P.knn_label_purity(x[sel], label[sel], np.zeros(sel.sum(), int), k=a.k)
            ms["clustering"][panel.datasets[f]] = P.clustering_scores(x[sel], label[sel])
        metrics[s] = ms
        print(f"[exp74] {s}: median m path={ms['pathological']['median_m']} benign={ms['benign']['median_m']} "
              f"ratio(pair-level)={ratio[0]:.3f} [{ratio[1]:.3f},{ratio[2]:.3f}] "
              f"ratio(query-level)={ratio_q[0]:.3f} [{ratio_q[1]:.3f},{ratio_q[2]:.3f}] ({time.time()-ts:.0f}s)", flush=True)
        del x

    # The expensive per-space work is done: persist the pair table, the distance arrays and a
    # first result.json/metrics.json BEFORE the optional curated-21 block, so a missing or
    # malformed hard_twins artefact can never throw away hours of computation.
    import pandas as pd
    df = pd.DataFrame({"i": rows[pairs[:, 0]], "j": rows[pairs[:, 1]], "family": [panel.datasets[f] for f in fam_r[pairs[:, 0]]],
                       "cell_i": cells_r[pairs[:, 0]], "cell_j": cells_r[pairs[:, 1]], "cos_base": cosb, "jaccard": jac})
    for lam in LAMS:
        df[f"argmax{int(lam)}_i"] = am[lam][pairs[:, 0]]; df[f"argmax{int(lam)}_j"] = am[lam][pairs[:, 1]]
    for k, v in flags.items():
        df[k] = v
    df.to_csv(out / "pairs.csv", index=False)
    np.savez_compressed(out / "distances.npz", spaces=np.array(spaces), **dist)
    settings = {"k": a.k, "cos_min": a.cos_min, "lambda_head": 100.0, "jac_path": 0.5, "jac_benign": 0.8,
                "n_norm_pairs": int(len(norm_pairs)), "n_boot": a.n_boot, "smoke": a.smoke, "rows_used": int(len(rows))}

    def write_report() -> dict:
        rep = {**panel.provenance(), "experiment": EXP, "representations": ["computed/gte-qwen2-7b@prompt (base)"] + [f"{s}: {SPACES[s][0] or SPACES[s][1]}" for s in spaces],
               "settings": settings, "counts": {k: int(v.sum()) for k, v in flags.items()} | {"candidates": int(len(pairs))},
               "metrics": metrics, "seconds": time.time() - t0}
        (out / "result.json").write_text(json.dumps(rep, indent=2, default=float))
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
        return rep

    write_report()
    try:
        metrics["curated21"] = curated_pairs(PANEL, spaces)
    except (FileNotFoundError, KeyError) as e:
        metrics["curated21"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[exp74] curated21 skipped: {type(e).__name__}: {e}", flush=True)
    rep = write_report()

    md = [f"# exp74 — pathological pairs ({PANEL}{' SMOKE' if a.smoke else ''})", "",
          f"Rows {len(rows)}; candidates {len(pairs)} (k={a.k}, cos ≥ {a.cos_min}); " + ", ".join(f"{k} {int(v.sum())}" for k, v in flags.items()), "",
          "Ratio interval: query-level cluster bootstrap; pair-level in metrics.json (`ratio_path_over_benign`).", "",
          "space | median m (pathological) | median m (benign) | ratio [95% CI, query-level] | mean purity base → aug", "---|---|---|---|---"]
    for s in spaces:
        ms = metrics[s]; pb = np.mean(list(metrics["base"]["purity"].values())); ps = np.mean(list(ms["purity"].values()))
        rq = ms["ratio_path_over_benign_by_query"]
        md.append(f"{s} | {ms['pathological']['median_m']:.3f} | {ms['benign']['median_m']:.3f} | {rq[0]:.3f} [{rq[1]:.3f}, {rq[2]:.3f}] | {pb:.3f} → {ps:.3f}")
    md += ["", f"Curated 21 pairs: {json.dumps(metrics['curated21'])}", "", f"Seconds: {rep['seconds']:.0f}"]
    (out / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
