#!/usr/bin/env python
"""exp74b — pathological pairs measured with dimension-free distances.

Same pairs as exp74: for each query, its 10 nearest same-family rows in the emitted GTE
space with cosine >= 0.90; pathological if the utility argmax at lambda=100 differs (ties to
the cheaper model, unavailable models excluded) or the solver-set Jaccard is <= 0.5; benign
if the argmax agrees and the Jaccard is >= 0.8.

exp74 compared spaces through 1 - cos divided by the median over random same-family pairs.
That corrects scale but not concentration (in 40k dimensions distances bunch around their
median), so this experiment adds measures that do not depend on the dimension:

  * unit Euclidean distance ||x_i - x_j|| (a true metric) and its multiplier
    m_euc = (d_aug / median_aug) / (d_base / median_base);
  * percentile of the pair's distance among 20,000 random pairs of the SAME family in the
    SAME space (0 = closest, 1 = farthest);
  * neighbour rank: position of j among i's same-family rows and of i among j's; the mutual
    rank is their geometric mean; "separated" = neither is in the other's top 10.

Per augmented space the headline contrasts pathological against benign pairs: shift of the
percentile, shift of log10 mutual rank, and the ratio of median multipliers (both the old
1 - cos version and the Euclidean one), each with a query-level cluster bootstrap.  Ranks and
percentiles are identical for cosine, 1 - cos, angle and unit Euclidean distance (all
monotone in one another); only the multiplier depends on the choice.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from rb.data import SUCCESS_THRESHOLD, load_panel                         # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT                               # noqa: E402
from rb.reps import find_rep, load_rep                                    # noqa: E402
from rb import pathological as P                                          # noqa: E402
from exp74_pathological_pairs import SPACES, load_space                   # noqa: E402

EXP = "exp74b_pair_metrics"


def weighted_mean(v, w):
    t = w.sum()
    return float((v * w).sum() / t) if t > 0 else float("nan")


def boot_frac_diff(a, pa, b, pb, n_rows, n_boot, seed=0):
    """Query-level cluster bootstrap of mean(a) - mean(b) for 0/1 indicators."""
    est = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
    rng = np.random.default_rng(seed); dr = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows).astype(np.float64)
        dr.append(weighted_mean(a, w[pa[:, 0]] * w[pa[:, 1]]) - weighted_mean(b, w[pb[:, 0]] * w[pb[:, 1]]))
    dr = np.array(dr); dr = dr[np.isfinite(dr)]
    return (est, float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))) if len(dr) else (est, float("nan"), float("nan"))


def measure(x, fam, pairs, norm_pairs, rng):
    cos = P.pair_cosine(x, pairs)
    euc = P.pair_unit_euclidean(x, pairs)
    ref_cos = P.pair_cosine(x, norm_pairs)
    med_cosd = float(np.median(1.0 - ref_cos))
    med_euc = float(np.median(P.pair_unit_euclidean(x, norm_pairs)))
    pct = P.family_percentile(x, fam, pairs, cos, rng)
    with threadpool_limits(limits=8):
        r_ij, r_ji, n_f = P.neighbor_rank(x, fam, pairs)
    mutual = np.sqrt(r_ij.astype(np.float64) * r_ji)
    return {"cos": cos, "cosd_norm": (1.0 - cos) / med_cosd, "euc": euc, "euc_norm": euc / med_euc,
            "pct": pct, "r_ij": r_ij, "r_ji": r_ji, "n_f": n_f, "mutual_rank": mutual,
            "sep10": ((r_ij > 10) & (r_ji > 10)).astype(np.float64),
            "_med_cosd": med_cosd, "_med_euc": med_euc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--cos-min", type=float, default=0.90)
    ap.add_argument("--spaces", default="gte_l21,reader_l21,big40k,hash_bow")
    ap.add_argument("--n-norm-pairs", type=int, default=200_000); ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(741)
    out = RESULTS_ROOT / "pathological_pairs" / (EXP + ("_smoke" if a.smoke else ""))
    if out.exists() and not a.force:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    panel = load_panel(PANEL); fam = panel.dataset_index()
    rows = np.arange(panel.n)
    if a.smoke:
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(fam == f), size=min(3000 // len(np.unique(fam)), (fam == f).sum()), replace=False) for f in np.unique(fam)]))
    spaces = [s for s in a.spaces.split(",") if s and not (a.smoke and s == "big40k")]
    take = (lambda arr: arr[rows]) if a.smoke else (lambda arr: arr)
    r = take(np.nan_to_num(panel.rewards, nan=0.0).astype(np.float64))
    c = take(np.nan_to_num(panel.costs, nan=0.0, posinf=0.0).astype(np.float64))
    avail = take(np.asarray(panel.available, dtype=bool)); fam_r = take(fam)
    base = take(P.unit_rows(load_rep(find_rep(PANEL, "computed/gte-qwen2-7b@prompt"), panel.n, panel.keep)))
    n_rows = len(rows)

    with threadpool_limits(limits=8):
        pairs, _ = P.candidate_pairs(base, fam_r, k=a.k, cos_min=a.cos_min)
    am = {100.0: P.argmax_utility(r, c, 100.0, available=avail)}
    jac = P.jaccard_pairs(P.solver_sets(r, SUCCESS_THRESHOLD), pairs)
    flags = P.classify_pairs(pairs, am, jac)
    pa, be = flags["pathological"], flags["benign"]
    print(f"[exp74b] panel={PANEL} rows={n_rows} candidates={len(pairs)} pathological={int(pa.sum())} benign={int(be.sum())}", flush=True)
    norm_pairs = P.random_pairs(fam_r, a.n_norm_pairs if not a.smoke else 20_000, rng, same_family=True)

    per = {"base": measure(base, fam_r, pairs, norm_pairs, np.random.default_rng(1))}
    del base
    for s in tqdm(spaces, desc="spaces"):
        x = take(load_space(s, panel))
        per[s] = measure(x, fam_r, pairs, norm_pairs, np.random.default_rng(1)); del x
        print(f"[exp74b] {s}: done ({time.time()-t0:.0f}s)", flush=True)

    B = per["base"]; summ = {}
    for s in ["base"] + spaces:
        M = per[s]; d = {}
        for g, sel in (("pathological", pa), ("benign", be)):
            d[g] = {"n": int(sel.sum()), "median_pct": float(np.median(M["pct"][sel])) if sel.any() else float("nan"),
                    "median_mutual_rank": float(np.median(M["mutual_rank"][sel])) if sel.any() else float("nan"),
                    "frac_separated_top10": float(M["sep10"][sel].mean()) if sel.any() else float("nan")}
        if s != "base":
            m_cos = M["cosd_norm"] / np.maximum(B["cosd_norm"], 1e-6)
            m_euc = M["euc_norm"] / np.maximum(B["euc_norm"], 1e-6)
            dpct = M["pct"] - B["pct"]
            dlr = np.log10(M["mutual_rank"]) - np.log10(B["mutual_rank"])
            d["contrast"] = {
                "ratio_median_m_cos_old": P.bootstrap_ratio_of_medians_by_query(m_cos[pa], pairs[pa], m_cos[be], pairs[be], n_rows, a.n_boot),
                "ratio_median_m_euc": P.bootstrap_ratio_of_medians_by_query(m_euc[pa], pairs[pa], m_euc[be], pairs[be], n_rows, a.n_boot),
                "diff_median_delta_pct": P.bootstrap_median_diff_by_query(dpct[pa], pairs[pa], dpct[be], pairs[be], n_rows, a.n_boot),
                "diff_median_delta_log10_rank": P.bootstrap_median_diff_by_query(dlr[pa], pairs[pa], dlr[be], pairs[be], n_rows, a.n_boot),
                "diff_frac_separated_top10": boot_frac_diff(M["sep10"][pa], pairs[pa], M["sep10"][be], pairs[be], n_rows, a.n_boot),
                "median_delta_pct": {"pathological": float(np.median(dpct[pa])), "benign": float(np.median(dpct[be]))},
                "median_delta_log10_rank": {"pathological": float(np.median(dlr[pa])), "benign": float(np.median(dlr[be]))},
                "median_m_euc": {"pathological": float(np.median(m_euc[pa])), "benign": float(np.median(m_euc[be]))},
                "median_m_cos_old": {"pathological": float(np.median(m_cos[pa])), "benign": float(np.median(m_cos[be]))},
            }
        d["normaliser"] = {"median_1_minus_cos": M["_med_cosd"], "median_unit_euclidean": M["_med_euc"]}
        summ[s] = d

    import pandas as pd
    df = pd.DataFrame({"i": rows[pairs[:, 0]], "j": rows[pairs[:, 1]], "family": [panel.datasets[f] for f in fam_r[pairs[:, 0]]],
                       "pathological": pa, "benign": be, "jaccard": jac})
    for s, M in per.items():
        for k in ("cos", "euc", "pct", "r_ij", "r_ji", "mutual_rank"):
            df[f"{s}__{k}"] = M[k]
    df.to_csv(out / "pair_metrics.csv", index=False)
    fam_counts = df.groupby("family")[["pathological", "benign"]].sum().astype(int).to_dict(orient="index")
    rep = {**panel.provenance(), "experiment": EXP,
           "representations": ["computed/gte-qwen2-7b@prompt (base)"] + [f"{s}: {SPACES[s][0] or SPACES[s][1]}" for s in spaces],
           "settings": {"k": a.k, "cos_min": a.cos_min, "lambda_head": 100.0, "jac_path": 0.5, "jac_benign": 0.8,
                        "n_norm_pairs": int(len(norm_pairs)), "percentile_ref_pairs_per_family": 20_000,
                        "n_boot": a.n_boot, "smoke": a.smoke, "rows_used": n_rows},
           "counts": {"candidates": int(len(pairs)), "pathological": int(pa.sum()), "benign": int(be.sum()), "per_family": fam_counts},
           "summary": summ, "seconds": time.time() - t0}
    (out / "result.json").write_text(json.dumps(rep, indent=2, default=float))

    f3 = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    r3 = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    md = [f"# exp74b — pathological pairs, dimension-free measures ({PANEL}{' SMOKE' if a.smoke else ''})", "",
          f"Rows {n_rows}; candidates {len(pairs)}; pathological {int(pa.sum())}; benign {int(be.sum())}. Intervals: query-level cluster bootstrap ({a.n_boot}).", "",
          "Levels (median over pairs): percentile of distance among same-family random pairs; mutual neighbour rank; share separated (neither in the other's top 10).", "",
          "space | pct path | pct benign | rank path | rank benign | sep10 path | sep10 benign", "---|---|---|---|---|---|---"]
    for s in ["base"] + spaces:
        p_, b_ = summ[s]["pathological"], summ[s]["benign"]
        md.append(f"{s} | {p_['median_pct']:.4f} | {b_['median_pct']:.4f} | {p_['median_mutual_rank']:.1f} | {b_['median_mutual_rank']:.1f} | {p_['frac_separated_top10']:.3f} | {b_['frac_separated_top10']:.3f}")
    md += ["", "Contrasts, pathological minus benign (shift relative to the emitted GTE space); ratios are pathological / benign:", "",
           "space | Δpct (path − benign) | Δlog10 rank (path − benign) | Δ sep10 share | ratio median m (unit Euclidean) | ratio median m (old 1−cos)",
           "---|---|---|---|---|---"]
    for s in spaces:
        C = summ[s]["contrast"]
        md.append(f"{s} | {f3(C['diff_median_delta_pct'])} | {f3(C['diff_median_delta_log10_rank'])} | {f3(C['diff_frac_separated_top10'])} | {r3(C['ratio_median_m_euc'])} | {r3(C['ratio_median_m_cos_old'])}")
    md += ["", f"Per family (pathological, benign): {json.dumps(fam_counts)}", "", f"Seconds: {rep['seconds']:.0f}"]
    (out / "result.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
