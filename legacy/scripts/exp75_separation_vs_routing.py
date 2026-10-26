#!/usr/bin/env python
"""exp75 — separation gain versus routing gain, per (panel, family, representation).

x: separation gain from exp74 (10-NN argmax-label purity, augmented minus base).
y: routing gain of a frozen cosine kNN-100 router on the augmented space minus the same
router on the emitted GTE space, per family, mean over the five family-stratified 70/30
splits (routing_gain), with the population std over those five per-seed differences
(gain_sd) as the quantity to plot as an error bar, and a split-0-only paired query
bootstrap per family kept for reference (gain_split0, gain_lo, gain_hi -- NOT centred on
routing_gain; see "Fix round 2" below).  Pooled values for the reader block (exp36 wide)
and the (family, predicted argmax) key router (exp44) are copied from their result files
so the figure can show the head change beside the space change.
The kNN routine is exp67's (train-neighbour mean utility, argmax over models).

Note (2026-09-22, controller ruling 1): exp36_augmented_knn_wide's by_lambda[lam][method]
is NOT the flat {per_query, per_family_equal_weight} shape this script originally assumed
(verified against exp44, which does have that flat shape). It is instead
{"all": {"per_query": {"utility": [est, lo, hi], "reward": [...], "cost": [...]},
         "per_family_equal_weight": {"utility": [est, lo, hi], ...}},
 "disagreement": {...same shape, disagreement subset...}}
(checked on both panels; the 3-element list is ordered lo <= est <= hi, consistent with
this repo's (est, lo, hi) bootstrap convention -- not per-seed values, since len==3 while
len(seeds)==5). Per the ruling, we copy the scalar point estimate (index 0) of the
"utility" field, for the whole-panel ("all") population, for per_query and
per_family_equal_weight, for plain_knn100/all_knn100/cell_lookup. exp36's per_family
(family -> scalar utility) already matches the brief's assumed shape and is copied as-is.

Fix round 1 (2026-09-22, code review): the pooled per-query/equal-weight aggregation and
the per-family mean-over-seeds aggregation were each duplicated 2-3x inline; factored into
``pooled_stats`` and ``family_seed_means`` respectively, reused for the base block and
every per-space block. ``routing_gain`` is the mean, over seeds, of the per-seed (aug -
base) family-mean differences that ``family_seed_means`` returns for the augmented and
base blocks; ``gain_sd`` is the population std of those same per-seed differences;
``gain_split0`` is the point estimate of a separate, single-split (seed 0) paired
bootstrap (both added in "Fix round 2" below, which also explains why ``gain_split0`` is
kept apart from ``routing_gain``). ``base`` (the embedding matrix) is freed right after
its utilities are computed, mirroring the per-space ``del x``. The exp44 reference block
is now guarded by ``if lam in j["by_lambda"]``, matching the exp36 blocks.

Fix round 2 (2026-09-22, controller ruling): the figure was built with
``yerr=[routing_gain - gain_lo, gain_hi - routing_gain]``, which went negative for 2/68
smoke rows and crashed matplotlib. Root cause: ``routing_gain`` is the mean over all five
seeds' per-seed family-mean differences, while ``gain_lo``/``gain_hi`` are a bootstrap CI
built from seed 0's test split only -- the two are not centred on the same estimate, so an
error bar mixing them is not meaningful (and can be negative). table.csv now also carries
``gain_sd`` (population std, ddof=0, over the five per-seed family-mean differences that
``routing_gain`` is the mean of -- this is what the figure plots as a symmetric error bar)
and ``gain_split0`` (the point estimate ``P.bootstrap_mean_diff`` returns, i.e. what
``gain_lo``/``gain_hi`` are centred on; previously computed but discarded). ``gain_lo`` and
``gain_hi`` are unchanged.

Fix round 3 (2026-09-22, code review): ``family_mean`` became dead code in Fix round 2 --
``routing_gain``/``gain_sd`` are computed directly from the ``*_seeds`` lists that
``family_seed_means`` returns, so nothing called ``family_mean`` any more. Removed it;
``family_seed_means`` is the single per-seed-values helper.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from rb.data import load_panel                                             # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT                                # noqa: E402
from rb.probes.sweep import SEEDS, split                                   # noqa: E402
from rb.reps import find_rep, load_rep                                     # noqa: E402
from rb import pathological as P                                           # noqa: E402
from exp74_pathological_pairs import load_space                           # noqa: E402

EXP = "exp75_sep_vs_routing"; LAMS = (0.0, 100.0); K = 100


def topk(xq, xt, k):
    out = np.empty((len(xq), k), dtype=np.int32)
    for st in range(0, len(xq), 512):
        with threadpool_limits(limits=8):
            s = xq[st:st + 512] @ xt.T
        ind = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, ind, axis=1), axis=1)
        out[st:st + len(ind)] = np.take_along_axis(ind, order, axis=1)
    return out


def knn_utilities(x, r, c, fam, seeds, lams, k=K, avail=None):
    """Returns {seed: {lam: per-test-row utility array}} and the test index per seed.

    ``avail`` (N x M bool, optional) is the panel's availability mask: models that were not
    run on a test row are given -inf mean neighbour utility so the router can never "choose"
    a model whose reward/cost for that row is a fill value rather than a measurement.
    """
    res, tests = {}, {}
    for seed in tqdm(seeds, desc="splits"):
        tr, te = split(len(x), fam, seed); tr = np.sort(tr); te = np.sort(te); tests[seed] = te
        nb = topk(x[te], x[tr], k); res[seed] = {}
        for lam in lams:
            u = r - lam * c
            mu = u[tr][nb].mean(1)
            if avail is not None:
                mu = np.where(avail[te], mu, -np.inf)
            chosen = mu.argmax(1)
            res[seed][lam] = u[te, chosen]
    return res, tests


def pooled_stats(u: dict, lam: float, seeds: tuple, tests: dict, fam: np.ndarray) -> dict:
    """Pool a ``knn_utilities`` result into per-query and per-family-equal-weight means.

    Args:
        u: {seed: {lam: per-test-row utility array}}, as returned by ``knn_utilities``.
        lam: the utility lambda to pool.
        seeds: seeds to average over.
        tests: {seed: test-row index array}, as returned by ``knn_utilities``.
        fam: per-row family index of the full (pre-split) row set that ``tests`` indexes into.

    Returns:
        {"per_query_mean", "per_query_sd", "equal_weight_mean"} -- mean/sd over seeds of the
        raw per-query mean, and mean over seeds of the per-family-equal-weighted mean.
    """
    per_query = [u[s][lam].mean() for s in seeds]
    equal_weight = [np.mean([u[s][lam][fam[tests[s]] == f].mean() for f in np.unique(fam)]) for s in seeds]
    return {"per_query_mean": float(np.mean(per_query)), "per_query_sd": float(np.std(per_query)),
            "equal_weight_mean": float(np.mean(equal_weight))}


def family_seed_means(u: dict, seeds: tuple, tests: dict, fam: np.ndarray, lam: float, f: int) -> list[float]:
    """Per-seed mean utility for family ``f`` test rows, one value per seed, in seed order."""
    return [float(u[s][lam][fam[tests[s]] == f].mean()) for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--spaces", default="gte_l21,reader_l21,big40k")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(); t0 = time.time()
    out = RESULTS_ROOT / "pathological_pairs" / (EXP + ("_smoke" if a.smoke else ""))
    if out.exists() and not a.force:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    e74 = RESULTS_ROOT / "pathological_pairs" / ("exp74_pairs_smoke" if a.smoke else "exp74_pairs")
    m74 = json.load(open(e74 / "result.json"))["metrics"]
    panel = load_panel(PANEL); fam = panel.dataset_index(); n = panel.n
    r = np.nan_to_num(panel.rewards, nan=0.0).astype(np.float64); c = np.nan_to_num(panel.costs, nan=0.0, posinf=0.0).astype(np.float64)
    avail = np.asarray(panel.available, dtype=bool)
    rows = np.arange(n)
    if a.smoke:
        rng = np.random.default_rng(75)
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(fam == f), size=min(600, (fam == f).sum()), replace=False) for f in np.unique(fam)]))
        r, c, fam, avail = r[rows], c[rows], fam[rows], avail[rows]
    seeds = SEEDS[:2] if a.smoke else SEEDS
    spaces = [s for s in a.spaces.split(",") if s and not (a.smoke and s == "big40k")]
    # Outside --smoke rows is arange(n): the fancy index would copy the whole matrix
    # (5.6 GB for big40k) for nothing, so take() is the identity there.
    take = (lambda arr: arr[rows]) if a.smoke else (lambda arr: arr)
    base = take(P.unit_rows(load_rep(find_rep(PANEL, "computed/gte-qwen2-7b@prompt"), n, panel.keep)))
    ub, tests = knn_utilities(base, r, c, fam, seeds, LAMS, avail=avail); del base
    recs, pooled = [], {"base": {}}
    for lam in LAMS:
        pooled["base"][str(lam)] = pooled_stats(ub, lam, seeds, tests, fam)
    for sp in spaces:
        x = take(load_space(sp, panel)); ua, _ = knn_utilities(x, r, c, fam, seeds, LAMS, avail=avail); del x
        pooled[sp] = {}
        for lam in LAMS:
            pooled[sp][str(lam)] = pooled_stats(ua, lam, seeds, tests, fam)
            for f in np.unique(fam):
                name = panel.datasets[f]
                kb_seeds = family_seed_means(ub, seeds, tests, fam, lam, f); ka_seeds = family_seed_means(ua, seeds, tests, fam, lam, f)
                kb = float(np.mean(kb_seeds)); ka = float(np.mean(ka_seeds))
                gain_sd = float(np.std([ka_s - kb_s for ka_s, kb_s in zip(ka_seeds, kb_seeds)]))
                sel0 = fam[tests[seeds[0]]] == f
                gain_split0, lo, hi = P.bootstrap_mean_diff(ua[seeds[0]][lam][sel0], ub[seeds[0]][lam][sel0], n_boot=a.n_boot, seed=0)
                pb = m74["base"]["purity"].get(name); pa = m74.get(sp, {}).get("purity", {}).get(name)
                recs.append({"panel": PANEL, "family": name, "space": sp, "lambda": lam, "sep_purity_base": pb, "sep_purity_aug": pa,
                             "sep_gain": (None if pb is None or pa is None else pa - pb),
                             "knn100_base": kb, "knn100_aug": ka,
                             "routing_gain": ka - kb, "gain_sd": gain_sd, "gain_split0": float(gain_split0),
                             "gain_lo": lo, "gain_hi": hi, "n_test": int(sel0.sum())})
        print(f"[exp75] {sp}: pooled per-query knn100 λ=100 {pooled[sp]['100.0']['per_query_mean']:.4f} vs base {pooled['base']['100.0']['per_query_mean']:.4f}", flush=True)
    # copied references
    refs = {}
    e36 = RESULTS_ROOT / "difficulty_llm" / "exp36_augmented_knn_wide" / "result.json"
    if e36.exists():
        j = json.load(open(e36))
        # ruling 1: by_lambda[lam][method] is {"all": {...}, "disagreement": {...}}, each with
        # per_query/per_family_equal_weight -> {"utility": [est, lo, hi], ...}; we copy the
        # point estimate (index 0) of "utility" for the whole-panel ("all") population.
        refs["exp36_wide"] = {lam: {m: {"per_query_utility": j["by_lambda"][lam][m]["all"]["per_query"]["utility"][0],
                                        "per_family_equal_weight_utility": j["by_lambda"][lam][m]["all"]["per_family_equal_weight"]["utility"][0]}
                                    for m in ("plain_knn100", "all_knn100", "cell_lookup")} for lam in ("0.0", "100.0") if lam in j["by_lambda"]}
        refs["exp36_wide_per_family"] = {lam: {m: j["per_family"][lam][m] for m in ("plain_knn100", "all_knn100")} for lam in ("0.0", "100.0") if lam in j["per_family"]}
    e44 = RESULTS_ROOT / "difficulty_llm" / "exp44_argmax_key_null" / "result.json"
    if e44.exists():
        j = json.load(open(e44)); refs["exp44"] = {lam: {m: j["by_lambda"][lam][m] for m in ("family", "family+topic", "family+predicted_argmax", "DEPLOY::predfamily+predicted_argmax")} for lam in ("0.0", "100.0") if lam in j["by_lambda"]}
    df = pd.DataFrame(recs); df.to_csv(out / "table.csv", index=False)
    (out / "pooled.json").write_text(json.dumps({"knn100": pooled, "references": refs}, indent=2))
    rep = {**panel.provenance(), "experiment": EXP, "representations": ["computed/gte-qwen2-7b@prompt (base)"] + spaces,
           "settings": {"k": K, "lambdas": LAMS, "seeds": list(seeds), "smoke": a.smoke, "rows_used": int(len(rows)), "exp74_source": str(e74)}, "pooled": pooled, "references": refs, "seconds": time.time() - t0}
    (out / "result.json").write_text(json.dumps(rep, indent=2))
    md = [f"# exp75 — separation gain vs routing gain ({PANEL}{' SMOKE' if a.smoke else ''})", "", "space | λ | pooled kNN-100 per query (base → aug) | families with routing gain CI > 0 | mean sep gain", "---|---|---|---|---"]
    for sp in spaces:
        for lam in LAMS:
            d = df[(df.space == sp) & (df["lambda"] == lam)]
            md.append(f"{sp} | {lam:g} | {pooled['base'][str(lam)]['per_query_mean']:.4f} → {pooled[sp][str(lam)]['per_query_mean']:.4f} | {(d.gain_lo > 0).sum()}/{len(d)} | {d.sep_gain.mean() if d.sep_gain.notna().any() else float('nan'):.4f}")
    md += ["", "References copied: " + ", ".join(refs.keys()), "", f"Seconds: {rep['seconds']:.0f}"]
    (out / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
