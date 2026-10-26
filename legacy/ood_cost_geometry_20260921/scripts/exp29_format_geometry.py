"""exp29 - why does one space read the answer regime and another not? Geometry of FORMAT vs CONTENT in a representation.
Format classes by family (RB: multiple-choice = ARC, HellaSwag, MMLU, Winogrande; free-form = GSM8K, MBPP, MT-Bench.
LRB: multiple-choice = MMLU-Pro, GPQA; short answer = SimpleQA, AIME, LiveMathBench, HLE; long-form = LiveCodeBench,
ArenaHard, SWE-bench, tau2). Measures, per representation:
 (a) cross-family same-format share: among a query's 100 nearest neighbours FROM OTHER FAMILIES, the share with the same
     format class (chance = format class frequency outside the family) - does format organise the space beyond family?
 (b) K-means K = #format classes: purity w.r.t. format; and w.r.t. family for comparison;
 (c) held-out-family format probe: linear classifier format-class trained on the other families, accuracy on the held-out
     family (the OOD read-out the cost router needs), pooled over families;
 (d) family-centroid geometry: distance between family centroids of the same vs different format, in units of the
     within-family spread;
 (e) linear predictability between spaces: ridge from PCA-256(GTE) to the held-out coordinate direction of this space,
     i.e. how much of the verbosity direction of rep is linearly available in emitted GTE (R^2 on held-out families).
Usage: exp29 <panel> <rep> (rep as in exp27)."""
from __future__ import annotations
import re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import common as C
from sprout_common import knn_idx, unit
from exp05_verbosity_axis_router import axes, coords

EXP = "exp29_format_geometry"
FORMAT = {"routerbench-0shot": {"arc-challenge": "mc", "hellaswag": "mc", "mmlu": "mc", "winogrande": "mc", "grade-school-math": "free", "mbpp": "free", "mtbench": "free"},
          "llmrouterbench-performance-cost": {"mmlupro": "mc", "gpqa": "mc", "simpleqa": "short", "aime": "short", "livemathbench": "short", "hle": "short", "livecodebench": "long", "arenahard": "long", "swe-bench": "long", "tau2": "long"}}


def load_rep(D, panel_name, rep):
    if rep == "gte": return D["x"]
    h = np.load(C.E4 / "features" / panel_name / "prompt" / (rep + ".npy"), mmap_mode="r"); return np.asarray(h[D["panel"].keep] if h.shape[0] != len(D["r"]) else h, dtype=np.float32)


def main():
    panel_name, rep = sys.argv[1], sys.argv[2]; t0 = time.time(); D = C.load(panel_name); r, c, fam, fn = D["r"], D["c"], D["fam"], D["fnames"]; F, n = len(fn), len(r)
    fmap = FORMAT[panel_name]; fkey = {f: next((v for k, v in fmap.items() if k in f.lower()), "other") for f in fn}; classes = sorted(set(fkey.values())); fmt = np.array([classes.index(fkey[fn[g]]) for g in fam])
    X = load_rep(D, panel_name, rep); xu = unit(X); tag = re.sub(r"[^A-Za-z0-9]+", "_", rep).strip("_"); res = C.provenance(D); res.update({"experiment": EXP, "representation": rep, "format_classes": classes, "family_format": fkey})
    # (a) cross-family same-format share
    nb = knn_idx(xu, xu, 100, exclude_self=True); other = fam[nb] != fam[:, None]; same_fmt = fmt[nb] == fmt[:, None]
    has = other.sum(1) > 0; share = float((same_fmt & other).sum(1)[has].sum() / other.sum(1)[has].sum())
    chance = float(np.mean([((fmt == fmt[i]) & (fam != fam[i])).sum() / (fam != fam[i]).sum() for i in np.random.default_rng(0).choice(n, min(n, 3000), replace=False)]))
    per_fam_share = {fn[g]: float((same_fmt & other)[fam == g].sum() / max(other[fam == g].sum(), 1)) for g in range(F)}; cross_frac = float(other.mean())
    # (b) K-means with K = #formats
    lab = KMeans(len(classes), n_init=3, random_state=0).fit_predict(xu); pur_fmt = sum(np.bincount(fmt[lab == k]).max() for k in np.unique(lab)) / n; pur_fam = sum(np.bincount(fam[lab == k]).max() for k in np.unique(lab)) / n
    # (c) held-out-family format probe
    correct = 0; tot = 0; per_fam_probe = {}
    for g in range(F):
        te = fam == g; tr = ~te
        if len(np.unique(fmt[tr])) < 2: continue
        sc = StandardScaler().fit(xu[tr]); clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(sc.transform(xu[tr]), fmt[tr]); pred = clf.predict(sc.transform(xu[te])); acc = float((pred == fmt[te]).mean()); per_fam_probe[fn[g]] = acc; correct += (pred == fmt[te]).sum(); tot += te.sum()
    # (d) centroid geometry
    cen = np.stack([xu[fam == g].mean(0) for g in range(F)]); spread = np.mean([np.linalg.norm(xu[fam == g] - cen[g], axis=1).mean() for g in range(F)])
    dsame, ddiff = [], []
    for a in range(F):
        for b in range(a + 1, F): (dsame if fkey[fn[a]] == fkey[fn[b]] else ddiff).append(np.linalg.norm(cen[a] - cen[b]) / spread)
    # (e) is this space's verbosity direction linearly available in emitted GTE? (held-out families)
    Y = np.log(c + C.EPS); g_full = None; r2s = []
    if rep != "gte":
        G = D["x"]; pg = PCA(n_components=256, svd_solver="randomized", random_state=0).fit(G); Zg = pg.transform(G)
        for g in range(F):
            te = fam == g; tr = ~te; S = [s for s in range(F) if s != g]; rt, ct = C.family_tables(r, c, fam, S); alpha, B, _ = axes(np.log(ct + C.EPS), 1); v = coords(Y, alpha, B)[:, 0]
            sc = StandardScaler().fit(xu[tr]); w = Ridge(alpha=10.0).fit(sc.transform(xu[tr]), v[tr]); proj = w.predict(sc.transform(xu))   # this space's verbosity read-out (fitted on training families)
            m = Ridge(alpha=10.0).fit(Zg[tr], proj[tr]); ph = m.predict(Zg[te]); r2s.append(float(1 - ((proj[te] - ph) ** 2).sum() / max(((proj[te] - proj[te].mean()) ** 2).sum(), 1e-12)))
    res["measures"] = {"cross_family_same_format_share": share, "chance": chance, "cross_family_neighbour_fraction": cross_frac, "per_family_cross_same_format": per_fam_share, "kmeans_purity_format": float(pur_fmt), "kmeans_purity_family": float(pur_fam),
                       "heldout_format_probe_pooled": float(correct / max(tot, 1)), "heldout_format_probe_per_family": per_fam_probe, "centroid_dist_same_format": float(np.mean(dsame)) if dsame else None, "centroid_dist_diff_format": float(np.mean(ddiff)) if ddiff else None, "verbosity_direction_r2_from_gte_heldout": (float(np.mean(r2s)) if r2s else None)}
    m_ = res["measures"]; md = [f"# {EXP} ({panel_name}, rep = {rep})", "", f"format classes {classes}; family -> format {fkey}", "",
          f"- (a) cross-family neighbours: {cross_frac:.3f} of the 100 nearest are from other families; among those, same-format share {share:.3f} (chance {chance:.3f}); per family: " + ", ".join(f"{k} {v:.2f}" for k, v in per_fam_share.items()),
          f"- (b) K-means K={len(classes)}: purity w.r.t. format {pur_fmt:.3f}, w.r.t. family {pur_fam:.3f}",
          f"- (c) held-out-family format probe: pooled accuracy {m_['heldout_format_probe_pooled']:.3f}; per family: " + ", ".join(f"{k} {v:.2f}" for k, v in per_fam_probe.items()),
          f"- (d) family-centroid distance / within-family spread: same format {m_['centroid_dist_same_format']}, different format {m_['centroid_dist_diff_format']}",
          f"- (e) R^2 of this space's verbosity read-out predicted linearly from emitted GTE (PCA-256), held-out families: {m_['verbosity_direction_r2_from_gte_heldout']}"]
    C.save(C.RESULTS / EXP / panel_name / tag, res, md)


if __name__ == "__main__":
    main()
