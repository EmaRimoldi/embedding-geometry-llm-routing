"""exp18e - anatomy of SPROUT's within-cell difficulty structure. Per family: distribution of the solve fraction
(std, share solved by all 13, share solved by <= 3). For the largest cells of OpenHermes, MATH and MMLU-Pro: the two
embedding-only K-means clusters (sizes, mean solve fraction, distinguishing words, 3 example prompts each), example
prompts from the easiest and hardest quartiles with their nearest neighbour and its difficulty, and, for MATH, the
official Level: solve fraction per level, linear probe of the level from GTE inside each type, and whether the two
clusters align with level. Writes results/exp18e_sprout_difficulty_anatomy/."""
from __future__ import annotations
import collections, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = "exp18e_sprout_difficulty_anatomy"
CELLS = ["openhermes/chemistry_materials", "openhermes/biology_medicine", "openhermes/computer_science_programming", "openhermes/mathematics_statistics", "openhermes/everyday_life_general_knowledge", "math/algebra", "math/intermediate_algebra", "math/geometry", "mmlupro/math", "mmlupro/physics", "mmlupro/law", "mmlupro/psychology"]
STOP = set("the a an of to in and or for is are with that this on by as at from be it its which what how can does do you your their they its into using between each all any not if than then when where who whom".split())


def words(texts, other, top=10):
    def cnt(ts):
        c = collections.Counter()
        for t in ts: c.update(set(w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", t.lower()) if w not in STOP))
        return c
    a, b = cnt(texts), cnt(other); na, nb = len(texts), len(other)
    sc = {w: (a[w] / na) - (b.get(w, 0) / nb) for w in a if a[w] >= 0.03 * na}; return [w for w, _ in sorted(sc.items(), key=lambda t: -t[1])[:top]]


def main():
    D = S.load(); r, fam, fn, P = D["r"], D["fam"], D["fnames"], D["prompts"]; F, n = len(fn), D["n"]; xu = S.unit(D["x"]); s = (r >= 0.5).mean(1); names = D["names"]
    res = S.provenance(D); res["experiment"] = EXP; md = [f"# {EXP} (SPROUT)", "", "## Solve-fraction distribution per family (13 models)", "", "family | mean | std | solved by all 13 | solved by <= 3 | std within cells (mean over cells >= 50)", "---|---|---|---|---|---"]
    _, cell = np.unique(D["cell_str"], return_inverse=True); res["families"] = {}
    for g, f in enumerate(fn):
        m = fam == g; cs = [s[m & (cell == k)].std() for k in np.unique(cell[m]) if (m & (cell == k)).sum() >= 50]
        res["families"][f] = {"mean": float(s[m].mean()), "std": float(s[m].std()), "all13": float((s[m] == 1).mean()), "le3": float((s[m] <= 3 / 13).mean()), "within_cell_std": float(np.mean(cs))}
        md.append(f"{f} | {s[m].mean():.3f} | {s[m].std():.3f} | {(s[m]==1).mean():.3f} | {(s[m]<=3/13).mean():.3f} | {np.mean(cs):.3f}")
    md.append(""); res["cells"] = {}
    for cname in CELLS:
        idx = np.where(D["cell_str"] == cname)[0]
        if len(idx) < 100: continue
        km = KMeans(2, n_init=5, random_state=0).fit(xu[idx]); lab = km.labels_; out = {"rows": int(len(idx)), "solve_mean": float(s[idx].mean()), "solve_std": float(s[idx].std()), "clusters": []}
        md += [f"## {cname} ({len(idx)} rows; solve fraction mean {s[idx].mean():.3f}, std {s[idx].std():.3f})", ""]
        # per-model solve rate difference between clusters
        for j in range(2):
            mj = lab == j; o = lab != j; sr = (r[idx][mj] >= 0.5).mean(0); so = (r[idx][o] >= 0.5).mean(0); dif = sr - so; top_m = [f"{names[k].replace('wxai-','').replace('-instruct','')} {dif[k]:+.2f}" for k in np.argsort(dif)[:3]]
            ex = [P[idx[mj]][i][:220].replace("\n", " ") for i in np.random.default_rng(j).choice(mj.sum(), 3, replace=False)]
            cl = {"size": int(mj.sum()), "solve_mean": float(s[idx][mj].mean()), "words": words(P[idx[mj]], P[idx[o]]), "models_most_below_other_cluster": top_m, "examples": ex}; out["clusters"].append(cl)
            md += [f"- cluster {j}: {mj.sum()} rows, solve fraction {s[idx][mj].mean():.3f}; distinctive words: {', '.join(cl['words'])}; models losing most vs the other cluster: {', '.join(top_m)}"] + [f"    * {e}" for e in ex]
        q1, q3 = np.percentile(s[idx], 25), np.percentile(s[idx], 75); out["quartiles"] = {}
        nb = S.knn_idx(xu[idx], xu[idx], 2, exclude_self=True)[:, 0]
        for tag, sel in (("easiest quartile", np.where(s[idx] >= q3)[0]), ("hardest quartile", np.where(s[idx] <= q1)[0])):
            pick = np.random.default_rng(1).choice(sel, 3, replace=False); exs = []
            for i in pick:
                exs.append({"solve": float(s[idx][i]), "prompt": P[idx[i]][:240].replace("\n", " "), "nn_solve": float(s[idx][nb[i]]), "nn_cos": float(xu[idx[i]] @ xu[idx[nb[i]]]), "nn_prompt": P[idx[nb[i]]][:160].replace("\n", " ")})
            out["quartiles"][tag] = exs; md += [f"- {tag} (solve {'>=' if 'easiest' in tag else '<='} {q3 if 'easiest' in tag else q1:.2f}), examples with nearest neighbour:"] + [f"    * [{e['solve']:.2f}] {e['prompt']}\n        nn [{e['nn_solve']:.2f}, cos {e['nn_cos']:.2f}] {e['nn_prompt']}" for e in exs]
        md.append(""); res["cells"][cname] = out; print(cname, flush=True)
    # MATH levels
    rows = [json.loads(l) for l in open(S.ROOT / "panel" / "rows.jsonl")]; bykey = {r_["key"]: r_["dataset_level"] for r_ in rows}
    L = np.array([int(re.sub(r"\D", "", bykey[k]) or 0) for k in D["keys"]]); mm = (fam == fn.index("math")) & (L >= 1); res["math_levels"] = {"solve_by_level": {int(l): float(s[mm & (L == l)].mean()) for l in range(1, 6)}, "corr_level_solve": float(np.corrcoef(L[mm], s[mm])[0, 1])}
    md += ["## MATH official difficulty level", "", "level | rows | solve fraction", "---|---|---"] + [f"{l} | {int((mm & (L==l)).sum())} | {s[mm & (L==l)].mean():.3f}" for l in range(1, 6)] + [f"corr(level, solve fraction) = {res['math_levels']['corr_level_solve']:+.3f}", ""]
    md += ["type | rows | probe acc (level from GTE, 5 classes) | majority | corr(level, solve) | two-cluster: level means | two-cluster: solve means", "---|---|---|---|---|---|---"]; res["math_levels"]["per_type"] = {}
    for t in sorted(set(D["topic"][mm])):
        idx = np.where(mm & (D["topic"] == t))[0]; y = L[idx]
        if len(idx) < 200 or np.bincount(y)[1:].min() < 2: continue
        tr, te = train_test_split(np.arange(len(idx)), test_size=0.3, random_state=0, stratify=y); sc = StandardScaler().fit(xu[idx][tr]); clf = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(xu[idx][tr]), y[tr]); acc = float((clf.predict(sc.transform(xu[idx][te])) == y[te]).mean()); maj = float(np.bincount(y).max() / len(y))
        lab = KMeans(2, n_init=5, random_state=0).fit_predict(xu[idx]); lm = [float(y[lab == j].mean()) for j in range(2)]; sm = [float(s[idx][lab == j].mean()) for j in range(2)]
        res["math_levels"]["per_type"][t] = {"rows": int(len(idx)), "probe_acc": acc, "majority": maj, "corr": float(np.corrcoef(y, s[idx])[0, 1]), "cluster_level_means": lm, "cluster_solve_means": sm}
        md.append(f"{t} | {len(idx)} | {acc:.3f} | {maj:.3f} | {np.corrcoef(y, s[idx])[0,1]:+.3f} | {lm[0]:.2f} / {lm[1]:.2f} | {sm[0]:.3f} / {sm[1]:.3f}")
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
