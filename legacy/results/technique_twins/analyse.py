"""exp25: same solving technique, different surface story — does GTE group problems by technique or by story?
See experiments/exp25_technique_twins.yaml."""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = OUT_OVERRIDE or HERE / "results"


def unit(a): return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def loo_logistic(X, y):
    pred = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False; sc = StandardScaler().fit(X[m])
        pred.append(LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[m]), y[m]).predict(sc.transform(X[i:i + 1]))[0])
    return float(np.mean(np.array(pred) == y))


def analyse(E, tech, story, label):
    N = len(tech); S = E @ E.T; off = ~np.eye(N, dtype=bool)
    same_t = (tech[:, None] == tech[None, :]) & off; same_s = (story[:, None] == story[None, :]) & off; neither = off & ~same_t & ~same_s
    S2 = S.copy(); np.fill_diagonal(S2, -2); nn = S2.argmax(1)
    # per item: mean cos to same-technique items vs same-story items; how often same-technique wins
    mt = np.array([S[i, same_t[i]].mean() for i in range(N)]); ms = np.array([S[i, same_s[i]].mean() for i in range(N)])
    # kNN-1 LOO for technique / story (nearest neighbour's label)
    r = {"mean_cos_same_technique": float(S[same_t].mean()), "mean_cos_same_story": float(S[same_s].mean()), "mean_cos_neither": float(S[neither].mean()),
         "items_where_same_technique_closer_than_same_story": float((mt > ms).mean()),
         "nn1_same_technique": float((tech[nn] == tech).mean()), "nn1_same_story": float((story[nn] == story).mean()),
         "chance_nn1_technique": 3 / 31, "chance_nn1_story": 7 / 31,
         "loo_logistic_technique": loo_logistic(E, tech), "loo_logistic_story": loo_logistic(E, story), "chance_logistic_technique": 1 / 8, "chance_logistic_story": 1 / 4}
    # rank of the 3 same-technique items among the 31 candidates, per item (mean rank; perfect = 2.0)
    ranks = []
    for i in range(N):
        order = np.argsort(-S2[i]); pos = [int(np.where(order == j)[0][0]) + 1 for j in np.flatnonzero(same_t[i])]; ranks.append(np.mean(pos))
    r["mean_rank_of_same_technique_items"] = float(np.mean(ranks)); r["mean_rank_of_same_story_items"] = float(np.mean([np.mean([int(np.where(np.argsort(-S2[i]) == j)[0][0]) + 1 for j in np.flatnonzero(same_s[i])]) for i in range(N)]))
    # per-technique: mean within-technique cosine, and nn1 technique accuracy
    r["per_technique"] = {t: {"within_cos": float(S[np.ix_(tech == t, tech == t)][~np.eye((tech == t).sum(), dtype=bool)].mean()), "nn1_hits": float((tech[nn][tech == t] == t).mean())} for t in np.unique(tech)}
    r["per_story"] = {s: {"within_cos": float(S[np.ix_(story == s, story == s)][~np.eye((story == s).sum(), dtype=bool)].mean())} for s in np.unique(story)}
    return r


def main() -> None:
    coll = json.loads((HERE / "collection.json").read_text()); items = coll["items"]; N = len(items)
    tech = np.array([it["technique"] for it in items]); story = np.array([it["story"] for it in items]); texts = [it["text"] for it in items]
    E = unit(np.load(HERE / "embeddings" / f"{REP_NAME}__raw.npy").astype(np.float32))
    B = unit(HashingVectorizer(n_features=2 ** 14, alternate_sign=False, norm=None, ngram_range=(1, 2)).transform(texts).toarray().astype(np.float32))
    res = {"n": N, "gte": analyse(E, tech, story, "gte"), "bow": analyse(B, tech, story, "bow")}
    # panel pipelines placement
    panel = load_panel("routerbench-0shot"); n = panel.n
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = unit(x)
    rr = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); n_solved = (rr >= SUCCESS_THRESHOLD).sum(1).astype(np.float64); ds = panel.dataset_ids
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str); rec = rec[panel.keep] if panel.keep is not None else rec
    src = np.array(["wikihow" if re.search(r"\[(header|title|step|substeps)\]", str(t)) else "activitynet" for t in panel.prompts])
    cell_str = np.where(ds == "mmlu", np.array([s_.split(".")[0] for s_ in rec]), np.where(ds == "hellaswag", "hellaswag/" + src, ds))
    cell_names, cell = np.unique(cell_str, return_inverse=True); cell_names = [str(s) for s in cell_names]
    sc = StandardScaler().fit(x); xs = sc.transform(x); clf = LogisticRegression(C=1.0, max_iter=3000).fit(xs, cell); ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(xs, n_solved)
    nb = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute", n_jobs=8).fit(u).kneighbors(E)[1]
    es = sc.transform(E); pc = [cell_names[int(k)] for k in clf.predict(es)]; pn = ridge.predict(es)
    place = {"predicted_cell": pc, "ridge_n_solved": pn.tolist(), "knn100_majority_family": [collections.Counter(ds[row].tolist()).most_common(1)[0][0] for row in nb],
             "knn100_mean_n_solved": [float(n_solved[row].mean()) for row in nb]}
    # do same-technique problems share panel neighbours more than same-story ones? Jaccard of 100-NN sets
    sets = [set(row.tolist()) for row in nb]; jac = lambda a, b: len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    jt = np.mean([jac(i, j) for i in range(N) for j in range(N) if i != j and tech[i] == tech[j]]); js = np.mean([jac(i, j) for i in range(N) for j in range(N) if i != j and story[i] == story[j]]); jn = np.mean([jac(i, j) for i in range(N) for j in range(N) if i != j and tech[i] != tech[j] and story[i] != story[j]])
    place["panel_100nn_jaccard"] = {"same_technique": float(jt), "same_story": float(js), "neither": float(jn)}
    # ridge n_solved: variance explained by technique vs by story (one-way ANOVA R2)
    def r2_groups(v, g): return float(1 - sum(((v[g == k] - v[g == k].mean()) ** 2).sum() for k in np.unique(g)) / ((v - v.mean()) ** 2).sum())
    place["ridge_n_solved_R2_by_technique"] = r2_groups(pn, tech); place["ridge_n_solved_R2_by_story"] = r2_groups(pn, story)
    res["panel_placement"] = place
    OUT.mkdir(exist_ok=True); (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp25_technique_twins", **res}, indent=1, default=str))
    md = ["# exp25 — same technique, different story", "", f"{N} problems: 8 techniques × 4 stories. GTE unit vectors, cosine; hashed bag-of-words (1–2-grams) as the lexical control.", ""]
    for rep in ("gte", "bow"):
        r = res[rep]; md += [f"## [{rep}]", "", "statistic | same technique (different story) | same story (different technique) | neither", "---|---|---|---",
                             f"mean cosine | {r['mean_cos_same_technique']:.3f} | {r['mean_cos_same_story']:.3f} | {r['mean_cos_neither']:.3f}",
                             f"nearest neighbour shares the label (chance) | {r['nn1_same_technique']:.2f} ({r['chance_nn1_technique']:.2f}) | {r['nn1_same_story']:.2f} ({r['chance_nn1_story']:.2f}) | –",
                             f"mean rank of the 3 (technique) / 7 (story) matching items among 31 (perfect 2.0 / 4.0) | {r['mean_rank_of_same_technique_items']:.1f} | {r['mean_rank_of_same_story_items']:.1f} | –",
                             f"leave-one-out logistic accuracy (chance) | {r['loo_logistic_technique']:.2f} ({r['chance_logistic_technique']:.2f}) | {r['loo_logistic_story']:.2f} ({r['chance_logistic_story']:.2f}) | –",
                             f"items whose same-technique mean cosine exceeds their same-story mean cosine | {r['items_where_same_technique_closer_than_same_story']:.2f} | | ", "",
                             "technique | within-technique mean cosine | nearest-neighbour hit rate", "---|---|---"]
        md += [f"{t} | {v['within_cos']:.3f} | {v['nn1_hits']:.2f}" for t, v in r["per_technique"].items()]
        md += ["", "story | within-story mean cosine", "---|---"] + [f"{s} | {v['within_cos']:.3f}" for s, v in r["per_story"].items()] + [""]
    p = res["panel_placement"]
    md += ["## Placement by the panel-trained pipelines", "", f"predicted cell: {dict(collections.Counter(p['predicted_cell']))}; kNN-100 majority family: {dict(collections.Counter(p['knn100_majority_family']))}",
           f"ridge n_solved: mean {np.mean(p['ridge_n_solved']):.2f}, sd {np.std(p['ridge_n_solved']):.2f}; share of its variance explained by technique {p['ridge_n_solved_R2_by_technique']:.2f}, by story {p['ridge_n_solved_R2_by_story']:.2f}",
           f"Jaccard overlap of 100-NN panel sets: same technique {p['panel_100nn_jaccard']['same_technique']:.3f}, same story {p['panel_100nn_jaccard']['same_story']:.3f}, neither {p['panel_100nn_jaccard']['neither']:.3f}", "",
           "technique | story | predicted cell | ridge n_solved | kNN-100 family | kNN-100 mean n_solved", "---|---|---|---|---|---"]
    md += [f"{tech[i]} | {story[i]} | {pc[i]} | {pn[i]:.1f} | {p['knn100_majority_family'][i]} | {p['knn100_mean_n_solved'][i]:.1f}" for i in range(N)]
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
