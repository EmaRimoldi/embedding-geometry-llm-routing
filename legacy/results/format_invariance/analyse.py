"""exp24: same question, three formats (raw / MMLU template / GSM8K template) — does the embedding map by format or by content?
See experiments/exp24_format_invariance.yaml."""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME  # noqa: E402

HERE = Path(__file__).resolve().parent; OUT = OUT_OVERRIDE or HERE / "results"; FMTS = ("raw", "mmlu", "gsm")


def unit(a): return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def main() -> None:
    items = json.loads((HERE / "collection.json").read_text())["items"]; N = len(items)
    E = {f: np.load(HERE / "embeddings" / f"{REP_NAME}__{f}.npy").astype(np.float32) for f in FMTS}; Eu = {f: unit(E[f]) for f in FMTS}
    res = {"n_questions": N}; md = ["# exp24 — same question, three formats", "", f"{N} questions × 3 formats (raw / MMLU template / GSM8K template), GTE unit vectors, cosine.", ""]
    # ---- T1: cosine structure
    same_q = {f"{a}-{b}": float(np.mean(np.sum(Eu[a] * Eu[b], axis=1))) for i, a in enumerate(FMTS) for b in FMTS[i + 1:]}
    diff_same_fmt = {f: float((Eu[f] @ Eu[f].T)[~np.eye(N, dtype=bool)].mean()) for f in FMTS}
    diff_cross = {f"{a}-{b}": float((Eu[a] @ Eu[b].T)[~np.eye(N, dtype=bool)].mean()) for i, a in enumerate(FMTS) for b in FMTS[i + 1:]}
    res["T1"] = {"same_question_cross_format": same_q, "different_question_same_format": diff_same_fmt, "different_question_cross_format": diff_cross}
    md += ["## T1 — mean cosine", "", "pair | same question, different format | different question, same format | different question, cross format", "---|---|---|---"]
    for k in same_q: a, b = k.split("-"); md.append(f"{k} | {same_q[k]:.3f} | {a}: {diff_same_fmt[a]:.3f}, {b}: {diff_same_fmt[b]:.3f} | {diff_cross[k]:.3f}")
    # ---- T2: retrieval — for each text, nearest among all other 3N-1 texts: same question (other format) or another question (same format)?
    allE = np.vstack([Eu[f] for f in FMTS]); qid = np.tile(np.arange(N), 3); fid = np.repeat(np.arange(3), N); S = allE @ allE.T; np.fill_diagonal(S, -2)
    nn = S.argmax(1); same_question = (qid[nn] == qid); same_format = (fid[nn] == fid)
    per_fmt = {f: {"nn_is_same_question": float(same_question[fid == i].mean()), "nn_is_same_format_other_question": float((same_format & ~same_question)[fid == i].mean())} for i, f in enumerate(FMTS)}
    # cross-format matched retrieval: for text (q, a) find nearest among all texts of format b; rank of q
    xr = {}
    for i, a in enumerate(FMTS):
        for j, b in enumerate(FMTS):
            if a == b: continue
            Sab = Eu[a] @ Eu[b].T; ranks = (Sab > Sab[np.arange(N), np.arange(N)][:, None]).sum(1) + 1
            xr[f"{a}->{b}"] = {"top1": float((ranks == 1).mean()), "mean_rank_of_true_match": float(ranks.mean()), "n_candidates": N}
    res["T2"] = {"nearest_over_all_texts": per_fmt, "cross_format_matched_retrieval": xr}
    md += ["", "## T2 — retrieval", "", "For each text, its nearest neighbour among all other texts (3N − 1 candidates):", "", "format | nearest is the same question in another format | nearest is another question in the same format", "---|---|---"]
    md += [f"{f} | {v['nn_is_same_question']:.2f} | {v['nn_is_same_format_other_question']:.2f}" for f, v in per_fmt.items()]
    md += ["", "Cross-format matched retrieval (query in format a, candidates = the 24 questions in format b; chance top-1 = 0.04):", "", "a → b | top-1 | mean rank of the true match", "---|---|---"]
    md += [f"{k} | {v['top1']:.2f} | {v['mean_rank_of_true_match']:.1f}" for k, v in xr.items()]
    # ---- T3: remove the format mean (per-format centroid over the 24 questions) and redo retrieval
    R = {f: unit(E[f] - E[f].mean(0)) for f in FMTS}; xr2 = {}
    for a in FMTS:
        for b in FMTS:
            if a == b: continue
            Sab = R[a] @ R[b].T; ranks = (Sab > Sab[np.arange(N), np.arange(N)][:, None]).sum(1) + 1; xr2[f"{a}->{b}"] = {"top1": float((ranks == 1).mean()), "mean_rank_of_true_match": float(ranks.mean())}
    allR = np.vstack([R[f] for f in FMTS]); S2 = allR @ allR.T; np.fill_diagonal(S2, -2); nn2 = S2.argmax(1)
    res["T3"] = {"cross_format_matched_retrieval_after_removing_format_mean": xr2, "nearest_over_all_texts_is_same_question": float((qid[nn2] == qid).mean())}
    md += ["", "## T3 — after subtracting each format's mean vector (over the 24 questions)", "", "a → b | top-1 | mean rank", "---|---|---"] + [f"{k} | {v['top1']:.2f} | {v['mean_rank_of_true_match']:.1f}" for k, v in xr2.items()]
    md += [f"", f"Nearest neighbour over all texts is the same question: {res['T3']['nearest_over_all_texts_is_same_question']:.2f} (was {float(same_question.mean()):.2f} before removing the format mean).", ""]
    # ---- T4: placement by the trained pipelines
    panel = load_panel("routerbench-0shot"); n = panel.n
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = unit(x)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); n_solved = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
    fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str); rec = rec[panel.keep] if panel.keep is not None else rec
    src = np.array(["wikihow" if re.search(r"\[(header|title|step|substeps)\]", str(t)) else "activitynet" for t in panel.prompts])
    cell_str = np.where(ds == "mmlu", np.array([s_.split(".")[0] for s_ in rec]), np.where(ds == "hellaswag", "hellaswag/" + src, ds))
    cell_names, cell = np.unique(cell_str, return_inverse=True); cell_names = [str(s) for s in cell_names]
    km7 = KMeans(n_clusters=7, n_init=3, random_state=0).fit(u); km_fam = {k: collections.Counter(ds[km7.labels_ == k].tolist()).most_common(1)[0][0] for k in range(7)}
    sc = StandardScaler().fit(x); xs = sc.transform(x); clf = LogisticRegression(C=1.0, max_iter=3000).fit(xs, cell); ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(xs, n_solved)
    nnp = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute", n_jobs=8).fit(u); fam_cent = unit(np.vstack([x[fam == g].mean(0) for g in range(F)]))
    place = {}
    for f in FMTS:
        es = sc.transform(E[f]); nb = nnp.kneighbors(Eu[f])[1]
        place[f] = {"kmeans7_family": [km_fam[int(k)] for k in km7.predict(Eu[f])], "nearest_centroid": [names[int(i)] for i in (Eu[f] @ fam_cent.T).argmax(1)],
                    "knn100_majority_family": [collections.Counter(ds[row].tolist()).most_common(1)[0][0] for row in nb], "knn100_gsm_share": [float((ds[row] == "grade-school-math").mean()) for row in nb],
                    "knn100_mmlu_share": [float((ds[row] == "mmlu").mean()) for row in nb], "knn100_top_cell": [collections.Counter(cell_str[row].tolist()).most_common(1)[0][0] for row in nb],
                    "pred_cell": [cell_names[int(k)] for k in clf.predict(es)], "ridge_n_solved": ridge.predict(es).tolist(), "knn100_mean_n_solved": [float(n_solved[row].mean()) for row in nb]}
    res["T4"] = place
    md += ["## T4 — placement by the pipelines trained on the panel (share of the 24 questions)", "", "format | K-means-7 → family | nearest family centroid | kNN-100 majority family | mean GSM8K share in 100-NN | mean MMLU share | top cell (most common) | predicted cell (most common) | ridge n_solved mean | kNN-100 mean n_solved", "---|---|---|---|---|---|---|---|---|---"]
    for f in FMTS:
        p = place[f]; mc = lambda L: ", ".join(f"{k} {v}" for k, v in collections.Counter(L).most_common(2))
        md.append(f"{f} | {mc(p['kmeans7_family'])} | {mc(p['nearest_centroid'])} | {mc(p['knn100_majority_family'])} | {np.mean(p['knn100_gsm_share']):.2f} | {np.mean(p['knn100_mmlu_share']):.2f} | {mc(p['knn100_top_cell'])} | {mc(p['pred_cell'])} | {np.mean(p['ridge_n_solved']):.2f} ± {np.std(p['ridge_n_solved']):.2f} | {np.mean(p['knn100_mean_n_solved']):.2f}")
    md += ["", "Per question, ridge n_solved by format (raw / mmlu / gsm) and Spearman between formats:", ""]
    from scipy.stats import spearmanr
    rs = {f"{a}-{b}": float(spearmanr(place[a]["ridge_n_solved"], place[b]["ridge_n_solved"]).correlation) for i, a in enumerate(FMTS) for b in FMTS[i + 1:]}
    res["T4_ridge_rank_agreement"] = rs
    md.append("Spearman of ridge n_solved across formats: " + ", ".join(f"{k} {v:.2f}" for k, v in rs.items())); md.append("")
    md += ["id | raw | mmlu | gsm", "---|---|---|---"] + [f"{it['id']} | {place['raw']['ridge_n_solved'][i]:.1f} | {place['mmlu']['ridge_n_solved'][i]:.1f} | {place['gsm']['ridge_n_solved'][i]:.1f}" for i, it in enumerate(items)]
    OUT.mkdir(exist_ok=True); (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp24_format_invariance", **res}, indent=1, default=str))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
