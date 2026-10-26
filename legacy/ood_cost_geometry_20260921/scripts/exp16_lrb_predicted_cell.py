"""exp16 - predicted-cell lookup with the LLM (all-topics) labels on LLMRouterBench at the rescaled lambdas.

The paper's "predicted-cell lookup, all topics" row (exp33) was stored only at lambda in {0, 1, 10, 100, 1000}.
Same recipe as exp20/exp33: a linear classifier (StandardScaler + multinomial logistic regression, C=1) on the
GTE prompt embedding, trained on each split's training rows, predicts the LLM cell of every test row; the
(family, topic) lookup (>= 50 training rows, family fallback) is then applied with the predicted cell. Five
splits (seeds 0-4), lambda in {0, .5, 1, 2, 5, 10}; true-cell lookup, kNN-100, best single and oracle from the
same splits for reference; classifier cell accuracy per split. Writes results/exp16_lrb_predicted_cell/.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import common as C
from rb.probes.sweep import split
from exp14_lrb_topic_labels import labels, lookup, PANEL, SEEDS

EXP = "exp16_lrb_predicted_cell"; LAMS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)


def main():
    D = C.load(PANEL); x, r, c, fam, fn = D["x"], D["r"], D["c"], D["fam"], D["fnames"]; F, n = len(fn), len(r); xu = C.unit(x)
    L = labels(D); _, cell = np.unique(L["llm"], return_inverse=True)
    per = {m: {f"{l:g}": {"qw": [], "macro": []} for l in LAMS} for m in ("true_cell", "predicted_cell", "knn100", "best_single", "oracle")}; acc = []
    for seed in SEEDS:
        tr, te = split(n, fam, seed); sc = StandardScaler().fit(x[tr]); clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x[tr]), cell[tr])
        pred = cell.copy(); pred[te] = clf.predict(sc.transform(x[te])); acc.append(float((pred[te] == cell[te]).mean()))
        nb = tr[C.knn_mean(xu[te], xu[tr], 100)]; ii = np.arange(len(te))
        for lam in LAMS:
            U = r - lam * c; Ute = U[te]
            ch = {"true_cell": lookup(U, tr, te, fam, cell, F)[0], "predicted_cell": lookup(U, tr, te, fam, pred, F)[0], "knn100": U[nb].mean(1).argmax(1), "best_single": np.full(len(te), int(U[tr].mean(0).argmax())), "oracle": Ute.argmax(1)}
            for m, ch_ in ch.items():
                u = Ute[ii, ch_]; per[m][f"{lam:g}"]["qw"].append(float(u.mean())); per[m][f"{lam:g}"]["macro"].append(float(np.mean([u[fam[te] == g].mean() for g in range(F)])))
        print("seed", seed, "cell acc", round(acc[-1], 4), flush=True)
    res = C.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMS), "classifier_cell_accuracy": acc, "results": {m: {L_: {"qw": float(np.mean(v["qw"])), "macro": float(np.mean(v["macro"])), "qw_minmax": [float(np.min(v["qw"])), float(np.max(v["qw"]))]} for L_, v in per[m].items()} for m in per}})
    md = [f"# {EXP} ({PANEL})", "", f"Predicted LLM cell from a linear probe on GTE (cell accuracy per split: " + ", ".join(f"{a:.3f}" for a in acc) + f"; mean {np.mean(acc):.3f}). Mean over 5 splits, query-weighted [macro].", "", "method | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
    for m in per: md.append(f"{m} | " + " | ".join(f"{res['results'][m][f'{l:g}']['qw']:.3f} [{res['results'][m][f'{l:g}']['macro']:.3f}]" for l in LAMS))
    C.save(C.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
