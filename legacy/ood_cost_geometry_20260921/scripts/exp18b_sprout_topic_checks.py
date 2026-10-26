"""exp18b - robustness checks for the SPROUT topic geometry (Claim 2) and the neighbourhood structure.
(a) same-topic share of the k=10 nearest neighbours inside each family (leave-one-out) vs chance;
(b) linear-probe topic accuracy inside each family (standardised GTE, multinomial logistic, 70/30 split, seed 0) vs
    majority class; (c) K-means NMI with n_init=5 for MMLU-Pro and MATH (convergence check of exp18);
(d) nearest-neighbour cosine per family (median, share > 0.9 / 0.95 / 0.98) - near-duplicate structure.
Writes results/exp18b_sprout_topic_checks/."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = "exp18b_sprout_topic_checks"; import os; EXP = EXP + ("_v2" if os.environ.get("SPROUT_TOPICS") == "v2" else "")


def main():
    t0 = time.time(); D = S.load(); fam, fn, x = D["fam"], D["fnames"], D["x"]; F, n = len(fn), D["n"]; xu = S.unit(x); del x
    res = S.provenance(D); res["experiment"] = EXP; res["checks"] = {}; md = [f"# {EXP} (SPROUT)", "", "family | label set | rows | topics | same-topic share k=10 | chance | probe acc | majority | K-means NMI n_init=5", "---|---|---|---|---|---|---|---|---"]
    sets = [("mmlupro", "authoritative category", (fam == fn.index("mmlupro")) & (D["topic_src"] == "authoritative")), ("math", "authoritative type", (fam == fn.index("math")) & (D["topic_src"] == "authoritative")),
            ("openhermes", "Qwen subject (accepted)", (fam == fn.index("openhermes")) & (np.char.startswith(D["topic_src"].astype(str), "qwen"))), ("gpqa", "Qwen subject (accepted)", (fam == fn.index("gpqa")) & (np.char.startswith(D["topic_src"].astype(str), "qwen"))),
            ("ragbench", "Qwen subject (accepted)", (fam == fn.index("ragbench")) & (np.char.startswith(D["topic_src"].astype(str), "qwen"))), ("ragbench", "released sub-dataset", fam == fn.index("ragbench"))]
    for f, lab_name, mask in sets:
        rows_ = np.where(mask)[0]
        if len(rows_) < 100: continue
        lab = D["dataset"][rows_] if lab_name == "released sub-dataset" else D["topic"][rows_]; _, t = np.unique(lab, return_inverse=True); K = int(t.max()) + 1
        nb = S.knn_idx(xu[rows_], xu[rows_], 10, exclude_self=True); share = float((t[nb] == t[:, None]).mean()); p = np.bincount(t) / len(t); chance = float((p ** 2).sum())
        tr, te = train_test_split(np.arange(len(rows_)), test_size=0.3, random_state=0, stratify=(t if np.bincount(t).min() >= 2 else None)); sc = StandardScaler().fit(xu[rows_][tr]); clf = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(xu[rows_][tr]), t[tr]); acc = float((clf.predict(sc.transform(xu[rows_][te])) == t[te]).mean()); maj = float(p.max())
        km = float(nmi(t, KMeans(K, n_init=5, random_state=0).fit_predict(xu[rows_]))) if f in ("mmlupro", "math") else float("nan")
        res["checks"][f"{f}|{lab_name}"] = {"rows": int(len(rows_)), "topics": K, "same_topic_share_k10": share, "chance": chance, "probe_acc": acc, "majority": maj, "kmeans_nmi_ninit5": km}
        md.append(f"{f} | {lab_name} | {len(rows_)} | {K} | {share:.3f} | {chance:.3f} | {acc:.3f} | {maj:.3f} | {km:.3f}"); print(f, lab_name, time.time() - t0, flush=True)
    nb1 = S.knn_idx(xu, xu, 2, exclude_self=True)[:, 0]; cs = np.einsum("ij,ij->i", xu, xu[nb1]); res["nn_cosine"] = {}
    md += ["", "## nearest-neighbour cosine (leave-one-out, whole panel)", "", "family | median | share > 0.9 | > 0.95 | > 0.98 | nn in same family", "---|---|---|---|---|---"]
    for g, f in enumerate(fn):
        m = fam == g; v = cs[m]; res["nn_cosine"][f] = {"median": float(np.median(v)), "gt0.9": float((v > 0.9).mean()), "gt0.95": float((v > 0.95).mean()), "gt0.98": float((v > 0.98).mean()), "same_family": float((fam[nb1][m] == g).mean())}
        md.append(f"{f} | {np.median(v):.3f} | {(v > 0.9).mean():.3f} | {(v > 0.95).mean():.3f} | {(v > 0.98).mean():.3f} | {(fam[nb1][m] == g).mean():.3f}")
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
