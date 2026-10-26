"""exp18c - which SPROUT families merge (K-means K=6 confusion, seed 0) and where cross-family neighbours go
(k=100, leave-one-out): destination family shares, and for GPQA / OpenHermes physics-chemistry-biology rows the
topic of their cross-family neighbours (does topic carry across families?). Writes results/exp18c_sprout_family_confusion/."""
from __future__ import annotations
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
import sprout_common as S

EXP = "exp18c_sprout_family_confusion"


def main():
    D = S.load(); fam, fn = D["fam"], D["fnames"]; F, n = len(fn), D["n"]; xu = S.unit(D["x"]); topic = D["topic"]
    res = S.provenance(D); res["experiment"] = EXP; md = [f"# {EXP} (SPROUT)", ""]
    lab = KMeans(F, n_init=3, random_state=0).fit_predict(xu); C = np.zeros((F, F), int)
    for g in range(F): C[g] = np.bincount(lab[fam == g], minlength=F)
    res["kmeans_confusion"] = {fn[g]: C[g].tolist() for g in range(F)}
    md += ["## K-means K=6 (seed 0, n_init 3): rows of each family per cluster (share of the family)", "", "family | " + " | ".join(f"c{k}" for k in range(F)), "---|" + "---|" * F] + [f"{fn[g]} | " + " | ".join(f"{C[g,k]/C[g].sum():.2f}" for k in range(F)) for g in range(F)] + [""]
    nb = S.knn_idx(xu, xu, 100, exclude_self=True); nf = fam[nb]; res["knn100_destinations"] = {}
    md += ["## kNN-100 (leave-one-out): share of a family's neighbours in each family", "", "family | " + " | ".join(fn), "---|" + "---|" * F]
    for g in range(F):
        row = np.bincount(nf[fam == g].ravel(), minlength=F) / (fam == g).sum() / 100; res["knn100_destinations"][fn[g]] = row.tolist(); md.append(f"{fn[g]} | " + " | ".join(f"{v:.3f}" for v in row))
    md += ["", "## Topic across families: for rows of the given cell, topic distribution of their cross-family kNN-100 neighbours (top 4)", ""]
    res["cross_family_topics"] = {}
    for cell in ("gpqa/physics_astronomy", "gpqa/chemistry_materials", "gpqa/biology_medicine", "openhermes/physics_astronomy", "openhermes/mathematics_statistics", "mmlupro/math", "mmlupro/physics", "mmlupro/chemistry", "math/algebra", "ragbench/biology_medicine"):
        idx = np.where(D["cell_str"] == cell)[0]
        if len(idx) == 0: continue
        cnt = collections.Counter(); tot = 0
        for i in idx:
            m = fam[nb[i]] != fam[i]; tot += int(m.sum()); cnt.update(D["cell_str"][nb[i][m]].tolist())
        top = cnt.most_common(4); res["cross_family_topics"][cell] = {"rows": int(len(idx)), "cross_family_share": tot / (len(idx) * 100), "top": top}
        md.append(f"- {cell} ({len(idx)} rows): {tot/(len(idx)*100):.3f} of neighbours cross-family; " + ", ".join(f"{k} {v/max(tot,1):.2f}" for k, v in top))
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
