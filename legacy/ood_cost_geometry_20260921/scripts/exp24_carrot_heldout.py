"""exp24 - CARROT-style per-query cost and performance regressors under held-out family (SPROUT six-family, RouterBench
seven-family). CARROT (Somerstep et al. 2025) embeds the prompt (text-embedding-3-small; here GTE-Qwen2-7B, the only
embedder available offline) and fits multi-label regressors from the embedding to each model's correctness and cost:
a 40-nearest-neighbour cosine regressor (their RouterBench setting) and a fine-tuned RoBERTa head (not reproducible
here; a multi-output ridge on standardised GTE stands in for the parametric variant). Decision: argmax_m p_hat[m] -
lambda * c_hat[m]. Variants isolate the cost estimator: soft-prior reward + CARROT cost. References: soft prior, soft
prior + true held-out cost vector (cost oracle), family ceiling, training best single, oracle. Usage: exp24 <panel>."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
panel = sys.argv[1]; os.environ["E4_PANEL"] = panel if panel != "sprout" else "routerbench-0shot"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = f"exp24_carrot_heldout_{'sprout' if panel == 'sprout' else 'rb'}"
LAMS = (0.0, 0.5, 5.0, 25.0, 50.0, 250.0, 500.0, 1000.0) if panel == "sprout" else (0.0, 1.0, 10.0, 50.0, 100.0, 500.0); LCOST = 50.0
METHODS = ("best_single_train", "carrot_knn40", "carrot_ridge", "prior_reward+knn40_cost", "prior_reward+ridge_cost", "soft_prior", "soft_prior+true_cost", "family_ceiling", "oracle")


def load():
    if panel == "sprout":
        D = S.load(); return D["r"], D["c"], D["fam"], D["fnames"], D["names"], D["x"]
    import common as Cm; D = Cm.load(panel); return D["r"], D["c"], D["fam"], D["fnames"], D["names"], D["x"]


def main():
    t0 = time.time(); r, c, fam, fn, names, x = load(); F, n = len(fn), len(r); xu = S.unit(x); del x
    per = {m: {f"{l:g}": [] for l in LAMS} for m in METHODS}; nte = []
    for g in range(F):
        te = np.where(fam == g)[0]; tr = np.where(fam != g)[0]; ii = np.arange(len(te)); src = [s for s in range(F) if s != g]; nte.append(len(te))
        nb = tr[S.knn_idx(xu[te], xu[tr], 40)]; r_knn, c_knn = r[nb].mean(1), c[nb].mean(1)
        sc = StandardScaler().fit(xu[tr]); Ztr, Zte = sc.transform(xu[tr]), sc.transform(xu[te]); rid_r = Ridge(alpha=10.0).fit(Ztr, r[tr]); rid_c = Ridge(alpha=10.0).fit(Ztr, c[tr]); r_rid, c_rid = rid_r.predict(Zte), np.maximum(rid_c.predict(Zte), 0)
        clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(Ztr, fam[tr]); post = clf.predict_proba(Zte)[:, [list(clf.classes_).index(s) for s in src]]
        R = np.stack([r[fam == s].mean(0) for s in src]); Cc = np.stack([c[fam == s].mean(0) for s in src]); pr = post @ R; pc = post @ Cc; rt, ct = r[te].mean(0), c[te].mean(0)
        for lam in LAMS:
            L = f"{lam:g}"; U = r - lam * c; Ute = U[te]
            ch = {"best_single_train": np.full(len(te), int(U[tr].mean(0).argmax())), "oracle": Ute.argmax(1), "carrot_knn40": (r_knn - lam * c_knn).argmax(1), "carrot_ridge": (r_rid - lam * c_rid).argmax(1),
                  "prior_reward+knn40_cost": (pr - lam * c_knn).argmax(1), "prior_reward+ridge_cost": (pr - lam * c_rid).argmax(1), "soft_prior": (pr - lam * pc).argmax(1), "soft_prior+true_cost": (pr - lam * ct[None, :]).argmax(1), "family_ceiling": np.full(len(te), int((rt - lam * ct).argmax()))}
            for m in METHODS: per[m][L].append(float(Ute[ii, ch[m]].mean()))
        print("fold", fn[g], round(time.time() - t0), flush=True)
    w = np.array(nte) / sum(nte); res = {"experiment": EXP, "panel": panel, "embedder": "gte-Qwen2-7B-instruct (stand-in for text-embedding-3-small)", "lambdas": list(LAMS), "families": fn, "n_test": nte,
                                          "results": {m: {L: {"pooled_per_query": float(np.dot(w, v)), "equal_weight": float(np.mean(v)), "per_family": dict(zip(fn, v))} for L, v in d.items()} for m, d in per.items()}}
    md = [f"# {EXP} ({panel}, held-out family; GTE stands in for text-embedding-3-small)", "", "pooled per query [equal weight over held-out families]", "", "method | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
    for m in METHODS: md.append(f"{m} | " + " | ".join(f"{res['results'][m][f'{l:g}']['pooled_per_query']:.3f} [{res['results'][m][f'{l:g}']['equal_weight']:.3f}]" for l in LAMS))
    L = f"{LCOST:g}"; md += ["", f"## per held-out family at λ={L}", "", "family | " + " | ".join(METHODS), "---|" + "---|" * len(METHODS)] + [f"{f} | " + " | ".join(f"{res['results'][m][L]['per_family'][f]:.3f}" for m in METHODS) for f in fn]
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
