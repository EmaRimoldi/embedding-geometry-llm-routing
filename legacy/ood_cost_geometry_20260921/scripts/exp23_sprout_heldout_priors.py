"""exp23 - the soft source-family prior and its decomposition on SPROUT, six-family holdout (the exp01 protocol of this
folder). For each held-out family: class-balanced logistic posterior over the five training families from GTE; family
mean reward and mean cost tables from training rows. Routers: soft prior (posterior x utility table), hard prior (argmax
family), soft prior with the TRUE held-out family's mean-cost vector substituted (cost oracle), with its TRUE mean-reward
vector (reward oracle), with both (= family ceiling: the held-out family's own best model), best single of the training
rows, kNN-100 over training rows, oracle. lambda 0, 0.5, 5, 25, 50, 250; per fold, pooled per query, equal weight.
Writes results/exp23_sprout_heldout_priors/."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sprout_common as S

EXP = "exp23_sprout_heldout_priors"; LAMS = (0.0, 0.5, 5.0, 25.0, 50.0, 250.0, 500.0, 1000.0)
METHODS = ("best_single_train", "knn100", "hard_prior", "soft_prior", "soft_prior+true_cost", "soft_prior+true_reward", "family_ceiling", "oracle")


def main():
    t0 = time.time(); D = S.load(); r, c, fam, fn = D["r"], D["c"], D["fam"], D["fnames"]; F, n = len(fn), D["n"]; xu = S.unit(D["x"])
    res = S.provenance(D); res["experiment"] = EXP; res["lambdas"] = list(LAMS); per = {m: {f"{l:g}": [] for l in LAMS} for m in METHODS}; nte = []; post_mass = {}
    for g in range(F):
        te = np.where(fam == g)[0]; tr = np.where(fam != g)[0]; ii = np.arange(len(te)); src = [s for s in range(F) if s != g]; nte.append(len(te))
        sc = StandardScaler().fit(xu[tr]); clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(sc.transform(xu[tr]), fam[tr]); post = clf.predict_proba(sc.transform(xu[te])); order = [list(clf.classes_).index(s) for s in src]; post = post[:, order]
        post_mass[fn[g]] = dict(zip([fn[s] for s in src], post.mean(0).round(3).tolist())); nb = tr[S.knn_idx(xu[te], xu[tr], 100)]
        R = np.stack([r[fam == s].mean(0) for s in src]); Cc = np.stack([c[fam == s].mean(0) for s in src]); rt, ct = r[te].mean(0), c[te].mean(0)
        for lam in LAMS:
            L = f"{lam:g}"; U = r - lam * c; Ute = U[te]; T = R - lam * Cc
            ch = {"best_single_train": np.full(len(te), int(U[tr].mean(0).argmax())), "oracle": Ute.argmax(1), "knn100": U[nb].mean(1).argmax(1), "hard_prior": T[post.argmax(1)].argmax(1), "soft_prior": (post @ T).argmax(1),
                  "soft_prior+true_cost": (post @ R - lam * ct[None, :]).argmax(1), "soft_prior+true_reward": (rt[None, :] - lam * (post @ Cc)).argmax(1), "family_ceiling": np.full(len(te), int((rt - lam * ct).argmax()))}
            for m in METHODS: per[m][L].append(float(Ute[ii, ch[m]].mean()))
        print("fold", fn[g], round(time.time() - t0), flush=True)
    w = np.array(nte) / sum(nte); res["posterior_mass"] = post_mass
    res["results"] = {m: {L: {"pooled_per_query": float(np.dot(w, v)), "equal_weight": float(np.mean(v)), "per_family": dict(zip(fn, v))} for L, v in d.items()} for m, d in per.items()}
    md = [f"# {EXP} (SPROUT, six-family holdout)", "", "pooled per query [equal weight over held-out families]", "", "method | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
    for m in METHODS: md.append(f"{m} | " + " | ".join(f"{res['results'][m][f'{l:g}']['pooled_per_query']:.3f} [{res['results'][m][f'{l:g}']['equal_weight']:.3f}]" for l in LAMS))
    for lam in (5.0, 50.0):
        L = f"{lam:g}"; md += ["", f"## per held-out family at λ={L}", "", "family | " + " | ".join(METHODS), "---|" + "---|" * len(METHODS)] + [f"{f} | " + " | ".join(f"{res['results'][m][L]['per_family'][f]:.3f}" for m in METHODS) for f in fn]
    md += ["", "## mean posterior mass over training families, per held-out family", ""] + [f"- {k}: " + ", ".join(f"{a} {b:.2f}" for a, b in v.items()) for k, v in post_mass.items()]
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
