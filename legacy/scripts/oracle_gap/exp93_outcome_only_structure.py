"""exp93: encoder-free structure in the query-by-model outcome matrix.

For each target model, hide that model's binary outcome and predict it from the
other M-1 outcomes on the same query.  No query text or embedding is used.
The key comparison is a leave-one-model-out difficulty baseline (cell and number
of other solvers) versus the identities of which other models solved.  All
prediction models are trained on outer-training queries and evaluated on held-out
queries.  The outcome-only router is privileged and diagnostic, not deployable.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb.data import SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, topic_str  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from common import base, out_dir, save, strat_auc  # noqa: E402

EXP = "exp93_outcome_only_structure"
MIN_CELL = 50
EPS = 1e-5


def clip(p): return np.clip(np.asarray(p, float), EPS, 1-EPS)


def cell_prob(ytr, ctr, cte, K):
    g = (ytr.sum() + 1) / (len(ytr) + 2)
    tab = np.full(K, g)
    for c in np.unique(ctr):
        z = ytr[ctr == c]
        if len(z) >= MIN_CELL: tab[c] = (z.sum() + 1) / (len(z) + 2)
    return tab[cte]


def features(cell, n_other, others, K, full):
    n = len(cell); C = np.eye(K, dtype=np.float32)[cell]
    D = np.eye(12, dtype=np.float32)[n_other]
    if not full: return np.column_stack([C, D])
    # Global outcome identities and cell-specific outcome identities. This can
    # represent different complementarity patterns in every family-topic cell.
    I = (C[:, :, None] * others[:, None, :]).reshape(n, K * others.shape[1])
    return np.column_stack([C, D, others, I]).astype(np.float32)


def fit_logit(Xtr, ytr, Xte):
    if ytr.min() == ytr.max(): return np.full(len(Xte), float(ytr[0]))
    m = LogisticRegression(C=1.0, max_iter=500, solver="liblinear").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def svd_predict(Ytr, ctr, Yte, cte, target, rank, K):
    out = np.zeros(len(Yte), float); obs = [j for j in range(Ytr.shape[1]) if j != target]
    global_mu = Ytr.mean(0)
    for c in np.unique(cte):
        a = Ytr[ctr == c]; rows = np.flatnonzero(cte == c)
        if len(a) < MIN_CELL: a = Ytr
        mu = a.mean(0) if len(a) else global_mu
        _, _, vt = np.linalg.svd(a - mu, full_matrices=False); V = vt[:rank].T
        A = V[obs]
        z = np.linalg.lstsq(A, (Yte[rows][:, obs] - mu[obs]).T, rcond=None)[0].T
        out[rows] = mu[target] + z @ V[target]
    return clip(out)


def agg(v, fam):
    return {"per_query": float(np.mean(v)),
            "per_family_equal_weight": float(np.mean([v[fam == g].mean() for g in np.unique(fam)]))}


def bootstrap_delta(d, fam, B=2000, seed=0):
    rng=np.random.default_rng(seed); groups=[np.flatnonzero(fam==g) for g in np.unique(fam)]
    pq=[]; ef=[]
    for _ in range(B):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]); pq.append(d[ix].mean())
        ef.append(np.mean([d[rng.choice(g,len(g),replace=True)].mean() for g in groups]))
    return {"point":float(d.mean()),"per_query_ci95":np.percentile(pq,[2.5,97.5]).tolist(),
            "equal_family_point":float(np.mean([d[g].mean() for g in groups])),"equal_family_ci95":np.percentile(ef,[2.5,97.5]).tolist()}


def main():
    t0=time.time(); panel=load_panel(PANEL)
    R=np.nan_to_num(panel.rewards.astype(float),nan=0.0); Y=(R>=SUCCESS_THRESHOLD).astype(np.int8)
    fam=panel.dataset_index().astype(int); _,cell=np.unique(np.asarray(topic_str(panel),str),return_inverse=True)
    K=int(cell.max()+1); M=panel.m; names=[str(x).split('/')[-1] for x in panel.models]
    runs=[]; boot={}
    for seed in SEEDS:
        tr,te=split(panel.n,fam,seed); tr=np.sort(tr); te=np.sort(te)
        P={k:np.zeros((len(te),M),float) for k in ("cell","difficulty","pattern","svd1","svd2","svd3")}
        per_model={}
        for m in range(M):
            oth=[j for j in range(M) if j!=m]
            ntr=Y[tr][:,oth].sum(1); nte=Y[te][:,oth].sum(1)
            P["cell"][:,m]=cell_prob(Y[tr,m],cell[tr],cell[te],K)
            xdtr=features(cell[tr],ntr,Y[tr][:,oth],K,False); xdte=features(cell[te],nte,Y[te][:,oth],K,False)
            xftr=features(cell[tr],ntr,Y[tr][:,oth],K,True); xfte=features(cell[te],nte,Y[te][:,oth],K,True)
            P["difficulty"][:,m]=fit_logit(xdtr,Y[tr,m],xdte)
            P["pattern"][:,m]=fit_logit(xftr,Y[tr,m],xfte)
            for rank in (1,2,3): P[f"svd{rank}"][:,m]=svd_predict(Y[tr],cell[tr],Y[te],cell[te],m,rank,K)
            strata=cell[te]*100+nte
            per_model[names[m]]={}
            for method in P:
                p=clip(P[method][:,m]); y=Y[te,m]
                wa=strat_auc(y,p,strata)[0]
                per_model[names[m]][method]={"auc":float(roc_auc_score(y,p)),"within_cell_and_leave_one_out_difficulty_auc":wa,
                    "log_loss":float(log_loss(y,p,labels=[0,1])),"brier":float(brier_score_loss(y,p))}
        routes={}; rows=np.arange(len(te))
        # Cell router fitted from training outcomes, same minimum-cell fallback.
        fam_tab=np.vstack([Y[tr][fam[tr]==g].mean(0) for g in range(len(np.unique(fam)))])
        cell_tab=np.full((K,M),np.nan)
        for c in np.unique(cell[tr]):
            z=Y[tr][cell[tr]==c]
            if len(z)>=MIN_CELL: cell_tab[c]=z.mean(0)
        B=cell_tab[cell[te]].copy(); bad=~np.isfinite(B).all(1); B[bad]=fam_tab[fam[te][bad]]
        choices={"cell_lookup":B.argmax(1),**{k:v.argmax(1) for k,v in P.items()}}
        choices["oracle"]=Y[te].argmax(1)
        for k,ch in choices.items(): routes[k]=agg(Y[te][rows,ch],fam[te])
        if seed==SEEDS[0]:
            base_u=Y[te][rows,choices["cell_lookup"]]
            for k in ("difficulty","pattern","svd1","svd2","svd3"):
                boot[f"{k}-cell_lookup"]=bootstrap_delta(Y[te][rows,choices[k]]-base_u,fam[te],seed=seed)
            boot["pattern-difficulty"]=bootstrap_delta(Y[te][rows,choices["pattern"]]-Y[te][rows,choices["difficulty"]],fam[te],seed=seed)
        runs.append({"seed":seed,"per_model":per_model,"routes":routes})
        print(f"seed {seed} done ({time.time()-t0:.0f}s)",flush=True)

    methods=("cell","difficulty","pattern","svd1","svd2","svd3")
    summary={}
    for method in methods:
        vals=[]
        for run in runs:
            for m in names: vals.append(run["per_model"][m][method])
        summary[method]={k:float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
    route_summary={k:{mode:float(np.mean([z["routes"][k][mode] for z in runs])) for mode in ("per_query","per_family_equal_weight")}
                   for k in runs[0]["routes"]}
    result=base(panel,{"experiment":EXP,"encoder_free":True,"privileged_other_model_outcomes_at_test":True,
        "target_definition":f"reward >= {SUCCESS_THRESHOLD}","methods":list(methods),"runs":runs,"summary":summary,
        "route_summary":route_summary,"bootstrap_split0":boot,"seconds":time.time()-t0})
    md=[f"# {EXP} — encoder-free outcome structure — {panel.name}","",
        "Each target model's outcome is hidden and predicted from the other models' outcomes on the same held-out query. Difficulty uses only cell and the number of other solvers. Pattern additionally uses their identities and cell-specific interactions. SVD is an unsupervised low-rank reconstruction. This is a privileged diagnostic, not a deployable router.","",
        "## Held-out entry prediction (mean over models and five splits)","",
        "method | AUC | AUC within cell × leave-one-out difficulty | log loss | Brier","---|---:|---:|---:|---:"]
    for k in methods:
        z=summary[k]; md.append(f"{k} | {z['auc']:.3f} | {z['within_cell_and_leave_one_out_difficulty_auc']:.3f} | {z['log_loss']:.4f} | {z['brier']:.4f}")
    md += ["","## Privileged λ=0 routing accuracy","","method | per query | equal family","---|---:|---:"]
    for k,z in route_summary.items(): md.append(f"{k} | {z['per_query']:.4f} | {z['per_family_equal_weight']:.4f}")
    md += ["","## Paired bootstrap, split 0","","comparison | per query [95% CI] | equal family [95% CI]","---|---|---"]
    for k,z in boot.items(): md.append(f"{k} | {z['point']:+.4f} [{z['per_query_ci95'][0]:+.4f}, {z['per_query_ci95'][1]:+.4f}] | {z['equal_family_point']:+.4f} [{z['equal_family_ci95'][0]:+.4f}, {z['equal_family_ci95'][1]:+.4f}]")
    save(out_dir(EXP),result,md)

if __name__=="__main__": main()
