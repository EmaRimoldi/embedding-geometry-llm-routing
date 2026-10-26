"""exp94: encoder-free degree-preserving null for model-specific structure.

Within every family-topic cell, Curveball trades randomise which models solve
which queries while preserving exactly (a) every query's number of solvers and
(b) every model's number of successes in that cell. Thus difficulty and model
strength are held fixed while query-model specialisation is destroyed.

The statistic is a fully discrete, held-out lookup: does the identity pattern of
the other M-1 solvers predict a hidden target model beyond cell and the count of
other solvers? No vector representation or semantic encoder is used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(Path(__file__).resolve().parent))
from rb.data import SUCCESS_THRESHOLD,load_panel  # noqa:E402
from rb.panelctx import PANEL,topic_str  # noqa:E402
from rb.probes.sweep import split  # noqa:E402
from common import base,out_dir,save  # noqa:E402

EXP="exp94_degree_preserving_null"; N_NULL=30; TRADES_PER_ROW=20; ALPHA=20.0; EPS=1e-6


def curveball_cell(A,rng,trades_per_row=TRADES_PER_ROW):
    """Binary matrix randomisation preserving every row and column sum exactly."""
    sets=[set(np.flatnonzero(row)) for row in A]
    n=len(sets)
    for _ in range(trades_per_row*n):
        i,j=rng.integers(0,n,size=2)
        if i==j: continue
        ai=sets[i]-sets[j]; aj=sets[j]-sets[i]
        u=list(ai|aj)
        if len(u)<2: continue
        rng.shuffle(u); ni=set(u[:len(ai)]); nj=set(u[len(ai):])
        common=sets[i]&sets[j]; sets[i]=common|ni; sets[j]=common|nj
    B=np.zeros_like(A)
    for i,s in enumerate(sets):
        if s: B[i,list(s)]=1
    assert np.array_equal(A.sum(0),B.sum(0)) and np.array_equal(A.sum(1),B.sum(1))
    return B


def randomise(Y,cell,tr,te,rng):
    # Randomise train and test portions separately, so fitted cell-model rates,
    # test row difficulties, and test model totals are all fixed exactly.
    Z=np.empty_like(Y); is_train=np.zeros(len(Y),bool); is_train[tr]=True
    for c in np.unique(cell):
        for side in (True,False):
            rows=np.flatnonzero((cell==c)&(is_train==side))
            Z[rows]=curveball_cell(Y[rows],rng) if len(rows)>1 else Y[rows]
    return Z


def bitmask(A):
    return (A.astype(np.int64)*(1<<np.arange(A.shape[1],dtype=np.int64))).sum(1)


def table_predict(ytr,coarse_tr,fine_tr,coarse_te,fine_te):
    # Coarse posterior shrinks to the global target rate. Fine posterior shrinks
    # to the corresponding coarse posterior. Keys are discrete throughout.
    g=(ytr.sum()+1)/(len(ytr)+2)
    ck={}
    for k in np.unique(coarse_tr):
        z=ytr[coarse_tr==k]; ck[int(k)]=(z.sum()+ALPHA*g)/(len(z)+ALPHA)
    pc=np.array([ck.get(int(k),g) for k in coarse_te])
    fk={}
    for k in np.unique(fine_tr):
        ix=fine_tr==k; c=int(coarse_tr[np.flatnonzero(ix)[0]]); prior=ck.get(c,g); z=ytr[ix]
        fk[int(k)]=(z.sum()+ALPHA*prior)/(len(z)+ALPHA)
    pf=np.array([fk.get(int(k),pc[i]) for i,k in enumerate(fine_te)])
    return np.clip(pc,EPS,1-EPS),np.clip(pf,EPS,1-EPS)


def score(Y,cell,fam,tr,te):
    M=Y.shape[1]; llc=[];llp=[];bc=[];bp=[]
    Pc=np.zeros((len(te),M)); Pp=np.zeros((len(te),M))
    for m in range(M):
        oth=[j for j in range(M) if j!=m]
        nt=Y[tr][:,oth].sum(1); ne=Y[te][:,oth].sum(1)
        coarse_tr=cell[tr]*12+nt; coarse_te=cell[te]*12+ne
        fine_tr=coarse_tr*(1<<(M-1))+bitmask(Y[tr][:,oth]); fine_te=coarse_te*(1<<(M-1))+bitmask(Y[te][:,oth])
        pc,pp=table_predict(Y[tr,m],coarse_tr,fine_tr,coarse_te,fine_te)
        Pc[:,m]=pc;Pp[:,m]=pp
        llc.append(log_loss(Y[te,m],pc,labels=[0,1]));llp.append(log_loss(Y[te,m],pp,labels=[0,1]))
        bc.append(brier_score_loss(Y[te,m],pc));bp.append(brier_score_loss(Y[te,m],pp))
    rows=np.arange(len(te)); uc=Y[te][rows,Pc.argmax(1)]; up=Y[te][rows,Pp.argmax(1)]
    return {"difficulty_logloss":float(np.mean(llc)),"pattern_logloss":float(np.mean(llp)),
        "logloss_gain":float(np.mean(llc)-np.mean(llp)),"difficulty_brier":float(np.mean(bc)),
        "pattern_brier":float(np.mean(bp)),"brier_gain":float(np.mean(bc)-np.mean(bp)),
        "difficulty_route_accuracy":float(uc.mean()),"pattern_route_accuracy":float(up.mean()),
        "routing_gain":float(up.mean()-uc.mean())}


def binary_gap(Y,cell,fam,tr,te):
    F=len(np.unique(fam));K=cell.max()+1; ft=fam[tr];fe=fam[te]
    ftab=np.vstack([Y[tr][ft==g].mean(0) for g in range(F)]); ctab=np.full((K,Y.shape[1]),np.nan)
    for c in np.unique(cell[tr]): ctab[c]=Y[tr][cell[tr]==c].mean(0)
    B=ctab[cell[te]].copy(); bad=~np.isfinite(B).all(1);B[bad]=ftab[fe[bad]]
    lookup=Y[te][np.arange(len(te)),B.argmax(1)].mean(); oracle=Y[te].max(1).mean()
    return float(lookup),float(oracle),float(oracle-lookup)


def main():
    t0=time.time();panel=load_panel(PANEL);R=np.nan_to_num(panel.rewards.astype(float),nan=0.0)
    Y=(R>=SUCCESS_THRESHOLD).astype(np.int8);fam=panel.dataset_index().astype(int);_,cell=np.unique(np.asarray(topic_str(panel),str),return_inverse=True)
    tr,te=split(panel.n,fam,0);tr=np.sort(tr);te=np.sort(te)
    observed=score(Y,cell,fam,tr,te); observed_gap=binary_gap(Y,cell,fam,tr,te)
    null=[]; null_gaps=[]
    for b in range(N_NULL):
        Z=randomise(Y,cell,tr,te,np.random.default_rng(10000+b));null.append(score(Z,cell,fam,tr,te));null_gaps.append(binary_gap(Z,cell,fam,tr,te))
        print(f"null {b+1}/{N_NULL} done ({time.time()-t0:.0f}s)",flush=True)
    keys=("logloss_gain","brier_gain","routing_gain")
    summary={k:{"observed":observed[k],"null_mean":float(np.mean([z[k] for z in null])),
        "null_ci95":np.percentile([z[k] for z in null],[2.5,97.5]).tolist(),
        "one_sided_p":float((1+sum(z[k]>=observed[k] for z in null))/(N_NULL+1))} for k in keys}
    invariant={"observed":{"lookup":observed_gap[0],"oracle":observed_gap[1],"gap":observed_gap[2]},
        "null_gap_min":float(min(z[2] for z in null_gaps)),"null_gap_max":float(max(z[2] for z in null_gaps))}
    result=base(panel,{"experiment":EXP,"encoder_free":True,"split":0,"n_null":N_NULL,"trades_per_row":TRADES_PER_ROW,
        "smoothing_alpha":ALPHA,"observed":observed,"null":null,"summary":summary,"binary_gap_invariance":invariant,"seconds":time.time()-t0})
    md=[f"# {EXP} — degree-preserving encoder-free null — {panel.name}","",
        f"{N_NULL} Curveball randomisations within family-topic cells; every query solve count and every cell-model success total is preserved exactly. Held-out discrete lookup with α={ALPHA:g} predicts each hidden model from the identities of the other solvers, beyond their count.","",
        "statistic | observed | null mean | null 95% interval | one-sided p","---|---:|---:|---:|---:"]
    for k,z in summary.items(): md.append(f"{k} | {z['observed']:+.5f} | {z['null_mean']:+.5f} | [{z['null_ci95'][0]:+.5f}, {z['null_ci95'][1]:+.5f}] | {z['one_sided_p']:.3f}")
    md += ["","## Binary λ=0 oracle-gap invariance","",f"Observed lookup {observed_gap[0]:.4f}, oracle {observed_gap[1]:.4f}, gap {observed_gap[2]:.4f}.",
        f"Across null panels the gap ranges from {invariant['null_gap_min']:.4f} to {invariant['null_gap_max']:.4f}. The margins alone therefore reproduce this binary oracle gap."]
    save(out_dir(EXP),result,md)

if __name__=="__main__":main()
