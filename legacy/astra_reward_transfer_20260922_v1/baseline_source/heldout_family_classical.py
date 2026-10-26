"""One leave-one-family-out fold for classical and Avengers-family routers."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

E4=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(E4/"src"))
from rb.data import load_panel
from rb.panelctx import topic_str
from rb.probes.sweep import LAMBDAS
from rb.reps import find_rep,load_rep

KS=(5,10,40,100); TOP=(1,3); ALPHAS=np.round(np.linspace(0,1,101),2); BETA=9.; MIN_CELL=50
def unit(x): return (x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)).astype(np.float32)
def weights(q,centers,top):
    d=1-q@centers.T; ids=np.argsort(d,axis=1)[:,:top]; dd=np.take_along_axis(d,ids,1); z=-BETA*dd; w=np.exp(z-z.max(1,keepdims=True)); return ids,w/w.sum(1,keepdims=True)
def rankscore(key):
    order=np.argsort(-key,axis=1,kind="stable"); ranks=np.empty_like(order); ranks[np.arange(len(key))[:,None],order]=np.arange(key.shape[1]); return 1/(ranks+1.)
def route(ids,w,rs): return (w[:,:,None]*rs[ids]).sum(1).argmax(1)
def cluster_stats(lab,idx,r,c,K):
    M=r.shape[1]; acc=np.array([r[idx[lab==k]].mean(0) if np.any(lab==k) else np.zeros(M) for k in range(K)]); cost=np.array([c[idx[lab==k]].mean(0) if np.any(lab==k) else np.zeros(M) for k in range(K)])
    lo=acc.min(1,keepdims=True); rg=acc.max(1,keepdims=True)-lo; an=np.where(rg>0,(acc-lo)/np.where(rg>0,rg,1),1); cm=cost.max(1,keepdims=True); cs=np.where(cm>0,1-cost/np.where(cm>0,cm,1),1); return acc,cost,an,cs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--panel",required=True); ap.add_argument("--family-index",type=int,required=True); a=ap.parse_args()
    assert os.environ.get("E4_PANEL")==a.panel
    panel=load_panel(a.panel); fam=panel.dataset_index(); F=len(panel.datasets); g=a.family_index; assert 0<=g<F
    name=str(panel.datasets[g]); n=panel.n; M=panel.m; tr=np.flatnonzero(fam!=g); te=np.flatnonzero(fam==g); ii=np.arange(len(te))
    x=load_rep(find_rep(a.panel,"computed/gte-qwen2-7b@prompt"),n,panel.keep).astype(np.float32); u=unit(x); r=np.nan_to_num(panel.rewards.astype(float),nan=0); c=np.nan_to_num(panel.costs.astype(float),nan=0)
    ts=np.asarray(topic_str(panel),dtype=object); _,cell=np.unique(ts,return_inverse=True); cellfam={q:int(fam[np.flatnonzero(cell==q)[0]]) for q in np.unique(cell)}
    # deterministic inner validation, stratified among the seen families
    rng=np.random.default_rng(20260920+g); val=np.concatenate([rng.choice(tr[fam[tr]==h],max(1,int(.15*np.sum(fam[tr]==h))),replace=False) for h in np.unique(fam[tr])]); fit=np.setdiff1d(tr,val)
    sc=StandardScaler().fit(x[tr]); pf=LogisticRegression(C=1,max_iter=3000,n_jobs=8).fit(sc.transform(x[tr]),fam[tr]).predict(sc.transform(x[te])); pc=LogisticRegression(C=1,max_iter=3000,n_jobs=8).fit(sc.transform(x[tr]),cell[tr]).predict(sc.transform(x[te]))
    nb=tr[NearestNeighbors(n_neighbors=100,metric="cosine",algorithm="brute",n_jobs=8).fit(u[tr]).kneighbors(u[te],return_distance=False)]
    # alpha selection on an outcome-held inner fold
    kin=KMeans(n_clusters=64,n_init=10,random_state=20260920+g).fit(u[fit]); ai,ci,ani,csi=cluster_stats(kin.labels_,fit,r,c,64); vi,vw={t:weights(u[val],kin.cluster_centers_,t) for t in TOP},{}
    chosen={t:{} for t in TOP}
    for t in TOP:
        ids,w=vi[t]
        ach=[route(ids,w,rankscore(al*ani+(1-al)*csi)) for al in ALPHAS]
        for lam in LAMBDAS:
            U=r-lam*c; vals=[U[val][np.arange(len(val)),ch].mean() for ch in ach]; chosen[t][float(lam)]=float(ALPHAS[int(np.argmax(vals))])
    km=KMeans(n_clusters=64,n_init=10,random_state=20260920+g).fit(u[tr]); acc,cost,an,cs=cluster_stats(km.labels_,tr,r,c,64); tw={t:weights(u[te],km.cluster_centers_,t) for t in TOP}
    out={}
    for lam in LAMBDAS:
        U=r-lam*c; Utr=U[tr]; methods={}
        for k in KS: methods[f"knn{k}"]=U[nb[:,:k]].mean(1).argmax(1)
        ftab={h:Utr[fam[tr]==h].mean(0).argmax() for h in np.unique(fam[tr])}; ctab={q:(Utr[cell[tr]==q].mean(0).argmax() if np.sum(cell[tr]==q)>=MIN_CELL else ftab[cellfam[q]]) for q in np.unique(cell[tr])}
        methods["classify_family_lookup"]=np.array([ftab[h] for h in pf]); methods["classify_cell_lookup"]=np.array([ctab[q] for q in pc]); methods["best_single_train"]=np.repeat(Utr.mean(0).argmax(),len(te)); methods["oracle"]=U[te].argmax(1); methods["in_domain_family_ceiling"]=np.repeat(U[te].mean(0).argmax(),len(te))
        key=np.array([U[tr][km.labels_==q].mean(0) if np.any(km.labels_==q) else np.zeros(M) for q in range(64)])
        for t in TOP:
            ids,w=tw[t]; methods[f"kmeans64_utility_top{t}"]=route(ids,w,rankscore(key)); al=chosen[t][float(lam)]; methods[f"avengerspro64_top{t}_inner_alpha"]=route(ids,w,rankscore(al*an+(1-al)*cs))
        out[f"{lam:g}"]={m:{"utility":float(U[te][ii,ch].mean()),"reward":float(r[te][ii,ch].mean()),"cost":float(c[te][ii,ch].mean()),"choice_counts":{str(j):int(np.sum(ch==j)) for j in np.unique(ch)}} for m,ch in methods.items()}
    dest=E4/"heldout_generalization_20260920"/"results"/a.panel/"family"; dest.mkdir(parents=True,exist_ok=True)
    rec={**panel.provenance(),"heldout_family":name,"family_index":g,"n_train":len(tr),"n_test":len(te),"inner_fit":len(fit),"inner_val":len(val),"lambdas":[float(q) for q in LAMBDAS],"chosen_avengers_alpha":chosen,"results":out}
    (dest/f"fold_{g:02d}_{name}.json").write_text(json.dumps(rec,indent=2)); print(json.dumps({"family":name,"n_test":len(te),"headline":{L:out[L] for L in ("0","10","100","1000")}},indent=2),flush=True)
if __name__=="__main__": main()
