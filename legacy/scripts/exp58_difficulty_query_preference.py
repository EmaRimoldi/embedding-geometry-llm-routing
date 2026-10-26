"""Query-level preference of same-difficulty peers over opposite peers.

Frozen GTE, family/topic conditional, endpoint quartiles fixed before drawing
peers. Distinct same-tail peer; 64 matched random draws per query. Bootstrap
unique queries within family and cell for confidence intervals.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rb.data import SUCCESS_THRESHOLD,load_panel
from rb.panelctx import PANEL,REP,topic_str
from rb.reps import find_rep,load_rep

def main():
    p=load_panel(PANEL); fam=p.dataset_index()
    names,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    x=load_rep(find_rep(PANEL,REP),p.n,p.keep).astype(np.float32)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    r=np.nan_to_num(p.rewards,nan=0.0)
    ys={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    report={"panel":PANEL,"representation":REP,"draws_per_query":64,"bootstrap":1000,
            "method":"Within-cell outcome quartiles; for each tail query, probability among 64 independent draws that cosine to distinct same-tail peer exceeds cosine to opposite-tail peer. Bootstrap queries within cells, then combine by query or family weights.","results":{}}
    for proxy,y in ys.items():
        rng=np.random.default_rng(90117); qs=[]; scores=[]; cs=[]; fs=[]
        for c,name in enumerate(names):
            ids=np.flatnonzero(cell==c)
            if len(ids)<60:continue
            lo,hi=np.quantile(y[ids],[.25,.75]);
            if lo==hi:continue
            low=ids[y[ids]<=lo];high=ids[y[ids]>=hi]
            low=low[y[low]<hi];high=high[y[high]>lo]
            if min(len(low),len(high))<15:continue
            for group,other in ((low,high),(high,low)):
                for batch in np.array_split(group,max(1,(len(group)+127)//128)):
                    aa=rng.choice(group,(len(batch),64));bb=rng.choice(other,(len(batch),64))
                    bad=aa==batch[:,None]
                    while bad.any():
                        aa[bad]=rng.choice(group,int(bad.sum()))
                        bad=aa==batch[:,None]
                    s=np.einsum('ijk,ik->ij',x[aa],x[batch],optimize=True)
                    t=np.einsum('ijk,ik->ij',x[bb],x[batch],optimize=True)
                    score=((s>t)+.5*(s==t)).mean(1)
                    qs.extend(batch.tolist());scores.extend(score.tolist())
                    cs.extend([c]*len(batch));fs.extend([int(fam[batch[0]])]*len(batch))
        q=np.asarray(qs);s=np.asarray(scores);cids=np.asarray(cs);fids=np.asarray(fs)
        groups=[np.flatnonzero(cids==c) for c in np.unique(cids)]
        # Cell-stratified query bootstrap with fixed original cell and family weights.
        cell_weights=np.array([len(g) for g in groups],dtype=float)
        boot_cell=np.stack([s[rng.choice(g,(1000,len(g)),replace=True)].mean(axis=1) for g in groups],axis=1)
        b_pooled=boot_cell@cell_weights/cell_weights.sum()
        # A family mean weights its tail queries, then families equally.
        bf=[]
        for f in np.unique(fids):
            ix=np.array([j for j,g in enumerate(groups) if fids[g[0]]==f])
            weights=cell_weights[ix]/cell_weights[ix].sum()
            bf.append(boot_cell[:,ix]@weights)
        b_equal=np.stack(bf,axis=1).mean(axis=1)
        report["results"][proxy]={"queries":len(q),"cells":len(groups),
           "per_query":float(s.mean()),"per_query_ci95":np.quantile(b_pooled,[.025,.975]).tolist(),
           "equal_family":float(np.mean([s[fids==f].mean() for f in np.unique(fids)])),
           "equal_family_ci95":np.quantile(b_equal,[.025,.975]).tolist(),
           "per_family":{p.datasets[int(f)]:{"n":int((fids==f).sum()),"preference":float(s[fids==f].mean())} for f in np.unique(fids)},
           "per_cell":{str(names[c]):{"n":len(g),"preference":float(s[g].mean())} for g in groups for c in [cids[g[0]]]}}
        print(proxy,{k:report["results"][proxy][k] for k in ("queries","cells","per_query","per_query_ci95","equal_family","equal_family_ci95")},flush=True)
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_query_preference.json")
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
