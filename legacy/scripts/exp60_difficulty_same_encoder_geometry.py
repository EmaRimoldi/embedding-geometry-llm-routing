"""Paired same-tail geometry in emitted GTE and its internal states.

Identical anchors, peer draws and outcome quartiles across representations.
The internal L21_last layer was selected on RouterBench before this comparison.
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
    p=load_panel(PANEL);fam=p.dataset_index()
    names,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    hroot=ROOT/"features"/PANEL/"prompt"/"hidden"/"gte-qwen2-7b"
    arrays={"emitted":load_rep(find_rep(PANEL,REP),p.n,p.keep).copy()}
    for layer in ("L21_last","L28_last"):
        a=np.load(hroot/(layer+".npy"),mmap_mode="r")
        arrays[layer]=np.array(a[p.keep] if p.keep is not None and len(a)==len(p.keep) else a,dtype=np.float32,copy=True)
    for key,a in arrays.items():
        assert a.shape==(p.n,3584),(key,a.shape)
        a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    r=np.nan_to_num(p.rewards,nan=0.0)
    targets={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    report={"panel":PANEL,"rep":REP,"draws_per_query":64,"bootstrap":1000,
            "method":"Identical within-cell outcome-quartile anchors and 64 matched same/opposite tail peer draws; compare cosine preference from GTE emitted, L21_last and L28_last vectors after unit normalization. Distinct same-tail peers. Query bootstrap stratified by cell.","results":{}}
    for proxy,y in targets.items():
        rng=np.random.default_rng(90117); qs=[]; cs=[]; fs=[];scores={key:[] for key in arrays}
        for c in range(len(names)):
            ids=np.flatnonzero(cell==c)
            if len(ids)<60:continue
            lo,hi=np.quantile(y[ids],[.25,.75])
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
                    for key,x in arrays.items():
                        a=np.einsum('ijk,ik->ij',x[aa],x[batch],optimize=True)
                        b=np.einsum('ijk,ik->ij',x[bb],x[batch],optimize=True)
                        scores[key].extend(((a>b)+.5*(a==b)).mean(1).tolist())
                    qs.extend(batch.tolist());cs.extend([c]*len(batch));fs.extend([int(fam[batch[0]])]*len(batch))
        q=np.asarray(qs);cids=np.asarray(cs);fids=np.asarray(fs);s={key:np.asarray(v) for key,v in scores.items()}
        groups=[np.flatnonzero(cids==c) for c in np.unique(cids)]
        weights=np.array([len(g) for g in groups],dtype=float)
        rng=np.random.default_rng(71717)
        bootstrap_ix=[rng.choice(g,(1000,len(g)),replace=True) for g in groups]
        data={"n_queries":len(q),"n_cells":len(groups),"representations":{},"paired_delta":{}}
        for key,v in s.items():
            cboot=np.stack([v[ix].mean(1) for ix in bootstrap_ix],axis=1)
            pooled=cboot@weights/weights.sum()
            family_boot=[]
            for f in np.unique(fids):
                ii=np.array([j for j,g in enumerate(groups) if fids[g[0]]==f])
                family_boot.append(cboot[:,ii]@(weights[ii]/weights[ii].sum()))
            eq=np.stack(family_boot,axis=1).mean(1)
            data["representations"][key]={"per_query":float(v.mean()),"ci95":np.quantile(pooled,[.025,.975]).tolist(),
                    "equal_family":float(np.mean([v[fids==f].mean() for f in np.unique(fids)])),
                    "equal_family_ci95":np.quantile(eq,[.025,.975]).tolist(),
                    "per_family":{p.datasets[int(f)]:float(v[fids==f].mean()) for f in np.unique(fids)}}
        for key in ("L21_last","L28_last"):
            diff=s[key]-s["emitted"]
            cboot=np.stack([diff[ix].mean(1) for ix in bootstrap_ix],axis=1)
            data["paired_delta"][key]={"per_query":float(diff.mean()),"ci95":np.quantile(cboot@weights/weights.sum(),[.025,.975]).tolist()}
        report["results"][proxy]=data
        print(PANEL,proxy,{key:round(v["per_query"],4) for key,v in data["representations"].items()},data["paired_delta"],flush=True)
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_same_encoder_geometry.json")
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
