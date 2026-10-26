"""Frozen GTE internal state versus emitted embedding, matched held-out probes.

L21_last selected from the earlier RouterBench layer screen and fixed before
this independent LLMRouterBench validation. L28_last is a final-layer control.
Same rows, input prompts, target, regularization grid, splits and probe.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rb.data import SUCCESS_THRESHOLD,load_panel
from rb.panelctx import PANEL,REP,topic_str
from rb.probes.constants import RIDGE_ALPHAS
from rb.probes.sweep import SEEDS,split
from rb.reps import find_rep,load_rep

def main():
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_same_encoder.json")
    if dest.exists():raise FileExistsError(dest)
    p=load_panel(PANEL);fam=p.dataset_index();names=np.asarray(p.datasets)
    _,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    r=np.nan_to_num(p.rewards,nan=0.0)
    targets={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    paths={"emitted":find_rep(PANEL,REP).path}
    for layer in ("L21_last","L28_last"):
        paths[layer]=ROOT/"features"/PANEL/"prompt"/"hidden"/"gte-qwen2-7b"/(layer+".npy")
    out={**p.provenance(),"panel":PANEL,"representation":REP,"selected_layer":"L21_last",
         "selection_note":"Selected for last-token pooling in the earlier RouterBench sweep; the same choice is validated on LLMRouterBench. L28_last is the final-state control.",
         "regularization":RIDGE_ALPHAS.tolist(),"split_seeds":list(SEEDS),"targets":list(targets),
         "method":"RidgeCV on training-standardized 3584d representation, identical 70/30 family-stratified splits, train cell-mean baseline; held-out R2 beyond cell and paired query bootstrap of R2 difference", "reps":{}}
    seed_data={}
    for seed in SEEDS:
        tr,te=split(p.n,fam,seed);tr,te=np.sort(tr),np.sort(te)
        floors={}
        for key,y in targets.items():
            fallback={f:float(y[tr[fam[tr]==f]].mean()) for f in np.unique(fam)}
            fl=np.array([fallback[f] for f in fam[te]])
            for c in np.unique(cell[te]):
                a=tr[cell[tr]==c]
                if len(a)>=50:fl[cell[te]==c]=float(y[a].mean())
            floors[key]=fl
        seed_data[seed]=(tr,te,floors)
    for rep,path in paths.items():
        arr=np.load(path,mmap_mode="r")
        x=arr[p.keep] if p.keep is not None and len(arr)==len(p.keep) else arr
        x=np.asarray(x,dtype=np.float32)
        assert x.shape==(p.n,3584),x.shape
        rows=[]
        for seed,(tr,te,floors) in seed_data.items():
            sc=StandardScaler().fit(x[tr]); a=sc.transform(x[tr]);b=sc.transform(x[te])
            row={"seed":seed,"n_test":len(te),"proxy":{}}
            for key,y in targets.items():
                model=RidgeCV(alphas=RIDGE_ALPHAS).fit(a,y[tr]);pred=model.predict(b)
                base=floors[key]
                r2=1-float(np.sum((y[te]-pred)**2))/float(np.sum((y[te]-base)**2))
                per_f={names[f]:float(1-np.sum((y[te[fam[te]==f]]-pred[fam[te]==f])**2)/
                       np.sum((y[te[fam[te]==f]]-base[fam[te]==f])**2)) for f in np.unique(fam)}
                row["proxy"][key]={"r2_beyond_cell":r2,"alpha":float(model.alpha_),"per_family":per_f}
                if seed==0:
                    np.save(ROOT/"results"/"difficulty_geometry"/(PANEL+"_same_encoder_"+rep+"_"+key+"_split0.npy"),
                            np.column_stack([te,y[te],base,pred]).astype(np.float32))
            rows.append(row)
            print(PANEL,rep,seed,{k:round(v["r2_beyond_cell"],4) for k,v in row["proxy"].items()},flush=True)
        out["reps"][rep]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"per_seed":rows,
                           "mean":{key:float(np.mean([z["proxy"][key]["r2_beyond_cell"] for z in rows])) for key in targets}}
    # Paired split-0 bootstrap, query strata fixed within family.
    rng=np.random.default_rng(7119)
    for key in targets:
        d={rep:np.load(ROOT/"results"/"difficulty_geometry"/(PANEL+"_same_encoder_"+rep+"_"+key+"_split0.npy")) for rep in paths}
        assert all(np.array_equal(d["emitted"][:,0],z[:,0]) for z in d.values())
        q=d["emitted"][:,0].astype(int); y=d["emitted"][:,1]; base=d["emitted"][:,2]
        group=[np.flatnonzero(fam[q]==f) for f in np.unique(fam)]
        diffs=[]
        for z in range(1000):
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in group])
            den=np.sum((y[ix]-base[ix])**2)
            rh=1-np.sum((y[ix]-d["L21_last"][ix,3])**2)/den
            re=1-np.sum((y[ix]-d["emitted"][ix,3])**2)/den
            diffs.append(float(rh-re))
        out.setdefault("paired_bootstrap_split0",{})[key]={"delta_L21_minus_emitted_ci95":np.quantile(diffs,[.025,.975]).tolist(),
              "delta_observed":float(out["reps"]["L21_last"]["per_seed"][0]["proxy"][key]["r2_beyond_cell"]-out["reps"]["emitted"]["per_seed"][0]["proxy"][key]["r2_beyond_cell"])}
    dest.write_text(json.dumps(out,indent=2))
    print("SAVED",dest,flush=True)

if __name__=="__main__":main()
