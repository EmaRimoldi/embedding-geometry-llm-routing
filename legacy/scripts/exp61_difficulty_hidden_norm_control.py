"""Independent LLM panel control: is L21 advantage merely hidden norm?"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rb.data import SUCCESS_THRESHOLD,load_panel
from rb.panelctx import PANEL,topic_str
from rb.probes.constants import RIDGE_ALPHAS
from rb.probes.sweep import SEEDS,split

def main():
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_hidden_norm_control.json")
    if dest.exists():raise FileExistsError(dest)
    p=load_panel(PANEL);fam=p.dataset_index()
    _,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    path=ROOT/"features"/PANEL/"prompt"/"hidden/gte-qwen2-7b/L21_last.npy"
    arr=np.load(path,mmap_mode="r")
    x=np.array(arr[p.keep] if p.keep is not None and len(arr)==len(p.keep) else arr,dtype=np.float32,copy=True)
    norm=np.linalg.norm(x,axis=1)
    x/=np.maximum(norm[:,None],1e-12)
    reps={"L21_unit_direction":x,"L21_log_norm_only":np.log(norm[:,None])}
    r=np.nan_to_num(p.rewards,nan=0.0)
    ys={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    out={**p.provenance(),"panel":PANEL,"method":"same standardization, RidgeCV alphas, five family-stratified 70/30 splits and train cell floor as exp59; L21 unit directions or log norm alone", "per_rep":{}}
    for key,x in reps.items():
        rows=[]
        for seed in SEEDS:
            tr,te=split(p.n,fam,seed);tr,te=np.sort(tr),np.sort(te)
            sc=StandardScaler().fit(x[tr]);a=sc.transform(x[tr]);b=sc.transform(x[te])
            row={"seed":seed,"proxy":{}}
            for proxy,y in ys.items():
                floor=np.array([y[tr[fam[tr]==f]].mean() for f in fam[te]])
                for c in np.unique(cell[te]):
                    z=tr[cell[tr]==c]
                    if len(z)>=50:floor[cell[te]==c]=y[z].mean()
                model=RidgeCV(alphas=RIDGE_ALPHAS).fit(a,y[tr]);pred=model.predict(b)
                r2=float(1-np.sum((y[te]-pred)**2)/np.sum((y[te]-floor)**2))
                row["proxy"][proxy]={"r2_beyond_cell":r2,"alpha":float(model.alpha_)}
                if seed==0 and key=="L21_unit_direction":
                    np.save(ROOT/"results"/"difficulty_geometry"/(PANEL+"_L21_unit_"+proxy+"_split0.npy"),np.column_stack([te,y[te],floor,pred]).astype(np.float32))
            rows.append(row)
            print(PANEL,key,seed,{k:round(v["r2_beyond_cell"],4) for k,v in row["proxy"].items()},flush=True)
        out["per_rep"][key]={"per_seed":rows,"mean":{proxy:float(np.mean([z["proxy"][proxy]["r2_beyond_cell"] for z in rows])) for proxy in ys}}
    rng=np.random.default_rng(7141)
    for proxy in ys:
        a=np.load(ROOT/"results"/"difficulty_geometry"/(PANEL+"_L21_unit_"+proxy+"_split0.npy"))
        b=np.load(ROOT/"results"/"difficulty_geometry"/(PANEL+"_same_encoder_emitted_"+proxy+"_split0.npy"))
        assert np.array_equal(a[:,0],b[:,0]); ids=a[:,0].astype(int)
        group=[np.flatnonzero(fam[ids]==f) for f in np.unique(fam)]
        boots=[]
        for _ in range(1000):
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in group])
            den=np.sum((a[ix,1]-a[ix,2])**2)
            boots.append(float((np.sum((a[ix,1]-b[ix,3])**2)-np.sum((a[ix,1]-a[ix,3])**2))/den))
        out.setdefault("paired_vs_emitted_split0",{})[proxy]={"ci95":np.quantile(boots,[.025,.975]).tolist(),
              "observed":float(out["per_rep"]["L21_unit_direction"]["per_seed"][0]["proxy"][proxy]["r2_beyond_cell"]-json.load(open(ROOT/"results"/"difficulty_geometry"/(PANEL+"_same_encoder.json")))["reps"]["emitted"]["per_seed"][0]["proxy"][proxy]["r2_beyond_cell"])}
    dest.write_text(json.dumps(out,indent=2));print("SAVED",dest,flush=True)

if __name__=="__main__":main()
