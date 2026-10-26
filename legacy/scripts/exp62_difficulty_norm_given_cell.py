"""Incremental value of internal norm conditional on released topic cell."""
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
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_norm_given_cell.json")
    if dest.exists():raise FileExistsError(dest)
    p=load_panel(PANEL);fam=p.dataset_index()
    _,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    z=np.eye(cell.max()+1,dtype=np.float32)[cell]
    arr=np.load(ROOT/"features"/PANEL/"prompt/hidden/gte-qwen2-7b/L21_last.npy",mmap_mode="r")
    x=arr[p.keep] if p.keep is not None and len(arr)==len(p.keep) else arr
    lognorm=np.log(np.linalg.norm(x,axis=1))[:,None].astype(np.float32)
    r=np.nan_to_num(p.rewards,nan=0.0)
    ys={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    out={**p.provenance(),"panel":PANEL,"method":"RidgeCV on cell one-hot plus standardized L21 log norm; held-out R2 beyond train-set cell mean with family fallback below 50; same five splits and ridge grid as exp59", "per_seed":[]}
    for seed in SEEDS:
        tr,te=split(p.n,fam,seed);tr,te=np.sort(tr),np.sort(te)
        sc=StandardScaler().fit(lognorm[tr]);a=np.column_stack([z[tr],sc.transform(lognorm[tr])]);b=np.column_stack([z[te],sc.transform(lognorm[te])])
        row={"seed":seed,"proxy":{}}
        for key,y in ys.items():
            floor=np.array([y[tr[fam[tr]==f]].mean() for f in fam[te]])
            for c in np.unique(cell[te]):
                ids=tr[cell[tr]==c]
                if len(ids)>=50:floor[cell[te]==c]=y[ids].mean()
            fit=RidgeCV(alphas=RIDGE_ALPHAS).fit(a,y[tr]);pred=fit.predict(b)
            row["proxy"][key]={"r2_beyond_cell":float(1-np.sum((y[te]-pred)**2)/np.sum((y[te]-floor)**2)),"alpha":float(fit.alpha_)}
        out["per_seed"].append(row);print(PANEL,seed,row["proxy"],flush=True)
    out["mean"]={key:float(np.mean([q["proxy"][key]["r2_beyond_cell"] for q in out["per_seed"]])) for key in ys}
    dest.write_text(json.dumps(out,indent=2));print("SAVED",dest,flush=True)

if __name__=="__main__":main()
