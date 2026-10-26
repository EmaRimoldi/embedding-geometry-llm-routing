"""Label-free, within-topic clustering, evaluated on held-out outcomes."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rb.data import SUCCESS_THRESHOLD,load_panel
from rb.panelctx import PANEL,REP,topic_str
from rb.probes.sweep import split
from rb.reps import find_rep,load_rep

def main():
    p=load_panel(PANEL); fam=p.dataset_index()
    names,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    x=load_rep(find_rep(PANEL,REP),p.n,p.keep).astype(np.float32)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    r=np.nan_to_num(p.rewards,nan=0.0)
    ys={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    out={"panel":PANEL,"rep":REP,"method":"embedding-only k=2 MiniBatchKMeans fitted separately within each released family/topic cell, evaluated on held-out queries against train-set cell mean; cells with <60 train rows use baseline, clusters <15 train rows use baseline", "seeds":[0,1,2],"per_seed":[]}
    for seed in (0,1,2):
        tr,te=split(p.n,fam,seed)
        preds={key:np.zeros(len(te)) for key in ys}; bases={key:np.zeros(len(te)) for key in ys}
        eligible=np.zeros(len(te),dtype=bool); used=[]
        for c,name in enumerate(names):
            a=tr[cell[tr]==c]; pos=np.flatnonzero(cell[te]==c); b=te[pos]
            if len(a)==0:
                for key,y in ys.items():
                    family_train=tr[fam[tr]==fam[b[0]]]
                    bases[key][pos]=preds[key][pos]=float(y[family_train].mean())
                continue
            for key,y in ys.items(): bases[key][pos]=float(y[a].mean())
            if len(a)<60:
                for key in ys:preds[key][pos]=bases[key][pos]
                continue
            model=MiniBatchKMeans(n_clusters=2,n_init=3,max_iter=75,batch_size=512,
                                  random_state=seed*100+c).fit(x[a])
            ctr=model.labels_; cte=model.predict(x[b]); eligible[pos]=True
            used.append((str(name),len(a),len(b)))
            for key,y in ys.items():
                cm=float(y[a].mean())
                for z in (0,1):
                    train_ids=a[ctr==z]; q=pos[cte==z]
                    preds[key][q]=float(y[train_ids].mean()) if len(train_ids)>=15 else cm
        res={"seed":seed,"eligible_queries":int(eligible.sum()),"cells_used":len(used),"proxy":{}}
        for key,y in ys.items():
            for subset, mask in (("all",np.ones(len(te),dtype=bool)),("eligible",eligible)):
                base_error=np.sum((y[te][mask]-bases[key][mask])**2)
                effect=1-np.sum((y[te][mask]-preds[key][mask])**2)/base_error
                res["proxy"].setdefault(key,{})[subset]=float(effect)
        out["per_seed"].append(res)
        print(PANEL,res,flush=True)
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_within_cell_clusters.json")
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(out,indent=2))

if __name__=="__main__":main()
