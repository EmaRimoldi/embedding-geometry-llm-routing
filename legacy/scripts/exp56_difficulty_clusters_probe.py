"""Independent read-only test: do embedding-only clusters partition outcomes?

Cluster each benchmark family without labels, then estimate cluster mean outcome
on training rows and evaluate on held-out queries against family/topic means.
"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rb.data import SUCCESS_THRESHOLD, load_panel
from rb.panelctx import PANEL, REP, topic_str
from rb.probes.sweep import split
from rb.reps import find_rep, load_rep


def main():
    p=load_panel(PANEL); fam=p.dataset_index()
    _,cell=np.unique(np.asarray(topic_str(p),dtype=str),return_inverse=True)
    x=load_rep(find_rep(PANEL,REP),p.n,p.keep).astype(np.float32)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    r=np.nan_to_num(p.rewards,nan=0.0)
    ys={"solve_fraction":(r>=SUCCESS_THRESHOLD).mean(1),"mean_reward":r.mean(1)}
    output={"panel":PANEL,"representation":REP,"split_seeds":[0,1],"k":[2,8],
            "method":"embedding-only MiniBatchKMeans within each family, train-set (cell,cluster) outcome means; held-out R2 beyond train family/topic cell mean, falling back to cell mean if fewer than 30 train rows in (cell,cluster)",
            "results":{}}
    for seed in (0,1):
        train,test=split(p.n,fam,seed)
        for k in (2,8):
            pred={key:np.zeros(len(test),dtype=float) for key in ys}
            base={key:np.zeros(len(test),dtype=float) for key in ys}
            for f,name in enumerate(p.datasets):
                tr=train[fam[train]==f]; tepos=np.flatnonzero(fam[test]==f); te=test[tepos]
                if len(tr)<50: continue
                model=MiniBatchKMeans(n_clusters=k,batch_size=1024,n_init=3,max_iter=75,
                                      random_state=seed*31+f).fit(x[tr])
                ctr=model.labels_; cte=model.predict(x[te])
                for key,y in ys.items():
                    family_mean=float(y[tr].mean())
                    cell_means={int(c):float(y[tr[cell[tr]==c]].mean()) for c in np.unique(cell[tr])}
                    for c in np.unique(cell[te]):
                        ix=np.flatnonzero(cell[te]==c)
                        cm=cell_means.get(int(c),family_mean)
                        base[key][tepos[ix]]=cm
                        for z in np.unique(cte[ix]):
                            q=ix[cte[ix]==z]
                            ids=tr[(cell[tr]==c)&(ctr==z)]
                            pred[key][tepos[q]]=float(y[ids].mean()) if len(ids)>=30 else cm
            for key,y in ys.items():
                yy=y[test]; sse0=np.sum((yy-base[key])**2)
                val=1-np.sum((yy-pred[key])**2)/sse0
                byfam={p.datasets[f]:float(1-np.sum((yy[fam[test]==f]-pred[key][fam[test]==f])**2)/
                       np.sum((yy[fam[test]==f]-base[key][fam[test]==f])**2)) for f in range(len(p.datasets))}
                output["results"].setdefault(key,{}).setdefault(str(k),[]).append({"seed":seed,"beyond_cell_r2":float(val),"per_family":byfam})
            print("finished",PANEL,seed,k,flush=True)
    dest=ROOT/"results"/"difficulty_geometry"/(PANEL+"_clusters.json")
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(output,indent=2))
    print(json.dumps(output["results"],indent=2),flush=True)

if __name__=="__main__":main()
