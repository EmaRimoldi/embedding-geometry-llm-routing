"""Matched fixed soft priors, kNN and K-means on each source-only outer fold."""
import argparse,json,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from learning_common import *
from predictors import prior

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);a=ap.parse_args()
    g=a.task//3;rep=REPS[a.task%3];d=load('sprout-sixfam',rep);t=time.time()
    f=d['fam'];tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g);assert len(np.unique(f[tr]))==5 and not np.isin(te,tr).any()
    x,r,c=d['x'],d['r'],d['c'];src=np.unique(f[tr]);rt=np.stack([r[tr][f[tr]==h].mean(0) for h in src]);ct=np.stack([c[tr][f[tr]==h].mean(0) for h in src])
    pr,pc=prior(x[tr],r[tr],c[tr],f[tr],x[te],42)
    sc=StandardScaler().fit(x[tr]);clf=LogisticRegression(C=1.,max_iter=500,class_weight='balanced').fit(sc.transform(x[tr]),f[tr])
    post=clf.predict_proba(sc.transform(x[te]));assert np.array_equal(clf.classes_,src)
    preds={'soft_prior':(pr,pc),'soft_prior_standardized':(post@rt,post@ct),'hard_lookup_standardized':(rt[post.argmax(1)],ct[post.argmax(1)])}
    nb,_=nearest(x[te],x[tr],100)
    for k in (40,100):preds[f'knn{k}']=(r[tr[nb[:,:k]]].mean(1),c[tr[nb[:,:k]]].mean(1))
    km=KMeans(64,n_init=1,random_state=0).fit(x[tr]);kt=km.predict(x[te]);kr=[];kc=[]
    for j in range(64):
        jj=tr[km.labels_==j];kr.append(r[jj].mean(0) if len(jj) else r[tr].mean(0));kc.append(c[jj].mean(0) if len(jj) else c[tr].mean(0))
    preds['kmeans64']=(np.asarray(kr)[kt],np.asarray(kc)[kt])
    preds['best_single_train']=(np.broadcast_to(r[tr].mean(0),r[te].shape),np.broadcast_to(c[tr].mean(0),c[te].shape))
    results={};arrays={'test_rows':te,'query_ids':d['ids'][te]}
    for name,(rr,cc) in preds.items():arrays[name+'|r']=rr;arrays[name+'|c']=cc
    for lam in LAMS['sprout-sixfam']:
        results[f'{lam:g}']={}
        for name,(rr,cc) in preds.items():
            ch=(rr-lam*cc).argmax(1);results[f'{lam:g}'][name]=evaluate(ch,r[te],c[te],d['missing'][te],lam);arrays[f'{lam:g}|choice|{name}']=ch.astype(np.int16)
    dest=ROOT/'results/baselines'/rep/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'predictions.npz',**arrays)
    (dest/'result.json').write_text(json.dumps(dict(panel='sprout-sixfam',representation=rep,family_index=g,family=d['meta']['families'][g],rows=len(te),results=results,
       snapshot_sha256=d['meta']['snapshot_sha256'],source_sha256=code_sha(__file__),seconds=time.time()-t),indent=2))
    print('DONE',rep,g,round(time.time()-t),flush=True)
if __name__=='__main__':main()
