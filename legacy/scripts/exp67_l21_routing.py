"""Matched L21 versus emitted GTE kNN model-choice sanity check.

Frozen representations, 5 family-stratified splits, held-out mean utility.
LLMRouterBench rows with >=8000 GTE tokens are removed before splitting.
No model training or selection on test outcomes.
"""
import json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rb.data import load_panel
from rb.panelctx import PANEL,REP
from rb.probes.sweep import SEEDS,split
from rb.reps import find_rep,load_rep
KS=(10,40,100);LAMBDAS=(0.,1.,10.,100.)
def topk(xq,xt,k=100):
    out=np.empty((len(xq),k),dtype=np.int32)
    for st in range(0,len(xq),512):
        with threadpool_limits(limits=8):s=xq[st:st+512]@xt.T
        ind=np.argpartition(-s,k-1,axis=1)[:,:k]
        order=np.argsort(-np.take_along_axis(s,ind,axis=1),axis=1)
        out[st:st+len(ind)]=np.take_along_axis(ind,order,axis=1)
    return out
p=load_panel(PANEL);fam=p.dataset_index();n=p.n
lengths=np.load(ROOT/'features'/PANEL/'prompt'/'lengths.npz')
key='qwen_tokens' if PANEL.startswith('routerbench') else 'gte_tokens'
l=lengths[key];l=l[p.keep] if p.keep is not None and len(l)==len(p.keep) else l
valid=l<8000
out=ROOT/'results'/'exp67_l21_routing'/PANEL
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
r=np.nan_to_num(p.rewards,nan=0.0).astype(np.float64)
c=np.nan_to_num(p.costs,nan=0.0).astype(np.float64)
assert np.isfinite(r).all() and np.isfinite(c).all() and p.available.all()
raw=np.load(ROOT/'features'/PANEL/'prompt'/'hidden'/'gte-qwen2-7b'/'L21_last.npy',mmap_mode='r')
h=np.asarray(raw[p.keep] if p.keep is not None and len(raw)==len(p.keep) else raw,dtype=np.float32)
g=load_rep(find_rep(PANEL,REP),n,p.keep).astype(np.float32)
assert h.shape==g.shape
reps={'emitted':g,'l21':h};rows=[];t0=time.time()
for name,x in reps.items():
 x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 for seed in SEEDS:
  tr,te=split(n,fam,seed)
  tr=np.sort(tr[valid[tr]]);te=np.sort(te[valid[te]])
  nb=topk(x[te],x[tr],max(KS))
  for lam in LAMBDAS:
   u=r-lam*c
   for k in KS:
    chosen=u[tr][nb[:,:k]].mean(1).argmax(1)
    util=u[te,chosen]
    rows.append({'rep':name,'seed':int(seed),'lambda':lam,'k':k,'n_test':len(te),
                 'per_query_utility':float(util.mean()),
                 'equal_family_utility':float(np.mean([util[fam[te]==f].mean() for f in np.unique(fam[te])]))})
  print(PANEL,name,seed,'done',round(time.time()-t0,1),flush=True)
report={'panel':PANEL,'valid_rows':int(valid.sum()),'method':'Frozen cosine kNN, identical family-stratified 70/30 splits and valid rows; train-neighbor mean utility chooses model','ks':KS,'lambdas':LAMBDAS,'rows':rows,'seconds':time.time()-t0}
(out/'result.json').write_text(json.dumps(report,indent=2))
