"""Freeze outcome-independent source folds; verify winner-only fitting preserves heads."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,load,split_path,sha
from run_mlp import atomic_json,tasks
from batched_mlp import BatchedMLP

def main():
 folds={}
 for panel in ['routerbench-0shot','sprout-sixfam']:
  d=load(panel);all_s=json.loads(split_path(panel).read_text());folds[panel]={}
  for sp,s in all_s.items():
   if not isinstance(s,dict) or 'inner' not in s:continue
   ids=list(range(len(s['inner'])))
   if sp.startswith('ood'):
    ids=sorted(ids,key=lambda i:hashlib.sha256(f'mlp-reduced-v1|{panel}|{sp}|{i}'.encode()).hexdigest())[:2]
    assert len(ids)==2
   else:assert len(ids)==1
   tr=set(s['train']);te=set(s['test']);assert not tr&te
   detail=[]
   for i in ids:
    f=s['inner'][i];fi=set(f['fit']);va=set(f['val'])
    assert not fi&va and fi|va==tr and not (fi|va)&te
    vf=np.unique(d['fam'][f['val']]);ff=np.unique(d['fam'][f['fit']]);tf=np.unique(d['fam'][s['test']])
    if sp.startswith('ood'):assert len(vf)==1 and len(tf)==1 and not set(vf)&set(ff) and not set(tf)&set(vf)
    detail.append(dict(index=i,validation_families=[str(d['meta']['families'][v]) for v in vf],n_fit=len(fi),n_validation=len(va)))
   folds[panel][sp]=dict(inner_indices=ids,fold_details=detail,split_sha256=sha(split_path(panel)))
 cfg=json.loads((ROOT/'configs/baselines.json').read_text())
 names=['src/baselines/run_mlp_reduced.py','src/baselines/prepare_mlp_reduced.py','src/baselines/report_reduced.py','jobs/baselines_mlp_reduced.sbatch','jobs/baselines_reduced_report.sbatch','jobs/baselines_reduced_preflight.sbatch']
 record=dict(version='mlp_reduced_v1',early_stopping=True,architectures=cfg['mlp']['architectures'],alphas=cfg['mlp']['alphas'],fold_selection='First two source folds by SHA256(mlp-reduced-v1|panel|outer_split|inner_index); IID unchanged',folds=folds,source_hashes={p:sha(ROOT/p) for p in names},original_source_hashes_sha256=sha(ROOT/'results/baselines/_run/source_hashes.json'))
 p=ROOT/'configs/mlp_reduced.json'
 if p.exists():assert json.loads(p.read_text())==record,'Frozen manifest changed'
 else:atomic_json(p,record)
 # Heads are independent. Check the same selected alpha with and without the other heads.
 torch.set_num_threads(2);rng=np.random.RandomState(888)
 x=rng.normal(size=(128,12));y=rng.normal(size=(128,2));alphas=cfg['mlp']['alphas'];m=2
 full_y=np.broadcast_to(y[:,None,:],(len(y),len(alphas),m)).reshape(len(y),-1)
 tests=[]
 for h in cfg['mlp']['architectures']:
  for dtype,tol in [(torch.float64,1e-10),(torch.float32,1e-5)]:
   kw=dict(device='cpu',dtype=dtype,max_iter=20,batch_size=32,early_stopping=True,n_iter_no_change=3)
   full=BatchedMLP(h,np.repeat(alphas,m),**kw).fit(x,full_y)
   j=2;one=BatchedMLP(h,[alphas[j]]*m,**kw).fit(x,y)
   delta=float(np.max(np.abs(full.predict(x)[:,j*m:(j+1)*m]-one.predict(x))))
   assert delta<tol,(h,str(dtype),delta)
   np.testing.assert_array_equal(full.iterations[j*m:(j+1)*m],one.iterations)
   tests.append(dict(architecture=h,dtype=str(dtype),max_difference=delta))
 import run_mlp_reduced as rr
 rr.check_sources();next_units=[];cached=0
 for task,t in enumerate(tasks()):
  u=rr.pending(task);next_units.append(dict(task=task,description=t,next_unit=u));cached+=u is None
 atomic_json(ROOT/'results/baselines/_run/reduced_preflight.json',dict(passed=True,head_subset_tests=tests,complete_from_cached_fits=cached,next_units=next_units,protocol_sha256=sha(p)))
 print('PREFLIGHT_PASS',cached,'evaluations can assemble from cached fits',tests,flush=True)
if __name__=='__main__':main()
