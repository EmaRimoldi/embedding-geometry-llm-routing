from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,load_data,unit
from geometry import Neighbours
import torch
out=[]
for dataset,fold,source in [('sprout',0,'mmlupro'),('routerbench',1,'hellaswag')]:
 d,_,_,_=load_data(dataset,'primary');r=d['reward'];f=d['family'];x=unit(np.load(ROOT/'artifacts/embeddings/gte-qwen2-7b'/dataset/'vectors.npy'))
 old=ROOT/'results/geometry/gte-qwen2-7b'/dataset/'primary'/f'ood_{fold}'
 z=np.load(old/'per_query.npz');te=z['test_rows'];a=np.flatnonzero(f==source)
 ng=Neighbours(x);nn,cos=ng.get(te,a,101)
 rel=r-r.mean(1,keepdims=True);err=((rel[nn[:,:100]].mean(1)-rel[te])**2).mean(1)
 older=z['source_family/'+source+'/relative_reward_mse']
 bad=np.flatnonzero(~np.isclose(err,older,rtol=1e-4,atol=2e-6))
 details=[]
 for i in bad[:20]:
  sim64=x[te[i]].astype('float64')@x[a].astype('float64').T
  order=np.argsort(-sim64,kind='stable')
  besterr=((rel[a[order[:100]]].mean(0)-rel[te[i]])**2).mean()
  details.append(dict(query_row=int(te[i]),old_error=float(older[i]),cpu_error=float(err[i]),float64_error=float(besterr),cpu_boundary_gap=float(cos[i,99]-cos[i,100]),float64_boundary_gap=float(sim64[order[99]]-sim64[order[100]])))
 out.append(dict(dataset=dataset,source=source,queries=len(te),mismatch_count=len(bad),old_mean=float(older.mean()),cpu_mean=float(err.mean()),max_difference=float(np.max(abs(err-older))),details=details))
p=ROOT/'results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer/numerical_check.json'
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
