"""Fixed linear accessibility diagnostic on all definitive OOD queries.
Query-equal source fitting and reporting. Alpha is not tuned on the target.
"""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
from scipy.linalg import eigh
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parent
sys.path.insert(0,str(BASE/'code'))
from common import load,sha
ALPHAS=[10.,100.,1000.]

def fit_predict(x,y,tr,te):
 # Source-only feature standardization and intercept; conventional ridge SSE objective.
 xt=x[tr].astype(float);mu=xt.mean(0);scale=xt.std(0);scale[scale<1e-12]=1.;xt=(xt-mu)/scale
 ym=y[tr].mean(0);target=y[tr]-ym;cov=xt.T@xt;cross=xt.T@target;ev,V=eigh(cov,check_finite=False);ev=np.maximum(ev,0);proj=V.T@cross;test=(x[te].astype(float)-mu)/scale
 out={}
 for a in ALPHAS:
  coef=V@(proj/(ev[:,None]+a));p=test@coef+ym
  assert np.isfinite(p).all()
  residual=(cov+a*np.eye(cov.shape[0]))@coef-cross
  rel=float(np.linalg.norm(residual)/max(np.linalg.norm(cross),1e-12));assert rel<1e-6
  out[a]=(p,ym,rel)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);a=ap.parse_args()
 tasks=[(p,g) for p,n in [('routerbench-0shot',7),('sprout-sixfam',6)] for g in range(n)];panel,g=tasks[a.task]
 split=f'ood_{g}';s=json.loads((BASE/'inputs'/panel/'splits.json').read_text())[split];tr=np.array(s['train']);te=np.array(s['test']);d=load(panel)
 assert np.intersect1d(d['fam'][tr],d['fam'][te]).size==0
 r,c=d['r'],d['c'];good=(c>0).all(1);m=r.shape[1]
 reward=np.column_stack([r.mean(1),r-r.mean(1,keepdims=True)])
 cost=np.full_like(reward,np.nan);logc=np.log(c[good]);cost[good]=np.column_stack([logc.mean(1),logc-logc.mean(1,keepdims=True)])
 dest=HERE/'results'/panel/split;dest.mkdir(parents=True,exist_ok=False);arrays={};results={};t=time.time()
 for rep in ['gte','reader']:
  x=load(panel,rep)['x']
  for group,y,fit,test,names in [('reward',reward,tr,te,['mean_reward','relative_reward']),('log_cost',cost,tr[good[tr]],te[good[te]],['mean_log_cost','relative_log_cost'])]:
   assert len(fit)>0 and len(test)>0
   pred=fit_predict(x,y,fit,test)
   arrays[f'{rep}|{group}|query_ids']=d['query_ids'][test];arrays[f'{rep}|{group}|test_rows']=test
   for alpha,(p,baseline,residual) in pred.items():
    arrays[f'{rep}|{group}|{alpha:g}|prediction']=p
    for target,cols in zip(names,[slice(0,1),slice(1,None)]):
     err=((p[:,cols]-y[test,cols])**2).mean(1);base=((baseline[cols]-y[test,cols])**2).mean(1)
     arrays[f'{rep}|{target}|{alpha:g}|sq_error']=err;arrays[f'{rep}|{target}|{alpha:g}|baseline_sq_error']=base
     results[f'{rep}|{target}|{alpha:g}']=dict(n=len(test),mse=float(err.mean()),source_mean_mse=float(base.mean()),reduction=float(1-err.mean()/base.mean()),normal_equation_residual=residual)
   print(panel,split,rep,group,'seconds',round(time.time()-t),flush=True)
 np.savez_compressed(dest/'predictions.npz',**arrays)
 record=dict(panel=panel,split=split,family=d['meta']['families'][g],primary_alpha=100.,alphas=ALPHAS,results=results,n_train=len(tr),n_test=len(te),cost_eligible_train=int(good[tr].sum()),cost_eligible_test=int(good[te].sum()),source_weighting='equal query',fit='unit vectors, source-only feature standardization, intercept, ridge sum of squared errors',code_sha256=sha(__file__),split_sha256=sha(BASE/'inputs'/panel/'splits.json'),seconds=time.time()-t)
 (dest/'result.json').write_text(json.dumps(record,indent=2));print('DONE',panel,split,flush=True)
if __name__=='__main__':main()
