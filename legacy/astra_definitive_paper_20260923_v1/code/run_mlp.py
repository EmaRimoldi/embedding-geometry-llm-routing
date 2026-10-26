"""RouterBench per-model MLP, adapted to regress utility in our USD objective.

Preserves sklearn optimizer/default stopping and independent model heads. Hidden
width/depth is selected by source-only routing utility, never outer test results.
The upstream router predicts reward and applies a different token-price formula;
this utility adaptation matches our historical baseline and is explicitly named.
"""
import argparse,json,time,warnings
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel,delayed
from common import ROOT,load,score,sha

ARCH=[(100,),(100,100),(100,100,100)]

def head(x,y,xe,h):
 net=MLPRegressor(hidden_layer_sizes=h,activation='relu',solver='adam',
   learning_rate='constant',learning_rate_init=.001,max_iter=200,
   random_state=1234,alpha=.0001,early_stopping=False)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always',ConvergenceWarning);net.fit(x,y)
 return net.predict(xe),dict(iterations=int(net.n_iter_),loss=float(net.loss_),
   convergence_warnings=[str(w.message) for w in ws if issubclass(w.category,ConvergenceWarning)])

def predict(x,u,tr,te,h,jobs):
 fits=Parallel(n_jobs=jobs,max_nbytes='50M')(delayed(head)(x[tr],u[tr,m],x[te],h) for m in range(u.shape[1]))
 return np.stack([v[0] for v in fits],1),[v[1] for v in fits]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--jobs',type=int,default=4);a=ap.parse_args()
 cfg=json.loads((ROOT/'config/protocol.json').read_text());tasks=[]
 for panel,nf in [('routerbench-0shot',7),('sprout-sixfam',6)]:
  for split in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(nf)]:
   for lam in cfg['lambdas'][panel]:tasks.append((panel,split,lam))
 panel,split,lam=tasks[a.task];d=load(panel,'gte');s=json.loads((ROOT/'inputs'/panel/'splits.json').read_text())[split]
 tr=np.array(s['train']);te=np.array(s['test']);x,r,c,f=d['x'],d['r'],d['c'],d['fam'];u=r-lam*c
 dest=ROOT/'results/mlp'/panel/split/'gte'/f'lambda_{lam:g}';dest.mkdir(parents=True,exist_ok=False)
 t=time.time();vs=np.zeros(len(ARCH));n=0;diag=[]
 for ii,inner in enumerate(s['inner']):
  fit=np.array(inner['fit']);val=np.array(inner['val']);n+=len(val)
  for ih,h in enumerate(ARCH):
   p,info=predict(x,u,fit,val,h,a.jobs);ch=p.argmax(1);vs[ih]+=u[val,np.asarray(ch)].sum()
   diag.append(dict(inner=ii,architecture=list(h),heads=info));print('INNER',ii,h,'seconds',round(time.time()-t),flush=True)
 vs/=n;best=int(vs.argmax());p,info=predict(x,u,tr,te,ARCH[best],a.jobs);ch=p.argmax(1)
 np.savez_compressed(dest/'predictions.npz',test_rows=te,query_ids=d['query_ids'][te],choice=ch,predicted_utility=p)
 record=dict(panel=panel,split=split,representation='gte',lambda_value=lam,
  results={str(lam):{'mlp_utility_validated':score(ch,r[te],c[te],f[te],lam)}},
  n_train=len(tr),n_test=len(te),selected_architecture=ARCH[best],candidate_architectures=ARCH,
  validation_utilities=vs.tolist(),validation_queries=n,inner_diagnostics=diag,final_diagnostics=info,
  split_sha256=sha(ROOT/'inputs'/panel/'splits.json'),code_sha256=sha(__file__),
  sklearn_version=sklearn.__version__,seconds=time.time()-t,
  upstream='RouterBench cc67d100, routers/mlp_router.py',
  adaptation='Regress observed reward minus lambda times USD cost instead of upstream reward plus token-price adjustment.')
 (dest/'result.json').write_text(json.dumps(record,indent=2));print('DONE',panel,split,lam,round(time.time()-t),flush=True)
if __name__=='__main__':main()
