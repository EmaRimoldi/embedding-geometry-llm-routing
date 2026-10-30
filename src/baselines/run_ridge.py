"""Linear Ridge utility routing on the frozen final-paper baseline splits."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from common import ROOT,load,split_path,destination,score,sha
from run_mlp import tasks,atomic_json,atomic_npz
CFG=json.loads((ROOT/'configs/baselines.json').read_text())
RCFG=json.loads((ROOT/'configs/ridge.json').read_text())
ALPHAS=RCFG['alpha_grid']

def predictions(x,y,fit,evaluate,alphas):
 """Each output is independent. No information from evaluate fits the map."""
 a=np.asarray(x[fit],np.float64);xm=a.mean(0);a-=xm
 target=np.asarray(y[fit],np.float64);ym=target.mean(0);target-=ym
 gram=a.T@a;rhs=a.T@target
 ev,V=eigh(gram,check_finite=True,driver='evd')
 assert ev.min()>-1e-8 and ev.min()+min(alphas)>0
 rhs=V.T@rhs
 q=(np.asarray(x[evaluate],np.float64)-xm)@V
 for alpha in alphas:
  yield q@(rhs/(ev[:,None]+alpha))+ym

def check():
 rng=np.random.RandomState(49);x=rng.normal(size=(91,13));r=rng.uniform(size=(91,4));c=rng.uniform(size=(91,4))*.005
 fi=np.arange(71);te=np.arange(71,91);y=np.concatenate([r,c],1);errors=[]
 for alpha,p in zip(ALPHAS,predictions(x,y,fi,te,ALPHAS)):
  for lam in [0,2,50,500]:
   direct=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(x[fi],r[fi]-lam*c[fi]).predict(x[te])
   equivalent=p[:,:4]-lam*p[:,4:]
   error=float(np.max(abs(direct-equivalent)));assert error<1e-10
   errors.append(dict(alpha=alpha,lambda_value=lam,max_abs_difference=error))
 out=ROOT/'results/baselines/_run'
 atomic_json(out/'ridge_equivalence.json',dict(pass_all=True,checks=errors))
 print('RIDGE_CHECK_PASS',len(errors),'direct utility equivalence checks',flush=True)

def freeze():
 files=['src/baselines/run_ridge.py','configs/ridge.json','src/baselines/common.py','src/baselines/run_mlp.py','configs/baselines.json']
 files += [str(split_path(p).relative_to(ROOT)) for p in CFG['panels']]
 rec={p:sha(ROOT/p) for p in files};dest=ROOT/'results/baselines/_run/ridge_sources.json'
 if dest.exists():assert json.loads(dest.read_text())==rec
 else:atomic_json(dest,rec)

def verify():
 for p,h in json.loads((ROOT/'results/baselines/_run/ridge_sources.json').read_text()).items():
  assert sha(ROOT/p)==h,('Ridge frozen dependency changed',p)

def run(task):
 verify();panel,sp,rep=tasks()[task];root=destination('ridge',panel,sp,rep)
 if (root/'result.json').exists():print('RIDGE_ALREADY_COMPLETE',panel,sp,rep,flush=True);return
 root.mkdir(parents=True,exist_ok=True);d=load(panel,rep);s=json.loads(split_path(panel).read_text())[sp]
 tr=np.array(s['train']);te=np.array(s['test']);x=d['x'];r=d['r'];c=d['c'];m=r.shape[1]
 y=np.concatenate([r,c],1);lams=CFG['lambdas'][panel];vt=np.zeros((len(lams),len(ALPHAS)));nv=0;t=time.time()
 for ii,q in enumerate(s['inner']):
  fi=np.array(q['fit']);va=np.array(q['val'])
  assert not np.intersect1d(fi,va).size and np.isin(fi,tr).all() and np.isin(va,tr).all()
  cp=root/f'inner_{ii}.npz'
  if cp.exists():
   z=np.load(cp);np.testing.assert_array_equal(z['validation_rows'],va);vt+=z['utility_sum'];nv+=len(va);continue
  sums=np.zeros_like(vt)
  for ia,p in enumerate(predictions(x,y,fi,va,ALPHAS)):
   assert np.isfinite(p).all()
   for il,lam in enumerate(lams):
    ch=(p[:,:m]-lam*p[:,m:]).argmax(1);sums[il,ia]=(r[va,ch]-lam*c[va,ch]).sum()
  atomic_npz(cp,validation_rows=va,utility_sum=sums);vt+=sums;nv+=len(va)
  print('RIDGE_INNER',panel,sp,rep,ii,'seconds',round(time.time()-t),flush=True)
 vt/=nv;best=vt.argmax(1);selected=sorted(set(best.tolist()));alphas=[ALPHAS[i] for i in selected]
 arrays=dict(test_rows=te,query_ids=d['query_ids'][te]);results={};selections={}
 for ia,p in zip(selected,predictions(x,y,tr,te,alphas)):
  assert np.isfinite(p).all()
  for il in np.flatnonzero(best==ia):
   lam=lams[il];u=p[:,:m]-lam*p[:,m:];ch=u.argmax(1)
   results[str(lam)]={'ridge':score(ch,r[te],c[te],d['fam'][te],lam)}
   selections[str(lam)]=dict(alpha=ALPHAS[ia],candidate_alphas=ALPHAS,validation_utilities=vt[il].tolist(),validation_queries=nv)
   arrays[f'{lam:g}|ridge']=ch.astype(np.int16);arrays[f'{lam:g}|predicted_utility']=u
   arrays[f'{lam:g}|predicted_reward']=p[:,:m];arrays[f'{lam:g}|predicted_cost']=p[:,m:]
 atomic_npz(root/'predictions.npz',**arrays)
 atomic_json(root/'result.json',dict(panel=panel,split=sp,representation=rep,display_name='Ridge',results=results,selections=selections,n_train=len(tr),n_test=len(te),seconds=time.time()-t,slurm_job_id=os.environ.get('SLURM_JOB_ID'),split_sha256=sha(split_path(panel)),code_sha256=sha(__file__),config_sha256=sha(ROOT/'configs/ridge.json')))
 print('RIDGE_DONE',panel,sp,rep,'seconds',round(time.time()-t),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');a.add_argument('--pack',type=int);args=a.parse_args()
 if args.check:check();freeze()
 if args.pack is not None:
  run(2*args.pack);run(2*args.pack+1)
