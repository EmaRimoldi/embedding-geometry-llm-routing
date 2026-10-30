"""User-approved reduced MLP validation; frozen trainer and original artifacts preserved."""
import argparse, functools, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import common
from common import ROOT, destination, split_path, sha, score
from run_mlp import tasks, done_path, atomic_json, atomic_npz
from batched_mlp import BatchedMLP
CFG = json.loads((ROOT/'configs/baselines.json').read_text())
P = CFG['mlp']
MANIFEST = ROOT/'configs/mlp_reduced.json'
STOP = False
# Cache one representation, avoiding repeated feature loads between fit batches.
load = functools.lru_cache(maxsize=1)(common.load)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def manifest(): return json.loads(MANIFEST.read_text())
def root_for(task):
 panel, sp, rep = tasks()[task]
 return destination('mlp_reduced', panel, sp, rep)
def split_for(task):
 panel, sp, _ = tasks()[task]
 return json.loads(split_path(panel).read_text())[sp]
def inner_ids(task):
 panel, sp, _ = tasks()[task]
 return manifest()['folds'][panel][sp]['inner_indices']
def check_sources():
 for f, h in json.loads((ROOT/'results/baselines/_run/source_hashes.json').read_text()).items():
  assert sha(ROOT/f) == h, ('Original source changed', f)
 for f,h in manifest()['source_hashes'].items(): assert sha(ROOT/f)==h, f

def existing(task, ia, lam, fold):
 """Prefer matching original completed fits; never overwrite or refit them."""
 panel, sp, rep = tasks()[task]
 for base in [destination('mlp', panel, sp, rep), root_for(task)]:
  p = done_path(base, ia, True, lam, fold)
  if p.exists():
   d = json.loads(p.read_text())
   assert d['panel']==panel and d['split']==sp and d['representation']==rep
   assert d['architecture']==P['architectures'][ia] and d['early_stopping'] is True
   assert d['alphas']==P['alphas'] and d['lambda_value']==lam and d['fold']==fold
   assert d['split_sha256']==sha(split_path(panel))
   assert d['trainer_sha256']==sha(Path(__file__).with_name('batched_mlp.py'))
   assert p.with_suffix('.npz').exists(), p
   return p
 return None

def selection(task, lam):
 s=split_for(task); ids=inner_ids(task)
 n=sum(len(s['inner'][i]['val']) for i in ids)
 candidates=[]
 for ia,h in enumerate(P['architectures']):
  sums=np.zeros(len(P['alphas']))
  for i in ids:
   p=existing(task,ia,lam,f'inner_{i}')
   assert p is not None
   rec=json.loads(p.read_text());assert rec['n_evaluation']==len(s['inner'][i]['val'])
   sums+=np.asarray(rec['utility_sum'])
  for j,a in enumerate(P['alphas']):
   candidates.append(dict(architecture=h,architecture_index=ia,alpha=a,alpha_index=j,early_stopping=True,validation_utility=float(sums[j]/n)))
 best=dict(candidates[int(np.argmax([v['validation_utility'] for v in candidates]))])
 best['candidates']=candidates;best['inner_indices']=ids;best['validation_queries']=n
 return best

def fit(task, ia, lam, fold, selected_alpha=None):
 panel,sp,rep=tasks()[task];s=split_for(task);d=load(panel,rep)
 if fold=='refit':fi=np.asarray(s['train']);va=np.asarray(s['test'])
 else:f=s['inner'][int(fold[6:])];fi=np.asarray(f['fit']);va=np.asarray(f['val'])
 assert not np.intersect1d(fi,va).size
 alphas=P['alphas'] if selected_alpha is None else [selected_alpha]
 u=d['r']-lam*d['c'];a=len(alphas);m=u.shape[1]
 y=np.broadcast_to(u[fi,None,:],(len(fi),a,m)).reshape(len(fi),a*m).astype(np.float32)
 start=time.time()
 net=BatchedMLP(P['architectures'][ia],np.repeat(alphas,m),max_iter=P['max_iter'],batch_size=P['batch_size'],tol=P['tol'],n_iter_no_change=P['n_iter_no_change'],early_stopping=True,validation_fraction=P['validation_fraction'],seed=P['seed']).fit(d['x'][fi],y)
 pred=net.predict(d['x'][va]).reshape(len(va),a,m)
 assert np.isfinite(pred).all()
 ch=pred.argmax(2)
 if selected_alpha is None:p=done_path(root_for(task),ia,True,lam,fold)
 else:p=root_for(task)/'selected_refits'/f'lambda_{lam:g}.json'
 p.parent.mkdir(parents=True,exist_ok=True)
 atomic_npz(p.with_suffix('.npz'),rows=va,choice=ch.astype(np.int16),predicted_utility=pred)
 atomic_json(p,dict(panel=panel,split=sp,representation=rep,architecture=P['architectures'][ia],early_stopping=True,alphas=alphas,lambda_value=lam,fold=fold,n_fit=len(fi),n_evaluation=len(va),utility_sum=u[va[:,None],ch].sum(0).tolist() if fold!='refit' else None,iterations=net.iterations.reshape(a,m).tolist(),losses=net.losses.reshape(a,m).tolist(),seconds=time.time()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'),split_sha256=sha(split_path(panel)),trainer_sha256=sha(Path(__file__).with_name('batched_mlp.py')),code_sha256=sha(__file__),protocol_sha256=sha(MANIFEST)))
 print('REDUCED_FIT',panel,sp,rep,ia,lam,fold,round(time.time()-start),flush=True)
 del net
 return p

def refit_path(task,lam,sel):
 p=existing(task,sel['architecture_index'],lam,'refit')
 if p is not None:return p,sel['alpha_index']
 p=root_for(task)/'selected_refits'/f'lambda_{lam:g}.json'
 if p.exists():
  d=json.loads(p.read_text())
  assert d['architecture']==sel['architecture'] and d['alphas']==[sel['alpha']]
  assert d['protocol_sha256']==sha(MANIFEST)
  return p,0
 return None,0

def pending(task):
 if (root_for(task)/'result.json').exists():return None
 panel,_,_=tasks()[task]
 for lam in CFG['lambdas'][panel]:
  for ia in range(len(P['architectures'])):
   for i in inner_ids(task):
    if existing(task,ia,lam,f'inner_{i}') is None:return (ia,lam,f'inner_{i}',None)
  sel=selection(task,lam);p,_=refit_path(task,lam,sel)
  if p is None:return (sel['architecture_index'],lam,'refit',sel['alpha'])
 return None

def assemble(task):
 root=root_for(task)
 root.mkdir(parents=True,exist_ok=True)
 if (root/'result.json').exists():return
 assert pending(task) is None
 panel,sp,rep=tasks()[task];s=split_for(task);d=load(panel,rep);te=np.asarray(s['test'])
 arrays=dict(test_rows=te,query_ids=d['query_ids'][te]);results={};sels={};sources={}
 for lam in CFG['lambdas'][panel]:
  sel=selection(task,lam);p,j=refit_path(task,lam,sel)
  with np.load(p.with_suffix('.npz')) as z:
   np.testing.assert_array_equal(z['rows'],te)
   pred=z['predicted_utility'][:,j,:];ch=z['choice'][:,j]
   assert np.isfinite(pred).all();np.testing.assert_array_equal(pred.argmax(1),ch)
   arrays[f'{lam:g}|mlp']=ch;arrays[f'{lam:g}|predicted_utility']=pred
  results[str(lam)]={'mlp':score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)}
  sels[str(lam)]=sel;sources[str(lam)]=str(p.relative_to(ROOT))
 atomic_npz(root/'predictions.npz',**arrays)
 atomic_json(root/'result.json',dict(panel=panel,split=sp,representation=rep,results=results,selections=sels,refit_sources=sources,validation_queries=sels[str(CFG['lambdas'][panel][0])]['validation_queries'],n_train=len(s['train']),n_test=len(te),protocol='reduced_v1',protocol_sha256=sha(MANIFEST),split_sha256=sha(split_path(panel)),trainer_sha256=sha(Path(__file__).with_name('batched_mlp.py')),code_sha256=sha(__file__)))
 print('REDUCED_COMPLETE',panel,sp,rep,flush=True)

def audit():
 check_sources();checks=0;completed=0;missing=[]
 for task,(panel,sp,rep) in enumerate(tasks()):
  root=root_for(task)
  if not (root/'result.json').exists():missing.append(task);continue
  rec=json.loads((root/'result.json').read_text());d=common.load(panel);s=split_for(task);te=np.asarray(s['test'])
  assert rec['protocol_sha256']==sha(MANIFEST)
  with np.load(root/'predictions.npz') as z:
   np.testing.assert_array_equal(z['test_rows'],te);np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
   for lam in CFG['lambdas'][panel]:
    expected=selection(task,lam);assert expected==rec['selections'][str(lam)]
    pred=z[f'{lam:g}|predicted_utility'];ch=z[f'{lam:g}|mlp'];assert np.isfinite(pred).all();np.testing.assert_array_equal(pred.argmax(1),ch)
    val=score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)
    for k in ['utility','reward','cost']:assert abs(val[k]-rec['results'][str(lam)]['mlp'][k])<1e-10
    checks+=1
  completed+=1
 out=dict(completed=completed,expected=46,audited_scores=checks,missing_tasks=missing,source_only_selection_verified=True)
 atomic_json(ROOT/'results/baselines/_summary_reduced/mlp_audit.json',out);print(out,flush=True)

def worker(index):
 check_sources();assert torch.cuda.is_available();start=time.monotonic()
 for task in range(index,len(tasks()),2):
  while True:
   unit=pending(task)
   if unit is None:assemble(task);break
   if STOP or time.monotonic()-start>4.75*3600:
    job=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+os.environ['SLURM_JOB_ID'],str(ROOT/'jobs/baselines_mlp_reduced.sbatch'),str(index)],text=True).strip()
    atomic_json(ROOT/'results/baselines/_logs'/f'continuation-reduced-{os.environ["SLURM_JOB_ID"]}-{index}.json',dict(kind='mlp_reduced',index=index,parent_job=os.environ['SLURM_JOB_ID'],job_id=job,submitted_at=time.time()));return
   fit(task,*unit)
  load.cache_clear()
 done=ROOT/'results/baselines/_run/reduced_finished';done.mkdir(exist_ok=True)
 (done/str(index)).write_text(os.environ['SLURM_JOB_ID'])
 print('REDUCED_WORKER_DONE',index,flush=True)
 if all((done/str(i)).exists() for i in [0,1]):
  import fcntl
  with (done/'report.lock').open('w') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   if not (done/'report.json').exists():
    job=subprocess.check_output(['sbatch','--parsable',str(ROOT/'jobs/baselines_reduced_report.sbatch')],text=True).strip();atomic_json(done/'report.json',dict(job=job))

def stop(*_):
 global STOP
 STOP=True
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);ap.add_argument('--audit',action='store_true');args=ap.parse_args()
 if args.audit:audit()
 else:
  assert args.worker in [0,1];signal.signal(signal.SIGUSR1,stop);worker(args.worker)
