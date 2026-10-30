"""Final-paper independent MLP heads; new resumable I/O, same training implementation.
One unit fits every alpha/model head for one lambda and one inner fold (or refit).
Architecture/alpha/stopping are selected using pooled inner utility only.
"""
import argparse,json,time,gc,os
from pathlib import Path
import numpy as np
import torch
from batched_mlp import BatchedMLP
from common import ROOT,load,split_path,destination,score,sha
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
CFG=json.loads((ROOT/'configs/baselines.json').read_text());P=CFG['mlp']

def tasks():
 return [(p,s,rep) for p,n in [('routerbench-0shot',7),('sprout-sixfam',6)] for s in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(n)] for rep in ['gte','reader']]
def configs():
 return [(ia,h,es) for ia,h in enumerate(P['architectures']) for es in P['early_stopping']]
def atomic_json(p,record):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False));tmp.replace(p)
def atomic_npz(p,**arrays):
 tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(p)
def done_path(root,ia,es,lam,fold):
 return root/'candidates'/f'arch{ia}_es{int(es)}'/f'lambda_{lam:g}'/f'{fold}.json'

def next_unit(task):
 panel,sp,rep=tasks()[task];s=json.loads(split_path(panel).read_text())[sp];root=destination('mlp',panel,sp,rep)
 for ia,h,es in configs():
  for lam in CFG['lambdas'][panel]:
   for fold in [f'inner_{i}' for i in range(len(s['inner']))]+['refit']:
    p=done_path(root,ia,es,lam,fold)
    if not p.exists():return (ia,h,es,lam,fold)
 return None

def run_one(task):
 assert torch.cuda.is_available(),'MLP requires a Slurm GPU'
 panel,sp,rep=tasks()[task];s=json.loads(split_path(panel).read_text())[sp];root=destination('mlp',panel,sp,rep)
 unit=next_unit(task)
 if unit is None:return False
 ia,h,es,lam,fold=unit
 d=load(panel,rep);x,r,c=d['x'],d['r'],d['c'];u=r-lam*c;m=r.shape[1];a=len(P['alphas'])
 if fold=='refit':fi=np.array(s['train']);va=np.array(s['test'])
 else:inner=s['inner'][int(fold[6:])];fi=np.array(inner['fit']);va=np.array(inner['val'])
 assert not np.intersect1d(fi,va).size
 # Flatten (alpha, model), with independent weights and identical sklearn RNG streams.
 y=np.broadcast_to(u[fi,None,:],(len(fi),a,m)).reshape(len(fi),a*m).astype(np.float32)
 alphas=np.repeat(P['alphas'],m)
 t=time.time()
 net=BatchedMLP(h,alphas,max_iter=P['max_iter'],batch_size=P['batch_size'],tol=P['tol'],n_iter_no_change=P['n_iter_no_change'],early_stopping=es,validation_fraction=P['validation_fraction'],seed=P['seed']).fit(x[fi],y)
 pred=net.predict(x[va]).reshape(len(va),a,m);assert np.isfinite(pred).all()
 ch=pred.argmax(2);values=u[va[:,None],ch]
 p=done_path(root,ia,es,lam,fold);p.parent.mkdir(parents=True,exist_ok=True)
 # Refit outcomes are only archived. They never determine candidate selection.
 atomic_npz(p.with_suffix('.npz'),rows=va,choice=ch.astype(np.int16),predicted_utility=pred)
 rec=dict(panel=panel,split=sp,representation=rep,architecture=h,early_stopping=es,alphas=P['alphas'],lambda_value=lam,fold=fold,n_fit=len(fi),n_evaluation=len(va),
          utility_sum=values.sum(0).tolist() if fold!='refit' else None,iterations=net.iterations.reshape(a,m).tolist(),losses=net.losses.reshape(a,m).tolist(),
          seconds=time.time()-t,slurm_job_id=os.environ.get('SLURM_JOB_ID'),split_sha256=sha(split_path(panel)),trainer_sha256=sha(Path(__file__).with_name('batched_mlp.py')),code_sha256=sha(__file__))
 atomic_json(p,rec)
 print('MLP_UNIT',panel,sp,rep,h,es,lam,fold,'seconds',round(time.time()-t),flush=True)
 return True

def assemble(task):
 panel,sp,rep=tasks()[task];root=destination('mlp',panel,sp,rep)
 if (root/'result.json').exists():return
 if next_unit(task) is not None:return
 d=load(panel);s=json.loads(split_path(panel).read_text())[sp];te=np.array(s['test']);nval=sum(len(f['val']) for f in s['inner'])
 results={};selections={};arrays=dict(test_rows=te,query_ids=d['query_ids'][te])
 for lam in CFG['lambdas'][panel]:
  candidates=[];scores=[]
  for ia,h,es in configs():
   sums=np.zeros(len(P['alphas']))
   for i in range(len(s['inner'])):
    v=json.loads(done_path(root,ia,es,lam,f'inner_{i}').read_text());sums+=v['utility_sum']
   for j,alpha in enumerate(P['alphas']):
    candidates.append((ia,h,es,j,alpha));scores.append(float(sums[j]/nval))
  b=int(np.argmax(scores));ia,h,es,j,alpha=candidates[b]
  z=np.load(done_path(root,ia,es,lam,'refit').with_suffix('.npz'));np.testing.assert_array_equal(z['rows'],te)
  ch=z['choice'][:,j];results[str(lam)]={'mlp':score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)}
  selections[str(lam)]=dict(architecture=h,early_stopping=es,alpha=alpha,validation_utility=scores[b],candidates=[dict(architecture=q[1],early_stopping=q[2],alpha=q[4],validation_utility=v) for q,v in zip(candidates,scores)])
  arrays[f'{lam:g}|mlp']=ch;arrays[f'{lam:g}|predicted_utility']=z['predicted_utility'][:,j,:]
 atomic_npz(root/'predictions.npz',**arrays)
 atomic_json(root/'result.json',dict(panel=panel,split=sp,representation=rep,results=results,selections=selections,validation_queries=nval,n_train=len(s['train']),n_test=len(te),split_sha256=sha(split_path(panel)),trainer_sha256=sha(Path(__file__).with_name('batched_mlp.py')),code_sha256=sha(__file__)))
 print('MLP_COMPLETE',panel,sp,rep,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--assemble',action='store_true');args=ap.parse_args()
 if not args.assemble:run_one(args.task)
 assemble(args.task)
