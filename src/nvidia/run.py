"""Resumable source-only layer selection and NVIDIA utility ensembles."""
import argparse,functools,json,os,signal,subprocess,time
import numpy as np
import torch
from common_nv import *
from timing import hardware
STOP=False
COMPLETED={}
@functools.lru_cache(maxsize=1)
def features(panel,sp,stage,layer):
 tr,ev=rowsets(panel,sp,stage);p=transform_dir(panel,tr,layer);info=json.loads((p/'ready.json').read_text())
 assert info['identity']['config_sha256']==sha(CONFIG) and info['identity']['train_row_hash']==sig(tr)
 x=np.load(p/'features.npy',mmap_mode='r');m=data(panel)['r'].shape[1]
 # Keep the reference's per-target feature concatenation, even for a shared layer.
 return np.tile(x[tr],(1,m)),np.tile(x[ev],(1,m)),tr,ev

def predict(net,x,device):
 out=[]
 with torch.inference_mode():
  for st in range(0,len(x),1024):out.append(net(torch.as_tensor(x[st:st+1024],device=device)).cpu().numpy())
 return np.concatenate(out)

def completed(panel,sp,layer,lam,seed,stage):
 p=fit_path(panel,sp,layer,lam,seed,stage)
 if p in COMPLETED:return COMPLETED[p]
 if p.exists():
  rec=json.loads(p.read_text());assert rec['identity']==fit_identity(panel,sp,layer,lam,seed,stage)
  assert p.with_suffix('.npz').exists()
  if stage=='final':assert p.with_suffix('.pt').exists()
  COMPLETED[p]=rec
  return rec
 return None

def fit(panel,sp,layer,lam,seed,stage):
 if completed(panel,sp,layer,lam,seed,stage):return
 fit_started=time.perf_counter()
 x,xe,tr,ev=features(panel,sp,stage,layer);assert not np.intersect1d(tr,ev).size
 feature_seconds=time.perf_counter()-fit_started
 d=data(panel);y=(d['r'][tr]-lam*d['c'][tr]).astype(np.float32);mod=trunk();start=time.time()
 torch.cuda.synchronize();phase_started=time.perf_counter()
 net=mod.train_mlp(mod.SharedTrunkNet(x.shape[1],y.shape[1]),x,y,device='cuda',seed=seed)
 torch.cuda.synchronize();training_seconds=time.perf_counter()-phase_started;phase_started=time.perf_counter()
 prediction=predict(net,xe,'cuda');assert np.isfinite(prediction).all()
 torch.cuda.synchronize();prediction_seconds=time.perf_counter()-phase_started;phase_started=time.perf_counter()
 order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed)).numpy();val=order[int(.85*len(tr)):]
 vp=predict(net,x[val],'cuda');mse=float(np.mean((vp-y[val])**2));np.testing.assert_allclose(mse,net._best_val_loss,rtol=2e-5,atol=1e-7)
 torch.cuda.synchronize();audit_seconds=time.perf_counter()-phase_started;phase_started=time.perf_counter()
 p=fit_path(panel,sp,layer,lam,seed,stage);p.parent.mkdir(parents=True,exist_ok=True)
 atomic_npz(p.with_suffix('.npz'),rows=ev,prediction=prediction)
 if stage=='final':
  tmp=p.with_suffix('.tmp.pt');torch.save(net.cpu().state_dict(),tmp);tmp.replace(p.with_suffix('.pt'))
 # Final test outcomes are not consulted in fitting or ensemble selection.
 utility_sum=None
 if stage!='final':
  choice=prediction.argmax(1);utility_sum=float((d['r'][ev]-lam*d['c'][ev])[np.arange(len(ev)),choice].sum())
 rec=dict(identity=fit_identity(panel,sp,layer,lam,seed,stage),validation_mse=mse,utility_sum=utility_sum,n_evaluation=len(ev),train_rows_hash=sig(tr),seconds=time.time()-start,job_id=os.environ.get('SLURM_JOB_ID'),input_dimensions=x.shape[1])
 rec['timing']=dict(feature_loading_seconds=feature_seconds,training_seconds=training_seconds,prediction_seconds=prediction_seconds,internal_validation_audit_seconds=audit_seconds,artifact_and_score_seconds=time.perf_counter()-phase_started,total_seconds=time.perf_counter()-fit_started,hardware=hardware(cuda=True))
 atomic_json(p,rec);print('NVIDIA_FIT',panel,sp,layer,lam,seed,stage,round(rec['seconds']),flush=True)
 del net

def select_layer(panel,sp):
 c=cfg();layers={};n=sum(len(splits(panel)[sp]['inner'][i]['val']) for i in folds(panel,sp))
 for layer in c['layers']:
  values=[]
  for lam in c['lambdas'][panel]:
   sums=0
   for i in folds(panel,sp):
    rec=completed(panel,sp,layer,lam,c['screen_seed'],f'inner_{i}');assert rec is not None
    assert rec['n_evaluation']==len(splits(panel)[sp]['inner'][i]['val']);sums+=rec['utility_sum']
   values.append(sums/n)
  layers[str(layer)]=dict(per_lambda_utility=values,mean_utility=float(np.mean(values)))
 chosen=max(c['layers'],key=lambda l:layers[str(l)]['mean_utility'])
 return dict(layer=chosen,candidate_scores=layers,validation_queries=n,inner_indices=folds(panel,sp),config_sha256=sha(CONFIG))

def next_fit(panel,sp):
 c=cfg()
 for layer in c['layers']:
  for i in folds(panel,sp):
   for lam in c['lambdas'][panel]:
    if not completed(panel,sp,layer,lam,c['screen_seed'],f'inner_{i}'):return layer,lam,c['screen_seed'],f'inner_{i}'
 selection=select_layer(panel,sp);dest=OUT/'results'/base.DATASETS[panel]/sp;atomic_json(dest/'selection.json',selection)
 for layer in sorted({21,selection['layer']}):
  for lam in c['lambdas'][panel]:
   for seed in c['ensemble_seeds']:
    if not completed(panel,sp,layer,lam,seed,'final'):return layer,lam,seed,'final'
 return None

def assemble(panel,sp):
 c=cfg();selection=select_layer(panel,sp);dest=OUT/'results'/base.DATASETS[panel]/sp;dest.mkdir(parents=True,exist_ok=True)
 te=np.asarray(splits(panel)[sp]['test']);d=data(panel);arrays=dict(test_rows=te,query_ids=d['query_ids'][te]);results={};ensembles={}
 for name,layer in [('selected_layer',selection['layer']),('fixed_l21',21)]:
  results[name]={};ensembles[name]={}
  for lam in c['lambdas'][panel]:
   records=[completed(panel,sp,layer,lam,s,'final') for s in c['ensemble_seeds']];assert all(records)
   keep=sorted(range(len(records)),key=lambda j:(records[j]['validation_mse'],c['ensemble_seeds'][j]))[:c['keep_seeds']]
   predictions=[]
   for j in keep:
    with np.load(fit_path(panel,sp,layer,lam,c['ensemble_seeds'][j],'final').with_suffix('.npz')) as z:
     np.testing.assert_array_equal(z['rows'],te);predictions.append(z['prediction'])
   pred=np.mean(np.asarray(predictions),axis=0);assert np.isfinite(pred).all();ch=pred.argmax(1)
   arrays[f'{name}|{lam:g}|prediction']=pred;arrays[f'{name}|{lam:g}|choice']=ch
   results[name][str(lam)]=base.score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)
   ensembles[name][str(lam)]=dict(layer=layer,retained_seeds=[c['ensemble_seeds'][j] for j in keep],seed_validation_mse={str(s):v['validation_mse'] for s,v in zip(c['ensemble_seeds'],records)})
 atomic_npz(dest/'predictions.npz',**arrays)
 atomic_json(dest/'result.json',dict(panel=panel,split=sp,n_test=len(te),selection=selection,ensembles=ensembles,results=results,config_sha256=sha(CONFIG)))
 print('NVIDIA_COMPLETE',panel,sp,selection['layer'],flush=True)

def worker(index):
 verify();assert torch.cuda.is_available();torch.zeros(1,device='cuda');torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 started=time.monotonic();c=cfg()
 for panel,sp in c['tasks'][index::2]:
  final=OUT/'results'/base.DATASETS[panel]/sp/'result.json'
  if final.exists():assert json.loads(final.read_text())['config_sha256']==sha(CONFIG);continue
  while True:
   unit=next_fit(panel,sp)
   if unit is None:assemble(panel,sp);break
   if STOP or time.monotonic()-started>4.75*3600:
    job=subprocess.check_output(['sbatch','--parsable','--dependency=afterany:'+os.environ['SLURM_JOB_ID'],str(ROOT/'jobs/nvidia_train.sbatch'),str(index)],text=True).strip()
    atomic_json(OUT/'_logs'/f'continuation-{os.environ["SLURM_JOB_ID"]}-{index}.json',dict(worker=index,job_id=job,parent=os.environ['SLURM_JOB_ID'],submitted_at=time.time()));return
   fit(panel,sp,*unit)
  features.cache_clear()
 finish=OUT/'finished';finish.mkdir(exist_ok=True);(finish/str(index)).write_text(os.environ['SLURM_JOB_ID'])
 if all((finish/str(i)).exists() for i in (0,1)):
  import fcntl
  with (finish/'report.lock').open('w') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   if not (finish/'report.json').exists():
    job=subprocess.check_output(['sbatch','--parsable',str(ROOT/'jobs/nvidia_report.sbatch')],text=True).strip();atomic_json(finish/'report.json',dict(job_id=job))
 print('NVIDIA_WORKER_DONE',index,flush=True)

def stop(*_):
 global STOP
 STOP=True
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True,choices=[0,1]);a=p.parse_args();signal.signal(signal.SIGUSR1,stop);worker(a.worker)
