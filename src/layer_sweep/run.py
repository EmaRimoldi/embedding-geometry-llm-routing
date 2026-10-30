"""Reuse the exact baseline routers at a different frozen layer."""
import argparse,json,sys,time,os
import numpy as np
from sklearn.linear_model import LogisticRegression
from common_layer import ROOT,CFG,BASE,OUT,PACKS,baseline,atomic,fold_dir,embedding_dir,result_dir,splits_for,verify_sources
import run_fixed

def prior_validation(d,s,lams):
 x,r,c,f=d['x'],d['r'],d['c'],d['fam']
 total=np.zeros(len(lams));n=0;folds=[]
 for inner in s['inner']:
  fi=np.asarray(inner['fit']);va=np.asarray(inner['val'])
  assert not set(fi)&set(va) and not set(va)&set(s['test'])
  if 'held_out_family' in s:assert not set(f[fi])&set(f[va])
  pr=BASE['family_adaptation']
  clf=LogisticRegression(C=pr['C'],class_weight='balanced',max_iter=pr['max_iter'],random_state=42).fit(x[fi],f[fi])
  rt=np.stack([r[fi][f[fi]==g].mean(0) for g in clf.classes_])
  ct=np.stack([c[fi][f[fi]==g].mean(0) for g in clf.classes_])
  pp=clf.predict_proba(x[va]);rr=pp@rt;cc=pp@ct
  scores=[]
  for lam in lams:
   ch=(rr-lam*cc).argmax(1);scores.append(float((r[va,ch]-lam*c[va,ch]).sum()))
  total+=scores;n+=len(va)
  folds.append(dict(n=len(va),utility_sums=scores,iterations=clf.n_iter_.tolist()))
 return total/n,folds,n

def run_fold(panel,layer,sp,s,d,feature_hash):
 dest=fold_dir(panel,layer,sp);done=dest/'result.json'
 if done.exists():
  v=json.loads(done.read_text());assert v['feature_sha256']==feature_hash
  return
 tasks=[(p,s0,rep) for p,nf in [('routerbench-0shot',7),('sprout-sixfam',6)]
        for s0 in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(nf)] for rep in ['gte','reader']]
 task=tasks.index((panel,sp,'reader'))
 for kind in ['knn','prior']:
  p=result_dir(panel,layer,sp,kind)
  if (p/'result.json').exists():continue
  assert layer!=21,('Expected completed L21 reference',p)
  if p.exists():p.rename(p.with_name(p.name+'_incomplete_'+str(time.time_ns())))
  oldload,olddest,oldargv=run_fixed.load,run_fixed.destination,sys.argv
  try:
   run_fixed.load=lambda *args,**kwargs:d
   run_fixed.destination=lambda *args,**kwargs:p
   sys.argv=['run_fixed.py','--task',str(task),'--kind',kind]
   run_fixed.main()
  finally:
   run_fixed.load,run_fixed.destination,sys.argv=oldload,olddest,oldargv
  atomic(p/'layer_provenance.json',dict(layer=layer,feature_sha256=feature_hash,
         baseline_code_sha256=baseline.sha(ROOT/'src/baselines/run_fixed.py'),
         wrapper_sha256=baseline.sha(__file__),job_id=os.environ.get('SLURM_JOB_ID')))
 lams=BASE['lambdas'][panel];knn=json.loads((result_dir(panel,layer,sp,'knn')/'result.json').read_text())
 val=np.asarray([knn['selections'][str(l)]['candidate_utilities'] for l in lams])
 assert np.isfinite(val).all()
 pv,inner,n=prior_validation(d,s,lams)
 methods=dict(knn100=val[:,BASE['knn_candidates'].index(100)].tolist(),
              knn_validated=val.max(axis=1).tolist(),family_adaptation=pv.tolist())
 records={}
 for kind in ['knn','prior']:
  p=result_dir(panel,layer,sp,kind)
  records[kind]=dict(path=str(p.relative_to(ROOT)),result_sha256=baseline.sha(p/'result.json'),
                     predictions_sha256=baseline.sha(p/'predictions.npz'))
 atomic(done,dict(panel=panel,layer=layer,split=sp,held_out_family=s.get('held_out_family'),
       test_rows=len(s['test']),validation_queries=n,lambdas=lams,
       validation_utilities=methods,prior_inner=inner,baseline_records=records,
       selected_k=[knn['selections'][str(l)]['k'] for l in lams],
       feature_sha256=feature_hash,split_sha256=baseline.sha(baseline.split_path(panel)),
       job_id=os.environ.get('SLURM_JOB_ID')))
 print('LAYER_FOLD_DONE',panel,layer,sp,flush=True)

def main(pack):
 verify_sources();panel,layer=PACKS[pack];d=baseline.load(panel)
 p=embedding_dir(panel,layer);meta=json.loads((p/'manifest.json').read_text())
 assert meta['layer']==layer and meta['rows']==len(d['r']) and meta['chat_template'] is False and meta['truncation'] is False
 assert meta['query_sha256']==baseline.sha(ROOT/'datasets'/CFG['datasets'][panel]/'prepared/queries.jsonl')
 h=baseline.sha(p/'vectors.npy');assert h==meta['vector_sha256']
 d['x']=baseline.unit(np.load(p/'vectors.npy',mmap_mode='r'))
 assert len(d['x'])==len(d['r'])
 for sp,s in splits_for(panel,layer):run_fold(panel,layer,sp,s,d,h)
 print('PACK_COMPLETE',pack,panel,layer,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pack',type=int,required=True)
 main(ap.parse_args().pack)
