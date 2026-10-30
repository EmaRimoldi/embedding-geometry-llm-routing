"""Fixed GTE layer21, exact kNN100 and existing fixed Family Adaptation."""
import argparse,json,sys,os,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/baselines'))
from common import load,unit,sha,split_path,score
from run_fixed import predict_knn
OUT=ROOT/'results/diagnostics/gte_l21'
TASKS=[(p,s) for p,n in [('routerbench-0shot',7),('sprout-sixfam',6)] for s in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(n)]]
def audit(path):
 z=np.load(path/'predictions.npz',allow_pickle=False);v=json.loads((path/'result.json').read_text())
 d=load(v['panel']);s=json.loads(split_path(v['panel']).read_text())[v['split']]
 te=np.array(s['test']);assert np.array_equal(z['test_rows'],te) and np.array_equal(z['query_ids'],d['query_ids'][te])
 assert not set(s['train'])&set(te)
 for method in ['knn100','family_adaptation']:
  rr,cc=z[method+'|r'],z[method+'|c'];assert np.isfinite(rr).all() and np.isfinite(cc).all()
  for lam in v['lambdas']:
   ch=(rr-lam*cc).argmax(1);assert np.array_equal(ch,z[f'{method}|{lam:g}'])
   got=score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)['utility']
   assert abs(got-v['results'][str(lam)][method]['utility'])<1e-10
 return v
def run(task):
 panel,sp=TASKS[task];dest=OUT/panel/sp;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():audit(dest);return
 d=load(panel);dataset={'routerbench-0shot':'routerbench','sprout-sixfam':'sprout'}[panel]
 ep=ROOT/'artifacts/embeddings/gte-qwen2-7b/layer_21'/dataset
 meta=json.loads((ep/'manifest.json').read_text());feature_sha=sha(ep/'vectors.npy')
 assert meta['layer']==21 and meta['append_eos'] and not meta['truncation'] and not meta['chat_template']
 assert feature_sha==meta['vector_sha256'] and meta['rows']==len(d['r'])
 assert meta['query_sha256']==sha(ROOT/'datasets'/dataset/'prepared/queries.jsonl')
 x=unit(np.load(ep/'vectors.npy',mmap_mode='r'));r,c,f=d['r'],d['c'],d['fam']
 cfg=json.loads((ROOT/'configs/baselines.json').read_text());lams=cfg['lambdas'][panel]
 s=json.loads(split_path(panel).read_text())[sp];tr,te=np.array(s['train']),np.array(s['test'])
 if sp.startswith('ood'):assert not set(f[tr])&set(f[te])
 arrays=dict(test_rows=te,query_ids=d['query_ids'][te]);times={};t=time.time()
 rr,cc=predict_knn(x,r,c,tr,te,[100])[100];times['knn_retrieval_prediction_seconds']=time.time()-t
 preds={'knn100':(rr,cc)};pr=cfg['family_adaptation'];t=time.time()
 clf=LogisticRegression(C=pr['C'],class_weight='balanced',max_iter=pr['max_iter'],random_state=42).fit(x[tr],f[tr])
 times['prior_fit_seconds']=time.time()-t;t=time.time()
 rt=np.stack([r[tr][f[tr]==g].mean(0) for g in clf.classes_]);ct=np.stack([c[tr][f[tr]==g].mean(0) for g in clf.classes_])
 pp=clf.predict_proba(x[te]);preds['family_adaptation']=(pp@rt,pp@ct);times['prior_prediction_seconds']=time.time()-t
 results={str(l):{} for l in lams}
 for method,(rr,cc) in preds.items():
  arrays[method+'|r']=rr;arrays[method+'|c']=cc
  for lam in lams:
   ch=(rr-lam*cc).argmax(1);arrays[f'{method}|{lam:g}']=ch
   results[str(lam)][method]=score(ch,r[te],c[te],f[te],lam)
 np.savez_compressed(dest/'predictions.npz',**arrays)
 rec=dict(panel=panel,split=sp,lambdas=lams,n_test=len(te),results=results,times=times,prior_iterations=clf.n_iter_.tolist(),
 feature_sha256=feature_sha,split_sha256=sha(split_path(panel)),code_sha256=sha(__file__),baseline_code_sha256=sha(ROOT/'src/baselines/run_fixed.py'),job=os.environ.get('SLURM_JOB_ID'))
 (dest/'result.json').write_text(json.dumps(rec,indent=2));audit(dest)
 (dest/'complete.json').write_text(json.dumps(dict(audited=True,result_sha256=sha(dest/'result.json'),predictions_sha256=sha(dest/'predictions.npz'))))
 print('COMPLETE',panel,sp,times,flush=True)
def report():
 rows=[audit(OUT/p/s) for p,s in TASKS];summary=[]
 for panel in ['routerbench-0shot','sprout-sixfam']:
  for mode in ['iid','ood']:
   rs=[v for v in rows if v['panel']==panel and v['split'].startswith(mode)]
   for method in ['knn100','family_adaptation']:
    for lam in rs[0]['lambdas']:
     vals=[v['results'][str(lam)][method]['utility'] for v in rs]
     value=float(np.average(vals,weights=[v['n_test'] for v in rs] if mode=='ood' else None))
     summary.append(dict(panel=panel,mode=mode,method=method,lambda_value=lam,utility=value))
 (OUT/'report.json').write_text(json.dumps(dict(complete=True,evaluations=len(rows),method_evaluations=len(rows)*2,prediction_reconstruction_passed=True,scores=summary),indent=2))
 print('AUDITED',len(rows),'splits',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',type=int);a.add_argument('--report',action='store_true');v=a.parse_args()
 report() if v.report else run(v.task)
