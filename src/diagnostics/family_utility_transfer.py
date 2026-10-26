"""GTE source-family relative-utility errors at every frozen penalty."""
from pathlib import Path
import sys,json,csv,os,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,load_data,sha,unit,write_json
from geometry import Neighbours
CFG_PATH=ROOT/'configs/family_utility_transfer.json'
CFG=json.loads(CFG_PATH.read_text())
OUT=ROOT/'results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer'
def verify():
 for p,h in json.loads((OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
def rank(v,value):
 return 1+sum(x>value for x in v)+(sum(x==value for x in v)-1)/2
def load(dataset):
 d,_,original,m=load_data(dataset,'primary')
 p=ROOT/'datasets'/dataset/'prepared'
 assert m['panel_sha256']==sha(p/'panel.npz') and m['queries_sha256']==sha(p/'queries.jsonl')
 ep=ROOT/'artifacts/embeddings/gte-qwen2-7b'/dataset
 em=json.loads((ep/'manifest.json').read_text())
 assert em['query_sha256']==m['queries_sha256'] and em['vector_sha256']==sha(ep/'vectors.npy')
 assert np.array_equal(original,np.arange(len(original)))
 return d,m,ep,em
def run(dataset):
 verify();started=time.time();d,m,ep,em=load(dataset)
 r=np.asarray(d['reward'],np.float64);c=np.asarray(d['cost'],np.float64);f=d['family']
 x=unit(np.load(ep/'vectors.npy',mmap_mode='r'));ng=Neighbours(x);ng.x=ng.x.double()
 assert ng.x.device.type=='cpu','This diagnostic is scheduled CPU-only'
 records=[]
 for fold,heldout in enumerate(sorted(set(f))):
  old=ROOT/'results/geometry/gte-qwen2-7b'/dataset/'primary'/f'ood_{fold}'
  oldm=json.loads((old/'metrics.json').read_text());done=json.loads((old/'complete.json').read_text())
  assert sha(old/'metrics.json')==done['metrics_sha256'] and sha(old/'per_query.npz')==done['per_query_sha256']
  assert oldm['panel_sha256']==m['panel_sha256'] and oldm['embedding_sha256']==em['vector_sha256']
  split=np.load(old/'split.npz');tr=split['train_rows'];te=split['test_rows']
  assert np.array_equal(te,np.flatnonzero(f==heldout)) and np.array_equal(tr,np.flatnonzero(f!=heldout))
  z=np.load(old/'per_query.npz');un=z['unrestricted/neighbors']
  assert np.array_equal(z['unrestricted/query_rows'],te) and np.isin(un,tr).all()
  for oldrec in oldm['source_family_transfer']:
   family=oldrec['source_family'];a=tr[f[tr]==family]
   dest=OUT/dataset/f'ood_{fold}'/family;dest.mkdir(parents=True,exist_ok=True)
   if (dest/'result.json').exists():
    rec=json.loads((dest/'result.json').read_text())
    assert rec['config_sha256']==sha(CFG_PATH) and rec['vector_sha256']==em['vector_sha256']
    records.append(rec);continue
   allnn,allcos=ng.get(te,a,min(101,len(a)))
   nn=allnn[:,:100];cos=allcos[:,:100]
   boundary_gap=allcos[:,99]-allcos[:,100] if len(a)>100 else np.full(len(te),np.inf)
   assert np.isin(nn,a).all() and not np.isin(nn,te).any()
   assert np.all(np.diff(np.sort(nn,axis=1),axis=1)>0)
   pr=r[nn].mean(1);pc=c[nn].mean(1)
   er=pr-r[te];er-=er.mean(1,keepdims=True)
   ec=pc-c[te];ec-=ec.mean(1,keepdims=True)
   A=np.mean(er*er,axis=1);B=np.mean(er*ec,axis=1);C=np.mean(ec*ec,axis=1)
   olderr=z['source_family/'+family+'/relative_reward_mse']
   mismatch=~np.isclose(A,olderr,rtol=1e-4,atol=2e-6)
   assert np.all(boundary_gap[mismatch] <= 8*np.finfo(np.float32).eps),('Non-boundary historical mismatch',dataset,heldout,family)
   discrepancy=dict(queries=int(mismatch.sum()),max_error_difference=float(np.max(abs(A-olderr))),mean_error_difference=float(A.mean()-olderr.mean()),max_boundary_gap=float(boundary_gap[mismatch].max()) if mismatch.any() else None)
   share=float((f[un]==family).mean());assert abs(share-oldrec['retrieval_share_in_unrestricted_100'])<1e-12
   per=np.stack([A-2*lam*B+lam*lam*C for lam in CFG['lambdas'][dataset]],1)
   assert np.isfinite(per).all() and per.min()>-1e-12
   predpath=dest/'predictions.npz'
   np.savez_compressed(predpath,test_rows=te,source_rows=a,neighbours=nn.astype(np.int32),predicted_reward=pr,predicted_cost=pc,A=A,B=B,C=C,per_query_mse=per,boundary_gap=boundary_gap,historical_lambda0_error=olderr,historical_mismatch=mismatch)
   rec=dict(dataset=dataset,fold=f'ood_{fold}',heldout_family=heldout,source_family=family,
      n_test=len(te),n_source=len(a),k=min(100,len(a)),lambdas=CFG['lambdas'][dataset],
      relative_utility_mse=per.mean(0).tolist(),retrieval_share=share,
      mean_nearest_cosine=oldrec['mean_nearest_cosine'],lambda0_old_mse=oldrec['relative_reward_mse'],
      lambda0_max_query_difference=float(np.max(abs(A-olderr))),historical_numerical_discrepancy=discrepancy,
      panel_sha256=m['panel_sha256'],vector_sha256=em['vector_sha256'],
      split_sha256=sha(old/'split.npz'),predictions_sha256=sha(predpath),config_sha256=sha(CFG_PATH),
      code_sha256=sha(__file__),slurm_job=os.environ.get('SLURM_JOB_ID'))
   write_json(dest/'result.json',rec);records.append(rec)
  print('FOLD_COMPLETE',dataset,heldout,flush=True)
 write_json(OUT/dataset/'complete.json',dict(sources=len(records),elapsed_seconds=time.time()-started,config_sha256=sha(CFG_PATH)))
 print('DATASET_COMPLETE',dataset,len(records),flush=True)
def savecsv(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def report():
 verify();ranks=[];errors=[];summary=[];bylambda=[];audited=0;historical_discrepancies=[]
 for dataset in CFG['datasets']:
  d,m,ep,em=load(dataset);r=d['reward'].astype(np.float64);c=d['cost'].astype(np.float64);fam=d['family']
  groups={}
  for p in sorted((OUT/dataset).glob('ood_*/*/result.json')):
   rec=json.loads(p.read_text());z=np.load(p.parent/'predictions.npz')
   assert sha(p.parent/'predictions.npz')==rec['predictions_sha256']
   assert rec['code_sha256'] in [sha(__file__),'3d3803e5ed30a3381c1c82ad761beb0f0b9518b5c63cf00532f98482b6830566','4e99adfc35e1e02588ab60e8826d217ef3696199ba2109d9ed43d236903ae6a8'] and rec['config_sha256']==sha(CFG_PATH)
   assert rec['panel_sha256']==m['panel_sha256'] and rec['vector_sha256']==em['vector_sha256']
   te=z['test_rows'];a=z['source_rows'];nn=z['neighbours']
   if 'historical_mismatch' in z:
    mask=z['historical_mismatch'];assert np.all(z['boundary_gap'][mask]<=8*np.finfo(np.float32).eps)
    if mask.any():historical_discrepancies.append(dict(dataset=dataset,heldout=rec['heldout_family'],source=rec['source_family'],**rec['historical_numerical_discrepancy']))
   assert np.array_equal(te,np.flatnonzero(fam==rec['heldout_family']))
   assert np.array_equal(a,np.flatnonzero(fam==rec['source_family']))
   assert nn.shape==(len(te),min(100,len(a))) and np.isin(nn,a).all() and not np.isin(nn,te).any()
   assert np.all(np.diff(np.sort(nn,axis=1),axis=1)>0)
   pr=r[nn].mean(1);pc=c[nn].mean(1)
   assert np.array_equal(pr,z['predicted_reward']) and np.array_equal(pc,z['predicted_cost'])
   for j,lam in enumerate(rec['lambdas']):
    u=r[te]-lam*c[te];u-=u.mean(1,keepdims=True)
    pred=pr-lam*pc;pred-=pred.mean(1,keepdims=True)
    direct=np.mean((pred-u)**2,axis=1)
    assert np.allclose(direct,z['per_query_mse'][:,j],rtol=1e-10,atol=1e-12)
    assert np.isclose(direct.mean(),rec['relative_utility_mse'][j],rtol=1e-10,atol=1e-12)
    errors.append(dict(dataset=dataset,heldout_family=rec['heldout_family'],source_family=rec['source_family'],lambda_value=lam,relative_utility_mse=float(direct.mean()),retrieval_share=rec['retrieval_share'],n_queries=len(te)))
    audited+=1
   groups.setdefault(rec['heldout_family'],[]).append(rec)
  nf=len(set(fam));assert len(groups)==nf
  for heldout,rs in groups.items():
   assert len(rs)==nf-1 and len({x['source_family'] for x in rs})==nf-1
   assert min(rs,key=lambda v:v['relative_utility_mse'][0])['source_family']==min(rs,key=lambda v:v['lambda0_old_mse'])['source_family'],('Historical best source changed',dataset,heldout)
   for j,lam in enumerate(CFG['lambdas'][dataset]):
    es=[v['relative_utility_mse'][j] for v in rs];best=min(es)
    winners=[v for v,e in zip(rs,es) if e==best]
    shares=[v['retrieval_share'] for v in rs];cosines=[v['mean_nearest_cosine'] for v in rs]
    rr=[rank(shares,v['retrieval_share']) for v in winners]
    cr=[rank(cosines,v['mean_nearest_cosine']) for v in winners]
    ranks.append(dict(dataset=dataset,heldout_family=heldout,lambda_value=lam,n_queries=rs[0]['n_test'],n_sources=nf-1,best_sources='|'.join(v['source_family'] for v in winners),retrieval_rank=float(np.mean(rr)),top1=float(np.mean(np.asarray(rr)<=1)),top2=float(np.mean(np.asarray(rr)<=2)),cosine_rank=float(np.mean(cr)),best_mse=best,most_retrieved_mse=float(np.mean([e for v,e in zip(rs,es) if v['retrieval_share']==max(shares)]))))
  for lam in CFG['lambdas'][dataset]:
   part=[v for v in ranks if v['dataset']==dataset and v['lambda_value']==lam];weights=np.array([v['n_queries'] for v in part])
   bylambda.append(dict(dataset=dataset,lambda_value=lam,query_weighted_rank=float(np.average([v['retrieval_rank'] for v in part],weights=weights)),mean_family_rank=float(np.mean([v['retrieval_rank'] for v in part])),top1_count=float(sum(v['top1'] for v in part)),top2_count=float(sum(v['top2'] for v in part)),heldout_families=nf,n_sources=nf-1))
  part=[v for v in bylambda if v['dataset']==dataset]
  summary.append(dict(dataset=dataset,query_weighted_mean_rank=float(np.mean([v['query_weighted_rank'] for v in part])),mean_family_penalty_rank=float(np.mean([v['mean_family_rank'] for v in part])),top1_count=sum(v['top1_count'] for v in part),top2_count=sum(v['top2_count'] for v in part),family_penalty_cases=nf*len(part),n_sources=nf-1,chance_rank=(nf)/2))
 savecsv(OUT/'source_errors.csv',errors);savecsv(OUT/'per_family_ranks.csv',ranks);savecsv(OUT/'per_lambda.csv',bylambda)
 write_json(OUT/'summary.json',summary)
 write_json(OUT/'audit.json',dict(status='passed',source_penalty_scores_reconstructed=audited,expected_scores=936,split_and_source_exclusion=True,finite_predictions=True,utility_direct_vs_quadratic_verified=True,source_hashes_verified=True,lambda0_best_source_reproduction=True,lambda0_bitwise_reproduction=False,historical_numerical_discrepancies=historical_discrepancies))
 assert audited==936
 lines=['# GTE source-family relative-utility diagnostic','',
 'Best source minimizes mean squared error of centred model utilities, assessed retrospectively on the held-out family. Retrieval rank orders source families by their share of unrestricted kNN-100 neighbours. Each source predicts utilities from its own nearest min(100, source size) examples, preserving the original diagnostic. Rank1 is best. No test-selected family is deployed as a router.','',
 'Per-penalty mean ranks weight held-out families equally; query-weighted ranks are also retained. The overall mean first averages families within each penalty, then weights all13 prescribed penalties equally. This averages ranks, not MSE values across differently scaled penalties. Exact ties use average ranks; tied best sources are averaged.','',
 '| Dataset | Query-weighted rank | Equal-family/penalty rank | Top1 | Top2 | Candidate sources |',
 '|---|---:|---:|---:|---:|---:|']
 for s in summary:lines.append(f"| {s['dataset']} | {s['query_weighted_mean_rank']:.4f} | {s['mean_family_penalty_rank']:.4f} | {s['top1_count']:g}/{s['family_penalty_cases']} | {s['top2_count']:g}/{s['family_penalty_cases']} | {s['n_sources']} |")
 lines+=['','| Dataset | Lambda | Equal-family rank | Query-weighted rank | Top1 |','|---|---:|---:|---:|---:|']
 for v in bylambda:lines.append(f"| {v['dataset']} | {v['lambda_value']:g} | {v['mean_family_rank']:.4f} | {v['query_weighted_rank']:.4f} | {v['top1_count']:g}/{v['heldout_families']} |")
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,indent=2));print('AUDIT_PASSED',audited)
if __name__=='__main__':
 if sys.argv[1]=='report':report()
 else:run(sys.argv[1])
