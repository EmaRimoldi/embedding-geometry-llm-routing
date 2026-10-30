"""Paired within-family bootstrap of source-family utility-error rankings."""
import sys,json,csv,os
import numpy as np
from ci_common import ROOT,OUT,CFG,load_panel,write,track,draws,record
def summarize(mse,share):
 # mse: replicates x families x penalties, share: replicates x families.
 winner=mse==mse.min(axis=1,keepdims=True)
 rank=1+(share[:,None,:]>share[:,:,None]).sum(2)+(share[:,None,:]==share[:,:,None]).sum(2)/2-.5
 # Match the point estimator in family_utility_transfer.py: Top-1 means
 # average retrieval rank <= 1. A tied maximum has rank > 1 and no credit.
 # This fixes the bootstrap/point-estimator tie inconsistency; frozen evidence
 # is preserved and the observed untied paper point estimates are unchanged.
 best=rank<=1
 return (winner*rank[:,:,None]).sum(1)/winner.sum(1),(winner*best[:,:,None]).sum(1)/winner.sum(1)
def run(task):
 ds=['routerbench','sprout'][task];d=load_panel(ds);prov={};track(ROOT/'datasets'/ds/'prepared/panel.npz',prov)
 base=ROOT/'results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer'
 folds=sorted((base/ds).glob('ood_*'));ranks=[];tops=[];point_ranks=[];point_tops=[]
 for fi,fold in enumerate(folds):
  p=ROOT/'results/geometry/gte-qwen2-7b'/ds/'primary'/fold.name/'per_query.npz';z=np.load(p);track(p,prov)
  orig=z['original_rows'];te=orig[z['unrestricted/query_rows']];nn=orig[z['unrestricted/neighbors'][:,:100]]
  mses=[];shares=[]
  for f in sorted(fold.glob('*/result.json')):
   j=json.loads(f.read_text());zz=np.load(f.parent/'predictions.npz');track(f,prov);track(f.parent/'predictions.npz',prov)
   np.testing.assert_array_equal(te,zz['test_rows']);lams=j['lambdas']
   mse=zz['per_query_mse'];np.testing.assert_allclose(mse.mean(0),j['relative_utility_mse'],rtol=1e-10,atol=1e-10)
   share=(d['family'][nn]==j['source_family']).mean(1)
   assert abs(share.mean()-j['retrieval_share'])<1e-12
   mses.append(mse);shares.append(share)
  mse=np.stack(mses,axis=1);share=np.stack(shares,axis=1)
  pr,pt=summarize(mse.mean(0)[None],share.mean(0)[None]);point_ranks.append(pr[0]);point_tops.append(pt[0])
  br=np.empty((CFG['replicates'],len(lams)));bt=np.empty_like(br)
  for st,c in draws(len(te),CFG['replicates'],CFG['seed']+fi):
   m=(c@mse.reshape(len(te),-1)/len(te)).reshape(len(c),len(mses),len(lams));s=c@share/len(te)
   br[st:st+len(c)],bt[st:st+len(c)]=summarize(m,s)
  ranks.append(br);tops.append(bt)
 pr=np.mean(point_ranks,axis=0);pt=np.sum(point_tops,axis=0);br=np.mean(ranks,axis=0);bt=np.sum(tops,axis=0)
 for old in csv.DictReader((base/'per_lambda.csv').open()):
  if old['dataset']!=ds:continue
  k=lams.index(float(old['lambda_value']))
  assert abs(pr[k]-float(old['mean_family_rank']))<1e-12
  assert abs(pt[k]-float(old['top1_count']))<1e-12
 rows=[]
 for k,lam in enumerate(lams):
  rows.extend([dict(metric='mean_rank',lambda_value=f'{lam:g}',**record(pr[k],br[:,k])),dict(metric='top1_count',lambda_value=f'{lam:g}',**record(pt[k],bt[:,k]))])
 rows.extend([dict(metric='mean_rank',lambda_value='mean',**record(pr.mean(),br.mean(1))),dict(metric='top1_count',lambda_value='mean',**record(pt.sum(),bt.sum(1)))])
 out=OUT/'family_rank'/ds;out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out/'bootstrap.npz',ranks=br,top1=bt)
 write(out/'report.json',dict(dataset=ds,complete=True,rows=rows,n_families=len(folds),n_sources=len(folds)-1,replicates=CFG['replicates'],source_hashes=prov,original_scores_reproduced=True,slurm_job=os.environ.get('SLURM_JOB_ID')))
 print('COMPLETE',ds,rows[-2:],flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
