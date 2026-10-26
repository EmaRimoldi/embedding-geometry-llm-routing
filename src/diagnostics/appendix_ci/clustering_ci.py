"""Conditional bootstrap of train-fitted subject NMI and ARI."""
import sys,json,os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from ci_common import ROOT,OUT,CFG,write,track,draws,record
def metrics(c):
 n=c.sum();a=c.sum(1);b=c.sum(0);nz=c>0
 mi=((c[nz]/n)*np.log(c[nz]*n/(a[:,None]*b[None,:])[nz])).sum()
 ha=-((a[a>0]/n)*np.log(a[a>0]/n)).sum();hb=-((b[b>0]/n)*np.log(b[b>0]/n)).sum()
 nmi=2*mi/(ha+hb) if ha+hb>0 else 1.
 choose=lambda v:(v*(v-1)/2).sum()
 comb=n*(n-1)/2;sa=choose(a);sb=choose(b);expected=sa*sb/comb if comb else 0.;den=(sa+sb)/2-expected
 ari=(choose(c)-expected)/den if den else 1.
 return np.array([nmi,ari])
def run(task):
 condition=task//2;ds=['routerbench','sprout'][task%2]
 ids=list(range(condition*10+(task%2)*5,condition*10+(task%2)*5+5));prov={};folds=[]
 for i in ids:
  p=OUT/'clustering_assignments'/str(i);j=json.loads((p/'report.json').read_text());assert j['original_scores_reproduced']
  z=dict(np.load(p/'assignments.npz'));folds.append((j,z));track(p/'assignments.npz',prov);track(p/'report.json',prov)
 rows=[];arrays={}
 for family in [v['family'] for v in folds[0][0]['records']]:
  query_ids=np.unique(np.concatenate([z[family+'|rows'] for j,z in folds]));forms=[];original=[]
  for j,z in folds:
   a=np.unique(z[family+'|truth'],return_inverse=True)[1];b=np.unique(z[family+'|pred'],return_inverse=True)[1]
   shape=(a.max()+1,b.max()+1);flat=a*shape[1]+b
   c=np.bincount(flat,minlength=np.prod(shape)).reshape(shape);m=metrics(c)
   np.testing.assert_allclose(m,[normalized_mutual_info_score(a,b),adjusted_rand_score(a,b)],atol=1e-12)
   original.append(m);forms.append((np.searchsorted(query_ids,z[family+'|rows']),flat,shape))
  boot=np.empty((CFG['replicates'],2))
  for st,counts in draws(len(query_ids),CFG['replicates'],CFG['seed']):
   for j,cnt in enumerate(counts):
    ms=[]
    for idx,flat,shape in forms:
     c=np.bincount(flat,weights=cnt[idx],minlength=np.prod(shape)).reshape(shape);assert c.sum()>1
     ms.append(metrics(c))
    boot[st+j]=np.mean(ms,axis=0)
  point=np.mean(original,axis=0)
  for k,metric in enumerate(['nmi','ari']):rows.append(dict(family=family,metric=metric,**record(point[k],boot[:,k])))
  arrays[family]=boot
 out=OUT/'clustering'/str(task);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'bootstrap.npz',**arrays)
 write(out/'report.json',dict(dataset=ds,encoder=folds[0][0]['encoder'],subset=folds[0][0]['subset'],complete=True,rows=rows,source_hashes=prov,replicates=CFG['replicates'],slurm_job=os.environ.get('SLURM_JOB_ID')))
 print('COMPLETE',task,flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
