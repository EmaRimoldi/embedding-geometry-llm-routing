"""Ordinary k-means replacement for the additional clustering and fine-group tests."""
import json,sys,os
import numpy as np
from common import ROOT,load_data,unit,write_json,sha,CONFIG
from geometry import Neighbours,outcomes,expected_composition,mean_ci
from clustering_tools import cluster_recovery
from refinements import fine_groups

def main(task):
 rep,ds,sub,fold=(task[k] for k in ['encoder','dataset','subset','fold'])
 base=ROOT/'results/geometry'/rep/ds/sub/fold;dest=base.parent/'additional_clustering_kmeans'/fold
 if (dest/'complete.json').exists():return
 assert (base/'complete.json').exists()
 d,_,original,meta=load_data(ds,sub);s=np.load(base/'split.npz');tr=np.searchsorted(original,s['train_rows']);te=np.searchsorted(original,s['test_rows'])
 assert np.array_equal(original[tr],s['train_rows']) and np.array_equal(original[te],s['test_rows']) and not np.intersect1d(tr,te).size
 ep=ROOT/'artifacts/embeddings'/rep/ds;x=unit(np.asarray(np.load(ep/'vectors.npy',mmap_mode='r')[original]));ng=Neighbours(x);ys=outcomes(d,tr);arrays={};fine={}
 for resolution in [2,4,8]:
  groups,detail=fine_groups(x,d,tr,te,resolution,42+int(fold[4:]));all_q=[];all_nn=[]
  for g in np.unique(groups[te]):
   a=tr[groups[tr]==g];b=te[groups[te]==g]
   if len(a)<10:continue
   ni,_=ng.get(b,a,10);all_q.extend(b);all_nn.extend(ni)
  q=np.asarray(all_q);nn=np.asarray(all_nn);summary=dict(group_details=detail,queries=len(q),metrics={})
  arrays[f'fine_groups/{resolution}/query_rows']=q;arrays[f'fine_groups/{resolution}/neighbors']=nn.astype(np.int32);arrays[f'fine_groups/{resolution}/assignments']=groups
  for name in ['mean_reward','solve_fraction','strict_solve_fraction','relative_reward']:
   y=ys[name];local=((y[nn]-y[q,None])**2).mean((1,2));random,_,_=expected_composition(y,q,tr,nn,groups);diff=random-local
   summary['metrics'][name]=dict(neighbor_pair_mse=float(local.mean()),same_fine_group_random_pair_mse=float(random.mean()),reduction=mean_ci(diff,CONFIG['bootstrap_repetitions']))
   arrays[f'fine_groups/{resolution}/{name}/pair_reduction']=diff.astype(np.float32)
  fine[str(resolution)]=summary
 dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'per_query.npz',original_rows=original,train_rows=tr,test_rows=te,**arrays)
 write_json(dest/'metrics.json',dict(**task,algorithm='ordinary sklearn KMeans',clustering=cluster_recovery(x,d,tr,te),fine_groups=fine,panel_sha256=meta['panel_sha256'],split_sha256=sha(base/'split.npz'),code_sha256={p:sha(ROOT/'src'/p) for p in ['run_additional_clustering.py','clustering_tools.py','refinements.py','geometry.py']},slurm_job=os.environ.get('SLURM_JOB_ID')))
 write_json(dest/'complete.json',dict(metrics_sha256=sha(dest/'metrics.json'),per_query_sha256=sha(dest/'per_query.npz')));print('COMPLETE ordinary clustering',task,flush=True)
if __name__=='__main__':
 tasks=json.loads((ROOT/'configs/additional_clustering_tasks.json').read_text());worker=int(sys.argv[1])
 for task in tasks[worker::10]:main(task)
