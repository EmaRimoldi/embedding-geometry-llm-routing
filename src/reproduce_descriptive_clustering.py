"""Restore the published descriptive family-clustering protocol, separately from predictive checks."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from common import ROOT,load_data,unit,sha,write_json

def run(task):
 rep,ds,sub=task['encoder'],task['dataset'],task['subset']
 d,_,original,meta=load_data(ds,sub);f=d['family'];rng=np.random.default_rng(20260923)
 rows=np.concatenate([np.sort(rng.choice(np.flatnonzero(f==g),min(600,int((f==g).sum())),replace=False)) for g in np.unique(f)])
 path=ROOT/'artifacts/embeddings'/rep/ds
 x=unit(np.asarray(np.load(path/'vectors.npy',mmap_mode='r')[original[rows]]))
 # Same full-dimensional, ordinary KMeans call and sample rule as the historical code.
 pred=KMeans(len(np.unique(f)),n_init=10,random_state=20260923).fit_predict(x)
 hits=[]
 for start in range(0,len(x),128):
  sim=x[start:start+128]@x.T
  sim[np.arange(len(sim)),np.arange(start,start+len(sim))]=-np.inf
  nn=np.argsort(-sim,axis=1,kind='stable')[:,:20]
  hits.extend((f[rows[nn]]==f[rows[start:start+len(sim)],None]).mean(1).tolist())
 dest=ROOT/'results/geometry'/rep/ds/sub/'descriptive_clustering_original_protocol'
 dest.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest/'sample_and_clusters.npz',rows=original[rows],keys=d['keys'][rows],family=f[rows],cluster=pred)
 info=dict(**task,protocol='historical_descriptive_family_clustering',sample_rule='up to 600 random queries per family without replacement; seed 20260923',sample_rows=len(rows),full_rows=len(f),family_counts={str(g):int((f[rows]==g).sum()) for g in np.unique(f)},algorithm='sklearn.cluster.KMeans',n_init=10,random_state=20260923,space='original L2-normalized embedding, no PCA',evaluation='same sample used for fitting; descriptive, not held-out prediction',family_kmeans_nmi=float(normalized_mutual_info_score(f[rows],pred)),family_purity_k20=float(np.mean(hits)),same_family_random_chance=float(np.mean([(np.sum(f[rows]==g)-1)/(len(rows)-1) for g in f[rows]])),panel_sha256=meta['panel_sha256'],embedding_manifest_sha256=sha(path/'manifest.json'),code_sha256=sha(__file__),notes=['Original sampling and clustering procedure applied to the clean native-label dataset. Fresh row masks/order can change sample identities and scores; this is protocol reproduction, not identical historical rows.','Historical implementation reference: reproducibility/manifests/descriptive_clustering_original_source.py.txt. That script is retained as provenance only and is never executed or used to import old labels.'])
 write_json(dest/'summary.json',info);write_json(dest/'complete.json',{'summary_sha256':sha(dest/'summary.json'),'sample_sha256':sha(dest/'sample_and_clusters.npz')});print(json.dumps(info),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);args=ap.parse_args()
 run(json.loads((ROOT/'configs/descriptive_clustering_tasks.json').read_text())[args.task])
