"""Training-fold-only standardization and PCA100; no labels enter these transforms."""
import argparse,json,time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common_nv import *
from timing import hardware
def main(i):
 verify();c=cfg();panel,layer=c['pca_tasks'][i];f=features_dir(panel,layer);meta=json.loads((f/'manifest.json').read_text());d=data(panel)
 assert not meta['chat_template'] and not meta['truncation'] and meta['layer']==layer
 assert meta['rows']==len(d['r']) and meta['query_sha256']==sha(ROOT/'datasets'/base.DATASETS[panel]/'prepared/queries.jsonl')
 feature_sha=sha(f/'vectors.npy');assert feature_sha==meta['vector_sha256']
 raw=np.load(f/'vectors.npy',mmap_mode='r');assert np.isfinite(raw).all()
 for key,rows in c['source_sets'][panel].items():
  dest=transform_dir(panel,rows,layer);identity=dict(config_sha256=sha(CONFIG),feature_sha256=feature_sha,train_row_hash=key,layer=layer,panel=panel)
  if (dest/'ready.json').exists():assert json.loads((dest/'ready.json').read_text())['identity']==identity;continue
  dest.mkdir(parents=True,exist_ok=True);tr=np.asarray(rows)
  started=time.perf_counter()
  scaler=StandardScaler().fit(raw[tr]);scaled=scaler.transform(raw)
  scaling_seconds=time.perf_counter()-started;phase_started=time.perf_counter()
  pca=PCA(n_components=100,random_state=42,whiten=False).fit(scaled[tr]);pca_fit_seconds=time.perf_counter()-phase_started;phase_started=time.perf_counter()
  x=pca.transform(scaled).astype(np.float32);projection_seconds=time.perf_counter()-phase_started
  assert x.shape==(len(raw),100) and np.isfinite(x).all()
  tmp=dest/'features.tmp.npy';np.save(tmp,x);tmp.replace(dest/'features.npy')
  atomic_npz(dest/'transform.npz',train_rows=tr,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,pca_mean=pca.mean_,components=pca.components_,explained_variance=pca.explained_variance_,explained_variance_ratio=pca.explained_variance_ratio_)
  atomic_json(dest/'ready.json',dict(identity=identity,solver=pca._fit_svd_solver,explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),feature_output_sha256=sha(dest/'features.npy'),timing=dict(scaling_seconds=scaling_seconds,pca_fit_seconds=pca_fit_seconds,projection_seconds=projection_seconds,total_seconds=time.perf_counter()-started,hardware=hardware())))
  print('PCA_DONE',panel,layer,key,len(tr),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);main(p.parse_args().task)
