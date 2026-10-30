"""Read-only adapters for the two previously audited, matched encoder cohorts."""
from pathlib import Path
import hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
E4=Path(os.environ.get('E4_ROOT',ROOT.parent))
PANELS=('routerbench-0shot','sprout-sixfam')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def unit(x):
 x=np.asarray(x,np.float32); norms=np.linalg.norm(x,axis=1,keepdims=True)
 assert np.isfinite(x).all() and np.isfinite(norms).all() and (norms>0).all()
 return x/norms
DATASETS={'routerbench-0shot':'routerbench','sprout-sixfam':'sprout'}
REPS={'gte':'gte-qwen2-7b','reader':'qwen2.5-7b-instruct/layer_21'}
def split_path(panel):return ROOT/'configs/baseline_splits'/f'{panel}.json'
def destination(kind,panel,split,rep):
 return ROOT/'results/baselines'/REPS[rep]/DATASETS[panel]/split/kind
def load(panel,rep=None):
 dataset=DATASETS[panel];src=ROOT/'datasets'/dataset/'prepared'
 z=dict(np.load(src/'panel.npz',allow_pickle=False))
 meta=json.loads((src/'manifest.json').read_text())
 families=sorted(set(z['family']));fam=np.searchsorted(families,z['family'])
 meta=dict(meta,families=families)
 d=dict(r=z['reward'],c=z['cost'],fam=fam,topic=z['topic'],subgroup=z['subgroup'],
        query_ids=z['keys'],meta=meta,panel=panel)
 assert d['r'].shape==d['c'].shape and np.isfinite(d['r']).all() and np.isfinite(d['c']).all()
 assert (d['c']>=0).all() and ((d['r']>=0)&(d['r']<=1)).all()
 assert len(set(d['query_ids']))==len(d['r'])
 if rep:
  p=ROOT/'artifacts/embeddings'/REPS[rep]/dataset
  m=json.loads((p/'manifest.json').read_text())
  assert m['rows']==len(d['r']) and m['query_sha256']==sha(src/'queries.jsonl')
  # Reproduction integrity fix: finite, correctly shaped but changed vectors
  # must not silently pass the manifest gate and alter the routing experiment.
  if sha(p/'vectors.npy') != m.get('vector_sha256'):
   raise ValueError('Embedding checksum differs from its manifest: '+str(p/'vectors.npy'))
  d['x']=unit(np.load(p/'vectors.npy',mmap_mode='r'))
 return d

def nearest(q,x,k,exclude=None):
 """Exact cosine retrieval. Stable row index breaks equal-similarity ties."""
 ix=[];ds=[]
 for st in range(0,len(q),128):
  sim=np.asarray(q[st:st+128],np.float32)@x.T
  if exclude is not None:
   e=np.asarray(exclude[st:st+len(sim)]);valid=e>=0;sim[np.flatnonzero(valid),e[valid]]=-np.inf
  # full stable sort defines deterministic ties (avoids argpartition boundary ambiguity)
  order=np.argsort(-sim,axis=1,kind='stable')[:,:k]
  ix.append(order);ds.append(1-np.take_along_axis(sim,order,axis=1))
 return np.concatenate(ix),np.concatenate(ds)

def score(ch,r,c,fam,lam):
 ii=np.arange(len(r));u=(r-lam*c)[ii,ch]
 return dict(utility=float(u.mean()),reward=float(r[ii,ch].mean()),cost=float(c[ii,ch].mean()),
  equal_family=float(np.mean([u[fam==g].mean() for g in np.unique(fam)])),
  per_family={str(int(g)):dict(n=int((fam==g).sum()),utility=float(u[fam==g].mean()),reward=float(r[ii,ch][fam==g].mean()),cost=float(c[ii,ch][fam==g].mean())) for g in np.unique(fam)})
