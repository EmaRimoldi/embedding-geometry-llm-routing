"""Read-only adapters for the two previously audited, matched encoder cohorts."""
from pathlib import Path
import hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
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
def source(panel):
 assert panel in PANELS
 return E4/('astra_reward_transfer_20260922_v1/inputs/'+panel if panel==PANELS[0] else 'astra_sprout_l21_20260922_v1/inputs')
def load(panel,rep=None):
 """Load a frozen routing panel (supplementary-material version).

 Reads the exact arrays saved by freeze.py in inputs/<panel>/panel.npz (rewards r, costs c, family ids fam,
 query_ids, topic, row maps). If rep is given, reads the query representation from
 E4_ROOT/embeddings/<panel>/<rep>.npy, whose rows must follow inputs/<panel>/panel.npz
 (see README, "Embeddings"). The original loader, which read the audited upstream cohorts,
 is kept below as _load_audited_sources for provenance.
 """
 assert panel in PANELS
 src=ROOT/'inputs'/panel; z=np.load(src/'panel.npz',allow_pickle=False)
 meta=json.loads((src/'source_manifest.json').read_text())
 d={k:z[k] for k in z.files};d['meta']=meta;d['panel']=panel
 d['r']=np.asarray(d['r'],float);d['c']=np.asarray(d['c'],float)
 assert d['r'].shape==d['c'].shape and np.isfinite(d['r']).all() and np.isfinite(d['c']).all()
 assert (d['c']>=0).all() and ((d['r']>=0)&(d['r']<=1)).all()
 assert len(set(d['query_ids']))==len(d['r'])
 if rep:
  assert rep in ('gte','reader')
  x=np.load(E4/'embeddings'/panel/f'{rep}.npy',mmap_mode='r')
  assert len(x)==len(d['r']),'embeddings/<panel>/<rep>.npy must have one row per query of inputs/<panel>/panel.npz'
  d['x']=unit(x)
 return d

def _load_audited_sources(panel,rep=None):
 src=source(panel); z=np.load(src/'panel.npz',allow_pickle=False); meta=json.loads((src/'manifest.json').read_text())
 d={k:z[k] for k in z.files};d['meta']=meta;d['panel']=panel
 d['r']=np.asarray(d['r'],float);d['c']=np.asarray(d['c'],float)
 assert d['r'].shape==d['c'].shape and np.isfinite(d['r']).all() and np.isfinite(d['c']).all()
 assert (d['c']>=0).all() and ((d['r']>=0)&(d['r']<=1)).all()
 assert len(set(d['query_ids']))==len(d['r'])
 d['topic']=np.full(len(d['r']),'',dtype='<U160')
 if panel==PANELS[0]:
  lab=E4/'astra_encoder_geometry_20260922_v1/inputs'/f'{panel}_metadata.npz'
  d['topic']=np.load(lab,allow_pickle=False)['topic']
 else:
  canary=json.loads((src.parent/'canary.json').read_text());assert canary['pass'] and canary['snapshot_sha256']==meta['snapshot_sha256']
  labels={r['key']:r.get('topic') or '' for r in map(json.loads,open(E4/'sprout/topics/authoritative_and_pending.jsonl'))}
  d['topic']=np.asarray([labels[str(k)] for k in d['query_ids']],dtype='<U160')
 if rep:
  assert rep in ('gte','reader')
  if panel==PANELS[0]: x=np.load(src/f'{rep}.npy',mmap_mode='r')
  else:
   ref=meta['representations'][rep];path=Path(ref['path']);st=path.stat()
   assert st.st_size==ref['bytes'] and st.st_mtime_ns==ref['mtime_ns'],'Frozen embedding source changed'
   x=np.load(path,mmap_mode='r')[d['gte_rows'] if rep=='gte' else d['panel_rows']]
  assert len(x)==len(d['r']);d['x']=unit(x)
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
