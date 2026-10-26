"""Conditional paired query bootstrap for frozen paper outputs."""
from pathlib import Path
import json,hashlib,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/diagnostics/appendix_confidence_intervals'
CFG=json.loads((ROOT/'configs/appendix_ci.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def track(p,prov):
 p=Path(p);prov[str(p.relative_to(ROOT))]=sha(p)
def interval(v):return np.quantile(v,[.025,.975],axis=0)
def draws(n,reps,seed,families=None,batch=16):
 rng=np.random.default_rng(seed)
 groups=[np.arange(n)] if families is None else [np.flatnonzero(families==g) for g in np.unique(families)]
 for start in range(0,reps,batch):
  b=min(batch,reps-start);counts=np.zeros((b,n),dtype=np.float64)
  for ix in groups:
   for j in range(b):
    counts[j,ix]=np.bincount(rng.integers(len(ix),size=len(ix)),minlength=len(ix))
  yield start,counts
def boot_sums(sums,weights,families=None,reps=None,seed=None):
 sums=np.asarray(sums,np.float64);weights=np.asarray(weights,np.float64)
 assert sums.shape[0]==len(weights) and np.isfinite(sums).all() and (weights>0).all()
 out=np.empty((reps or CFG['replicates'],sums.shape[1]),np.float64)
 for st,counts in draws(len(weights),len(out),seed if seed is not None else CFG['seed'],families):
  out[st:st+len(counts)]=(counts@sums)/(counts@weights)[:,None]
 assert np.isfinite(out).all()
 return out
def load_panel(dataset):
 p=ROOT/'datasets'/dataset/'prepared/panel.npz'
 return dict(np.load(p,allow_pickle=False))
def record(point,boot):
 ci=interval(boot)
 return dict(mean=float(point),low=float(ci[0]),high=float(ci[1]))
