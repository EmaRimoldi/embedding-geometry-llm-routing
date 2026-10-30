from pathlib import Path
import hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/geometry.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def unit(x):
 x=np.asarray(x,dtype=np.float32);n=np.linalg.norm(x,axis=1,keepdims=True)
 assert np.isfinite(x).all() and np.isfinite(n).all() and (n>0).all()
 return x/n

def load_data(name,subset='primary'):
 p=ROOT/'datasets'/name/'prepared'; z=dict(np.load(p/'panel.npz',allow_pickle=False))
 rows=[json.loads(s) for s in (p/'queries.jsonl').open()]
 ix=np.flatnonzero(z['minilm_keep']) if subset=='shared_minilm' else np.arange(len(rows))
 return {k:v[ix] for k,v in z.items()},[rows[i] for i in ix],ix,json.loads((p/'manifest.json').read_text())
def repdir(rep):
 if rep.startswith('qwen2.5-7b-instruct/'):
  return Path(rep)
 return Path(rep)
