"""SPROUT adapter for the frozen RouterBench/LRB evaluation implementation."""
from pathlib import Path
import os,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
E4=Path(os.environ['E4_ROOT'])
sys.path.insert(0,str(ROOT/'vendor'))
from original_learning_common import unit,nearest,evaluate,contrast,code_sha
LAMS={'sprout-sixfam':(0.,.5,1.,2.,5.,10.,25.,50.,100.,250.,500.,1000.)}
FOLDS=[('sprout-sixfam',g) for g in range(6)]
REPS=('gte','reader','reader_mean')

def load(panel,rep=None):
    assert panel=='sprout-sixfam'
    src=ROOT/'inputs';z=np.load(src/'panel.npz');meta=json.loads((src/'manifest.json').read_text())
    gate=json.loads((ROOT/'canary.json').read_text());assert gate['pass'] and gate['snapshot_sha256']==meta['snapshot_sha256']
    d=dict(r=z['r'].astype(float),c=z['c'].astype(float),fam=z['fam'],missing=np.zeros_like(z['r'],dtype=bool),
           ids=z['query_ids'],panel_rows=z['panel_rows'],tokens=z['reader_tokens'],meta=meta)
    if rep is not None:
        ref=meta['representations'][rep];p=Path(ref['path']);st=p.stat()
        assert st.st_size==ref['bytes'] and st.st_mtime_ns==ref['mtime_ns'],f'Input changed: {p}'
        ix=z['gte_rows'] if rep=='gte' else z['panel_rows']
        xf=np.asarray(np.load(p,mmap_mode='r')[ix],np.float32)
        norms=np.linalg.norm(xf,axis=1);assert np.isfinite(norms).all() and (norms>0).all()
        d['x']=unit(xf)
        d['controls']=np.c_[np.log1p(d['tokens']),np.zeros(len(xf)) if rep=='gte' else np.log(norms)].astype(np.float32)
    return d
