from pathlib import Path
import hashlib,json,os
import numpy as np

ROOT=Path(__file__).resolve().parent
E4=Path(os.environ.get('E4_ROOT',ROOT.parent))
OLD=E4/'astra_reward_transfer_20260922_v1'
GEO=E4/'astra_encoder_geometry_20260922_v1'
LAMS={'routerbench-0shot':(0.,1.,10.,50.,100.),'llmrouterbench-performance-cost':(0.,.5,1.,2.,5.,10.)}
FOLDS=[(p,g) for p,n in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)] for g in range(n)]
REPS=('gte','reader','reader_mean','reader_last8')

def unit(x):
    x=np.asarray(x,np.float32);assert np.isfinite(x).all()
    norm=np.linalg.norm(x,axis=1,keepdims=True);assert np.isfinite(norm).all() and (norm>0).all()
    return x/norm

def load(panel,rep=None):
    src=OLD/'inputs'/panel;z=np.load(src/'panel.npz');meta=json.loads((src/'manifest.json').read_text())
    d=dict(r=z['r'].astype(float),c=z['c'].astype(float),fam=z['fam'],missing=z['missing'],
           ids=z['query_ids'],panel_rows=z['panel_rows'],tokens=z['reader_tokens'],meta=meta)
    if rep is not None:
        if rep in ('gte','reader'):x=np.load(src/f'{rep}.npy',mmap_mode='r')
        else:
            pool=rep.removeprefix('reader_');path=E4/'features'/panel/'prompt/hidden/qwen2.5-7b-instruct'/f'L21_{pool}.npy'
            pm=json.loads((path.parent/'meta.json').read_text());assert pm['rows']==meta['original_rows']
            x=np.load(path,mmap_mode='r')[z['panel_rows']]
        assert len(x)==len(d['r'])
        xf=np.asarray(x,np.float32)
        norms=np.linalg.norm(xf,axis=1)
        d['x']=unit(xf)
        # GTE's explicit Normalize module makes norm a numerical artifact.
        d['controls']=np.c_[np.log1p(d['tokens']),np.zeros(len(xf)) if rep=='gte' else np.log(norms)].astype(np.float32)
    return d

def nearest(q,x,k=100):
    q=np.asarray(q,np.float32);x=np.asarray(x,np.float32);out=[];dist=[];sq=(x*x).sum(1)
    for i in range(0,len(q),256):
        a=q[i:i+256];dd=(a*a).sum(1)[:,None]+sq[None]-2*a@x.T;dd=np.maximum(dd,0)
        ix=np.argpartition(dd,k-1,axis=1)[:,:k];dv=np.take_along_axis(dd,ix,axis=1);order=np.argsort(dv,axis=1)
        out.append(np.take_along_axis(ix,order,axis=1));dist.append(np.take_along_axis(dv,order,axis=1))
    return np.concatenate(out),np.concatenate(dist)

def evaluate(ch,r,c,missing,lam):
    i=np.arange(len(r));return dict(utility=float((r-lam*c)[i,ch].mean()),reward=float(r[i,ch].mean()),cost=float(c[i,ch].mean()),missing_choice_rate=float(missing[i,ch].mean()))

def contrast(a,b,r,c,lam):
    i=np.arange(len(r));dr=r[i,a]-r[i,b];dc=c[i,a]-c[i,b];du=dr-lam*dc;changed=a!=b
    return dict(agreement=float((~changed).mean()),utility_difference=float(du.mean()),
        reward_contribution=float(dr.mean()),cost_contribution=float((-lam*dc).mean()),
        changed_queries=int(changed.sum()),conditional_utility_difference=float(du[changed].mean()) if changed.any() else None,
        improved_fraction=float((du>1e-9).mean()),harmed_fraction=float((du < -1e-9).mean()))

def code_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
