"""Cross-family kNN with train-family-centered utility transfer.

A nearby neighbor from another family may carry a useful model-specific
residual even if its family's typical outcomes differ. The target family's
train mean is restored before selecting a model. Known benchmark query family
is an oracle-family diagnostic; no test outcome enters the centering.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from rb.data import load_panel
from rb.panelctx import REP
from rb.probes.sweep import SEEDS,split
from rb.reps import find_rep,load_rep
from exp68_cross_family_knn import topk,affinity
KS=(10,40,100);LAM=(0.,1.,10.,100.)

def evaluate(choice,rr,cc,lam,fam,F,base=None):
 u=rr[np.arange(len(choice)),choice]-lam*cc[np.arange(len(choice)),choice]
 z={'utility':float(u.mean()),'equal_family':float(np.mean([u[fam==f].mean() for f in range(F)])),
    'per_family':{str(f):float(u[fam==f].mean()) for f in range(F)}}
 if base is not None:
  d=u-base
  z.update(delta=float(d.mean()),delta_equal_family=float(np.mean([d[fam==f].mean() for f in range(F)])),
           delta_per_family={str(f):float(d[fam==f].mean()) for f in range(F)})
 return z,u

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args()
 p=load_panel(a.panel);fam=p.dataset_index();F=len(p.datasets);n=p.n
 ll=np.load(ROOT/'features'/a.panel/'prompt'/'lengths.npz')
 key='qwen_tokens' if a.panel.startswith('routerbench') else 'gte_tokens'
 lens=ll[key];lens=lens[p.keep] if p.keep is not None and len(lens)==len(p.keep) else lens
 valid=lens<8000
 x=load_rep(find_rep(a.panel,REP),n,p.keep).astype(np.float32)
 x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 rr=np.nan_to_num(p.rewards,nan=0.0).astype(np.float32)
 cc=np.nan_to_num(p.costs,nan=0.0).astype(np.float32)
 assert p.available.all() and np.isfinite(rr).all() and np.isfinite(cc).all()
 out=ROOT/'results'/'exp70_centered_cross'/a.panel
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 rows=[];t0=time.time()
 for seed in SEEDS:
  tr,te=split(n,fam,seed);tr=np.sort(tr[valid[tr]]);te=np.sort(te[valid[te]])
  _,_,q99,low=affinity(x,fam,tr,seed)
  nb,sim=topk(x[te],x[tr],100)
  nf=fam[tr[nb]];qf=fam[te][:,None]
  cross=nf!=qf;rare=cross&(sim>=q99[fam[te][:,None],nf]);lowrare=rare&low[fam[te][:,None],nf]
  train_r=np.stack([rr[tr[fam[tr]==f]].mean(0) for f in range(F)])
  train_c=np.stack([cc[tr[fam[tr]==f]].mean(0) for f in range(F)])
  rn=rr[tr[nb]];cn=cc[tr[nb]]
  shift_r=train_r[fam[te],None,:]-train_r[nf]
  shift_c=train_c[fam[te],None,:]-train_c[nf]
  for k in KS:
   # Reweight the transferred residual; no outcome enters the neighbor choice.
   specs=[('knn',0.,1.,None),('center_half',.5,1.,None),('center_full',1.,1.,None)]
   for alpha in (.5,1.):
    for mult in (2.,4.):
     for gate,mask in [('cross',cross),('rare',rare),('lowrare',lowrare)]:
      specs.append((f'center{alpha:g}_{gate}x{mult:g}',alpha,mult,mask))
   for lam in LAM:
    ute=rr[te]-lam*cc[te]
    bchoice=(rn[:,:k].mean(1)-lam*cn[:,:k].mean(1)).argmax(1)
    _,ub=evaluate(bchoice,rr[te],cc[te],lam,fam[te],F)
    for name,alpha,mult,gate in specs:
     if gate is None:w=np.ones((len(te),k),np.float32)
     else:w=(1+(mult-1)*gate[:,:k]).astype(np.float32)
     if alpha==0:pred=(rn[:,:k]-lam*cn[:,:k])
     else:pred=(rn[:,:k]+alpha*shift_r[:,:k])-lam*(cn[:,:k]+alpha*shift_c[:,:k])
     scores=np.einsum('ij,ijk->ik',w,pred,optimize=True)/w.sum(1,keepdims=True)
     choice=scores.argmax(1)
     z,_=evaluate(choice,rr[te],cc[te],lam,fam[te],F,ub)
     z['active_rate']=float((gate[:,:k].any(1) if gate is not None else cross[:,:k].any(1)).mean())
     z['changed_choice']=float((choice!=bchoice).mean())
     rows.append({'seed':int(seed),'lambda':lam,'k':k,'method':name+str(k),**z})
  print(a.panel,'seed',seed,'ready',round(time.time()-t0,1),flush=True)
 report={**p.provenance(),'experiment':'exp70_centered_cross','representation':REP,
  'valid_rows':int(valid.sum()),'query_family':'known benchmark family diagnostic',
  'center':'training-only per-family per-model reward and cost; source residual restored to query family mean',
  'rows':rows,'seconds':time.time()-t0}
 (out/'result.json').write_text(json.dumps(report,indent=2))
 print('SAVED',out/'result.json',flush=True)
if __name__=='__main__':main()
