"""Exceptional cross-family bridge mixed into strong kNN-100.
All pair thresholds and family baselines use outer training rows only. Known
benchmark query family is an oracle-family diagnostic. Frozen emitted GTE.
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
LAM=(0.,1.,10.,100.)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args()
 p=load_panel(a.panel);fam=p.dataset_index();F=len(p.datasets);n=p.n
 L=np.load(ROOT/'features'/a.panel/'prompt'/'lengths.npz');key='qwen_tokens' if a.panel.startswith('routerbench') else 'gte_tokens';lens=L[key];lens=lens[p.keep] if p.keep is not None and len(lens)==len(p.keep) else lens;valid=lens<8000
 x=load_rep(find_rep(a.panel,REP),n,p.keep).astype(np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 rr=np.nan_to_num(p.rewards,nan=0.0).astype(np.float32);cc=np.nan_to_num(p.costs,nan=0.0).astype(np.float32)
 out=ROOT/'results'/'exp73_centered_bridge'/a.panel
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);rows=[];arrays={};t0=time.time()
 for seed in SEEDS:
  tr,te=split(n,fam,seed);tr=np.sort(tr[valid[tr]]);te=np.sort(te[valid[te]])
  _,_,q99,low=affinity(x,fam,tr,seed);nb,sim=topk(x[te],x[tr],100);nf=fam[tr[nb]];qf=fam[te][:,None]
  cross=nf!=qf;rare=cross&(sim>=q99[qf,nf]);lowrare=rare&low[qf,nf]
  fr=np.stack([rr[tr[fam[tr]==f]].mean(0) for f in range(F)]);fc=np.stack([cc[tr[fam[tr]==f]].mean(0) for f in range(F)])
  rn=rr[tr[nb]];cn=cc[tr[nb]];base_r=rn.mean(1);base_c=cn.mean(1)
  specs=[]
  for tag,mask in [('cross',cross),('rare',rare),('lowrare',lowrare)]:
   for delta in (.005,.01,.02,.04):
    gate=mask[:,:20]&(sim[:,:20]>=sim[:,0,None]-delta);has=gate.any(1)
    for cohort in (1,3,5):
     take=gate&(np.cumsum(gate,axis=1)<=cohort);den=np.maximum(take.sum(1,keepdims=True),1)
     # Family-centered model residual for each selected neighbor, restored to query-family mean.
     br=np.einsum('ij,ijk->ik',take,rn[:,:20]+fr[fam[te],None,:]-fr[nf[:,:20]],optimize=True)/den
     bc=np.einsum('ij,ijk->ik',take,cn[:,:20]+fc[fam[te],None,:]-fc[nf[:,:20]],optimize=True)/den
     for mix in (.1,.25,.5,1.):specs.append((f'{tag}_d{delta:g}_c{cohort}_m{mix:g}',has,br,bc,mix))
  for lam in LAM:
   ute=rr[te]-lam*cc[te];bch=(base_r-lam*base_c).argmax(1);ub=ute[np.arange(len(te)),bch]
   for label,has,br,bc,mix in specs:
    pr=base_r.copy();pc=base_c.copy();pr[has]=(1-mix)*base_r[has]+mix*br[has];pc[has]=(1-mix)*base_c[has]+mix*bc[has]
    ch=(pr-lam*pc).argmax(1);u=ute[np.arange(len(te)),ch];d=u-ub
    rows.append({'seed':seed,'lambda':lam,'method':label,'delta':float(d.mean()),'active':float(has.mean()),'changed':float((ch!=bch).mean()),
      'conditional_delta_changed':float(d[ch!=bch].mean()) if (ch!=bch).any() else 0.,
      'equal_family_delta':float(np.mean([d[fam[te]==f].mean() for f in range(F)])),
      'per_family_delta':{str(f):float(d[fam[te]==f].mean()) for f in range(F)}})
  print(a.panel,seed,'done',round(time.time()-t0,1),flush=True)
 report={**p.provenance(),'experiment':'exp73_centered_bridge','representation':REP,'baseline':'knn100','valid_rows':int(valid.sum()),
  'query_family':'known benchmark family diagnostic','method':'top-20 exceptional cross-family cohort; source-family residual restored to query-family training mean and mixed into kNN100 prediction','rows':rows,'seconds':time.time()-t0}
 (out/'result.json').write_text(json.dumps(report,indent=2));print('SAVED',out,flush=True)
if __name__=='__main__':main()
