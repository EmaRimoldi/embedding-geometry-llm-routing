"""Exploratory cross-family neighbor weighting on frozen emitted GTE.

Five standard family-stratified splits; all rare-pair thresholds are learned
from training embeddings only. Test outcomes are accessed solely for scoring.
The benchmark family of each test query is supplied: a diagnostic upper-bound
setting, not a deployable router without family prediction. LLMRouterBench
rows with >=8000 GTE tokens are excluded before fitting/evaluation.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rb.data import load_panel
from rb.panelctx import REP
from rb.probes.sweep import SEEDS,split
from rb.reps import find_rep,load_rep
KS=(10,40,100);LAM=(0.,1.,10.,100.)

def topk(xq,xt,k):
 ids=np.empty((len(xq),k),np.int32);cos=np.empty((len(xq),k),np.float32)
 for st in range(0,len(xq),384):
  with threadpool_limits(limits=8):s=xq[st:st+384]@xt.T
  ii=np.argpartition(-s,k-1,axis=1)[:,:k]
  order=np.argsort(-np.take_along_axis(s,ii,axis=1),axis=1)
  ii=np.take_along_axis(ii,order,axis=1)
  ids[st:st+len(ii)]=ii;cos[st:st+len(ii)]=np.take_along_axis(s,ii,axis=1)
 return ids,cos

def affinity(x,fam,tr,seed):
 F=int(fam.max())+1;mean=np.full((F,F),np.nan);q95=mean.copy();q99=mean.copy()
 rng=np.random.default_rng(6800+seed)
 for f in range(F):
  af=tr[fam[tr]==f]
  for g in range(f+1,F):
   ag=tr[fam[tr]==g]
   count=4096
   ia=rng.choice(af,count);ib=rng.choice(ag,count)
   sims=np.einsum('ij,ij->i',x[ia],x[ib],optimize=True)
   mean[f,g]=mean[g,f]=float(sims.mean())
   q95[f,g]=q95[g,f]=float(np.quantile(sims,.95))
   q99[f,g]=q99[g,f]=float(np.quantile(sims,.99))
 vals=mean[np.triu_indices(F,1)]
 low=mean<=np.quantile(vals,.4)
 return mean,q95,q99,low

def report_one(choice,Ute,famte,F,base=None,active=None):
 u=Ute[np.arange(len(choice)),choice]
 z={'per_query':float(u.mean()),'equal_family':float(np.mean([u[famte==f].mean() for f in range(F)])),
    'per_family':{str(f):float(u[famte==f].mean()) for f in range(F)}}
 if base is not None:
  d=u-base
  z['delta_per_query']=float(d.mean())
  z['delta_equal_family']=float(np.mean([d[famte==f].mean() for f in range(F)]))
  z['delta_per_family']={str(f):float(d[famte==f].mean()) for f in range(F)}
  z['changed_choice']=float(np.mean(choice!=np.argmax(Ute,axis=1))) if False else None
  if active is not None:z['active_rate']=float(active.mean())
 return z,u

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);ap.add_argument('--tag',default='batch1');a=ap.parse_args()
 p=load_panel(a.panel);fam=p.dataset_index();F=len(p.datasets);n=p.n
 lengths=np.load(ROOT/'features'/a.panel/'prompt'/'lengths.npz')
 lk='qwen_tokens' if a.panel.startswith('routerbench') else 'gte_tokens'
 ls=lengths[lk];ls=ls[p.keep] if p.keep is not None and len(ls)==len(p.keep) else ls
 valid=ls<8000
 x=load_rep(find_rep(a.panel,REP),n,p.keep).astype(np.float32)
 x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 r=np.nan_to_num(p.rewards,nan=0.0).astype(np.float32)
 c=np.nan_to_num(p.costs,nan=0.0).astype(np.float32)
 assert p.available.all() and np.isfinite(r).all() and np.isfinite(c).all()
 out=ROOT/'results'/'exp68_cross_family'/a.tag/a.panel
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 rows=[];diagnostics=[];t0=time.time()
 for seed in SEEDS:
  tr,te=split(n,fam,seed);tr=np.sort(tr[valid[tr]]);te=np.sort(te[valid[te]])
  means,q95,q99,low=affinity(x,fam,tr,seed)
  nb,sim=topk(x[te],x[tr],100)
  nf=fam[tr[nb]];qf=fam[te][:,None]
  cross=nf!=qf
  rare=cross&(sim>=q99[fam[te][:,None],nf])
  lowrare=rare&low[fam[te][:,None],nf]
  near=np.maximum(0,1-(sim[:,0,None]-sim)/.04)
  rn=r[tr[nb]];cn=c[tr[nb]]
  rngate=lambda mask,delta: mask[:,:20]&(sim[:,:20]>=sim[:,0,None]-delta)
  configs=[]
  for k in KS:
   configs.append((f'knn{k}',k,np.ones((len(te),k),np.float32),None,None))
   for gamma in (2.,4.,8.):
    configs.append((f'cross_x{gamma:g}_k{k}',k,1+(gamma-1)*cross[:,:k],None,cross[:,:k].any(1)))
   configs.append((f'near_cross_x4_k{k}',k,1+3*cross[:,:k]*near[:,:k],None,cross[:,:k].any(1)))
   for gamma in (4.,12.):
    configs.append((f'rare_x{gamma:g}_k{k}',k,1+(gamma-1)*rare[:,:k],None,rare[:,:k].any(1)))
   configs.append((f'lowrare_x8_k{k}',k,1+7*lowrare[:,:k],None,lowrare[:,:k].any(1)))
  # Bridge overrides start from k=40, look in the first 20 neighbors, and
  # either route from a single cross-family peer or a small same-family cohort.
  for maskname,mask in [('rare',rare),('lowrare',lowrare),('cross',cross)]:
   for delta in (.01,.03):
    gate=rngate(mask,delta);has=gate.any(1)
    j=gate.argmax(1);candidate=nb[np.arange(len(te)),j]
    # Mean of the first three neighbors of the bridge peer's family, among top 20.
    cohort=(nf[:,:20]==nf[np.arange(len(te)),j,None]) & mask[:,:20]
    order=np.cumsum(cohort,axis=1)<=3
    cohort=cohort&order
    for mode in ('single','cohort'):
     if mode=='single': br=r[tr[candidate]];bc=c[tr[candidate]]
     else:
      w=cohort.astype(np.float32);den=np.maximum(w.sum(1,keepdims=True),1)
      br=np.einsum('ij,ijk->ik',w,rn[:,:20],optimize=True)/den
      bc=np.einsum('ij,ijk->ik',w,cn[:,:20],optimize=True)/den
     for mix in (1.,.5):
      label=f'bridge_{maskname}_d{delta:g}_{mode}_mix{mix:g}'
      configs.append((label,40,None,(br,bc,mix),has))
  pred={}
  for label,k,w,bridge,active in configs:
   if bridge is None:
    ww=np.asarray(w,dtype=np.float32)
    denom=ww.sum(1,keepdims=True)
    pr=np.einsum('ij,ijk->ik',ww,rn[:,:k],optimize=True)/denom
    pc=np.einsum('ij,ijk->ik',ww,cn[:,:k],optimize=True)/denom
   else:
    br,bc,mix=bridge
    pr0=rn[:,:40].mean(1);pc0=cn[:,:40].mean(1)
    pr=pr0.copy();pc=pc0.copy()
    pr[active]=(1-mix)*pr0[active]+mix*br[active]
    pc[active]=(1-mix)*pc0[active]+mix*bc[active]
   pred[label]=(pr,pc,k,active)
  for lam in LAM:
   ute=r[te]-lam*c[te]
   base={}
   for k in KS:
    pr,pc,_,_=pred[f'knn{k}'];ch=(pr-lam*pc).argmax(1)
    _,base[k]=report_one(ch,ute,fam[te],F)
   for label,(pr,pc,k,active) in pred.items():
    choice=(pr-lam*pc).argmax(1)
    # Record change in selected model and conditional effect on genuinely changed queries.
    bpr,bpc,_,_=pred[f'knn{k}'];basechoice=(bpr-lam*bpc).argmax(1)
    z,u=report_one(choice,ute,fam[te],F,base[k],active)
    changed=choice!=basechoice
    z['changed_choice']=float(changed.mean())
    z['changed_delta']=float((u-base[k])[changed].mean()) if changed.any() else 0.
    rows.append({'seed':int(seed),'lambda':lam,'method':label,'k':k,**z})
  diagnostics.append({'seed':int(seed),'n_train':len(tr),'n_test':len(te),'families':list(p.datasets),
   'cross_top10':float(cross[:,:10].any(1).mean()),'cross_top20':float(cross[:,:20].any(1).mean()),
   'rare_top20':float(rare[:,:20].any(1).mean()),'lowrare_top20':float(lowrare[:,:20].any(1).mean()),
   'pair_mean':means.tolist(),'pair_q99':q99.tolist()})
  print(a.panel,'seed',seed,'ready',round(time.time()-t0,1),flush=True)
 report={**p.provenance(),'experiment':'exp68_cross_family','tag':a.tag,'representation':REP,
   'valid_rows':int(valid.sum()),'family_label':'released benchmark family of held-out query (oracle family diagnostic)',
   'lambda':LAM,'ks':KS,'rows':rows,'diagnostics':diagnostics,'seconds':time.time()-t0}
 (out/'result.json').write_text(json.dumps(report,indent=2))
 print('SAVED',out/'result.json',flush=True)
if __name__=='__main__':main()
