"""Nested validation of family-centered kNN using frozen emitted GTE.

Inner validation selects k and centering alpha independently for known-family
and nearest-centroid predicted-family variants. Plain kNN selects k on the
same inner split. Outer held-out queries provide the only reported utilities.
Per-query outputs permit paired family-stratified bootstrap intervals.
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
from exp68_cross_family_knn import topk
KS=(10,40,100);ALPHAS=(0.,.25,.5,.75,1.);LAM=(0.,1.,10.,100.)

def centroid_predict(x,index,fam,F,q):
 cen=np.stack([x[index[fam[index]==f]].mean(0) for f in range(F)]);cen/=np.maximum(np.linalg.norm(cen,axis=1,keepdims=True),1e-12)
 return (x[q]@cen.T).argmax(1)
def choices(rr,cc,nb,nf,qfam,fam_r,fam_c,k,alpha,lam):
 pred=rr[nb[:,:k]]+alpha*(fam_r[qfam,None,:]-fam_r[nf[:,:k]])
 pc=cc[nb[:,:k]]+alpha*(fam_c[qfam,None,:]-fam_c[nf[:,:k]])
 return (pred.mean(1)-lam*pc.mean(1)).argmax(1)
def util(ch,rr,cc,q,lam):return rr[q,ch]-lam*cc[q,ch]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args()
 p=load_panel(a.panel);fam=p.dataset_index();F=len(p.datasets);n=p.n
 qlen=np.load(ROOT/'features'/a.panel/'prompt'/'lengths.npz');key='qwen_tokens' if a.panel.startswith('routerbench') else 'gte_tokens'
 lens=qlen[key];lens=lens[p.keep] if p.keep is not None and len(lens)==len(p.keep) else lens;valid=lens<8000
 x=load_rep(find_rep(a.panel,REP),n,p.keep).astype(np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 rr=np.nan_to_num(p.rewards,nan=0.0).astype(np.float32);cc=np.nan_to_num(p.costs,nan=0.0).astype(np.float32)
 out=ROOT/'results'/'exp71_centered_validation'/a.panel
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);rows=[];arrays={};t0=time.time()
 for seed in SEEDS:
  otr,ote=split(n,fam,seed);otr=np.sort(otr[valid[otr]]);ote=np.sort(ote[valid[ote]])
  aa,bb=split(len(otr),fam[otr],100+seed);itr=otr[np.sort(aa)];ival=otr[np.sort(bb)]
  nbi,_=topk(x[ival],x[itr],100);nbo,_=topk(x[ote],x[otr],100)
  nfi=fam[itr[nbi]];nfo=fam[otr[nbo]]
  pfi=centroid_predict(x,itr,fam,F,ival);pfo=centroid_predict(x,otr,fam,F,ote)
  inner_r=np.stack([rr[itr[fam[itr]==f]].mean(0) for f in range(F)]);inner_c=np.stack([cc[itr[fam[itr]==f]].mean(0) for f in range(F)])
  outer_r=np.stack([rr[otr[fam[otr]==f]].mean(0) for f in range(F)]);outer_c=np.stack([cc[otr[fam[otr]==f]].mean(0) for f in range(F)])
  for lam in LAM:
   selected={};inner_scores={}
   for method,qfi,qfo in [('known',fam[ival],fam[ote]),('predicted',pfi,pfo)]:
    candidates=[]
    for k in KS:
     for alpha in ALPHAS:
      ch=choices(rr,cc,nbi,nfi,qfi,inner_r,inner_c,k,alpha,lam)
      candidates.append((float(util(ch,rr,cc,ival,lam).mean()),k,alpha))
    selected[method]=max(candidates,key=lambda z:z[0]);inner_scores[method]=candidates
   # Fair plain baseline: same inner split, select k. alpha=0 is independent of q family.
   plain=[]
   for k in KS:
    ch=choices(rr,cc,nbi,nfi,fam[ival],inner_r,inner_c,k,0.,lam)
    plain.append((float(util(ch,rr,cc,ival,lam).mean()),k,0.))
   selected['plain']=max(plain,key=lambda z:z[0]);inner_scores['plain']=plain
   values={};chs={}
   for method,qfo in [('plain',fam[ote]),('known',fam[ote]),('predicted',pfo)]:
    _,k,alpha=selected[method]
    ch=choices(rr,cc,nbo,nfo,qfo,outer_r,outer_c,k,alpha,lam);u=util(ch,rr,cc,ote,lam)
    values[method]=u;chs[method]=ch
   for method in ('plain','known','predicted'):
    u=values[method];d=u-values['plain'];_,k,alpha=selected[method]
    rows.append({'seed':seed,'lambda':lam,'method':method,'selected_k':k,'selected_alpha':alpha,
      'inner_utility':selected[method][0],'outer_utility':float(u.mean()),'delta_plain':float(d.mean()),
      'equal_family_delta':float(np.mean([d[fam[ote]==f].mean() for f in range(F)])),
      'per_family_delta':{str(f):float(d[fam[ote]==f].mean()) for f in range(F)},
      'family_accuracy':1.0 if method!='predicted' else float((pfo==fam[ote]).mean()),
      'changed_choice':float((chs[method]!=chs['plain']).mean())})
    arrays[f's{seed}_l{lam:g}_query']=ote.astype(np.int32);arrays[f's{seed}_l{lam:g}_family']=fam[ote].astype(np.int16)
    for method in ('plain','known','predicted'):
     arrays[f's{seed}_l{lam:g}_{method}_utility']=values[method].astype(np.float32)
  print(a.panel,'seed',seed,'done',round(time.time()-t0,1),flush=True)
 np.savez_compressed(out/'per_query.npz',**arrays)
 # Split-0 paired bootstrap for every lambda and centered variant.
 rng=np.random.default_rng(711);boot={}
 for lam in LAM:
  q=arrays[f's0_l{lam:g}_query'];ff=arrays[f's0_l{lam:g}_family'];groups=[np.flatnonzero(ff==f) for f in range(F)]
  for method in ('known','predicted'):
   d=arrays[f's0_l{lam:g}_{method}_utility']-arrays[f's0_l{lam:g}_plain_utility'];bs=[]
   for _ in range(2000):
    ids=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);bs.append(float(d[ids].mean()))
   boot[f'{method}@{lam:g}']={'point':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
 report={**p.provenance(),'experiment':'exp71_centered_validation','representation':REP,'valid_rows':int(valid.sum()),
  'method':'nested inner selection of k and alpha; family-centered per-model reward and cost residual; nearest train-family centroid for predicted-family variant',
  'ks':KS,'alphas':ALPHAS,'lambdas':LAM,'rows':rows,'split0_bootstrap':boot,'seconds':time.time()-t0}
 (out/'result.json').write_text(json.dumps(report,indent=2));print('SAVED',out,flush=True)
if __name__=='__main__':main()
