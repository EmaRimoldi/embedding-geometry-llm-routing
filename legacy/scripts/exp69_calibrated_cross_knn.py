"""Nested family/pair-calibrated cross-family kNN on frozen emitted GTE.

Outer 70/30 family-stratified splits. Within each outer train split, an inner
70/30 split selects which query-family -> neighbor-family transfers have
positive utility on inner-held-out data. Test outcomes never select a gate.
Known benchmark family is an oracle-family diagnostic. Untruncated inputs.
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
KS=(40,100);LAM=(0.,1.,10.,100.)

def predict(rn,cn,w,lam):
 if w is None:return (rn.mean(1)-lam*cn.mean(1)).argmax(1)
 w=w.astype(np.float32,copy=False);den=np.maximum(w.sum(1,keepdims=True),1)
 return ((np.einsum('ij,ijk->ik',w,rn,optimize=True)-lam*np.einsum('ij,ijk->ik',w,cn,optimize=True))/den).argmax(1)

def score(choice,rr,cc,lam,fam,F,base=None):
 u=rr[np.arange(len(choice)),choice]-lam*cc[np.arange(len(choice)),choice]
 out={'utility':float(u.mean()),'equal_family':float(np.mean([u[fam==f].mean() for f in range(F)])),
      'per_family':{str(f):float(u[fam==f].mean()) for f in range(F)}}
 if base is not None:
  d=u-base
  out.update(delta=float(d.mean()),delta_equal_family=float(np.mean([d[fam==f].mean() for f in range(F)])),
    delta_per_family={str(f):float(d[fam==f].mean()) for f in range(F)},
    changed_choice=float(np.mean(d!=0)))
 return out,u

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args()
 p=load_panel(a.panel);fam=p.dataset_index();F=len(p.datasets);n=p.n
 lens=np.load(ROOT/'features'/a.panel/'prompt'/'lengths.npz')
 key='qwen_tokens' if a.panel.startswith('routerbench') else 'gte_tokens'
 ll=lens[key];ll=ll[p.keep] if p.keep is not None and len(ll)==len(p.keep) else ll
 valid=ll<8000
 x=load_rep(find_rep(a.panel,REP),n,p.keep).astype(np.float32)
 x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 rr=np.nan_to_num(p.rewards,nan=0.0).astype(np.float32)
 cc=np.nan_to_num(p.costs,nan=0.0).astype(np.float32)
 assert p.available.all() and np.isfinite(rr).all() and np.isfinite(cc).all()
 out=ROOT/'results'/'exp69_calibrated_cross'/a.panel
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 rows=[];t0=time.time()
 for seed in SEEDS:
  tr0,te=split(n,fam,seed);tr0=np.sort(tr0[valid[tr0]]);te=np.sort(te[valid[te]])
  i0,i1=split(len(tr0),fam[tr0],100+seed);itr=tr0[np.sort(i0)];ival=tr0[np.sort(i1)]
  _,_,q99,_=affinity(x,fam,tr0,seed)
  ni,si=topk(x[ival],x[itr],100);nt,st=topk(x[te],x[tr0],100)
  nfi=fam[itr[ni]];nft=fam[tr0[nt]]
  qi=fam[ival];qt=fam[te]
  ri=rr[itr[ni]];ci=cc[itr[ni]]
  rt=rr[tr0[nt]];ct=cc[tr0[nt]]
  for k in KS:
   fi=nfi[:,:k];ft=nft[:,:k]
   rarei=(fi!=qi[:,None])&(si[:,:k]>=q99[qi[:,None],fi])
   raret=(ft!=qt[:,None])&(st[:,:k]>=q99[qt[:,None],ft])
   clost=np.maximum(0,1-(st[:,0,None]-st[:,:k])/.04)
   sri=ri[:,:k].sum(1);sci=ci[:,:k].sum(1)
   sums=[]
   for g in range(F):
    m=(fi==g).astype(np.float32)
    sums.append((np.einsum('ij,ijk->ik',m,ri[:,:k],optimize=True),
                 np.einsum('ij,ijk->ik',m,ci[:,:k],optimize=True),m.sum(1)))
   for lam in LAM:
    bi=predict(ri[:,:k],ci[:,:k],None,lam)
    bt=predict(rt[:,:k],ct[:,:k],None,lam)
    ui=rr[ival,bi]-lam*cc[ival,bi]
    _,ut=score(bt,rr[te],cc[te],lam,qt,F)
    rows.append({'seed':int(seed),'lambda':lam,'k':k,'method':f'knn{k}',**score(bt,rr[te],cc[te],lam,qt,F,ut)[0]})
    for gamma in (2.,4.):
     # Train an all-cross gate per query family and a directed family-pair gate.
     wi=1+(gamma-1)*(fi!=qi[:,None]);ch=predict(ri[:,:k],ci[:,:k],wi,lam)
     du=rr[ival,ch]-lam*cc[ival,ch]-ui
     famgood=np.zeros(F,bool)
     for f in range(F):
      exposed=(qi==f)&(fi!=f).any(1)
      famgood[f]=(exposed.sum()>=40 and du[exposed].mean()>0.0005)
     pairgood=np.zeros((F,F),bool);pairn=np.zeros((F,F),int)
     # Targeted pair calibration: boost just g's neighbors for f's query.
     for f in range(F):
      for g in range(F):
       if f==g:continue
       exposed=(qi==f)&(fi==g).any(1)
       pairn[f,g]=int(exposed.sum())
       if pairn[f,g]<25:continue
       sr,sc,nc=sums[g]
       cand=((sri+(gamma-1)*sr-lam*(sci+(gamma-1)*sc))/(k+(gamma-1)*nc[:,None])).argmax(1)
       diff=rr[ival,cand]-lam*cc[ival,cand]-ui
       pairgood[f,g]=diff[exposed].mean()>0.0005
     gates=[('fam',famgood[qt,None]&(ft!=qt[:,None])),
            ('pair25',pairgood[qt[:,None],ft]),
            ('pair100',pairgood[qt[:,None],ft]&(pairn[qt[:,None],ft]>=100)),
            ('pairrare25',pairgood[qt[:,None],ft]&raret),
            ('pairnear25',pairgood[qt[:,None],ft]&(clost>.25))]
     for tag,mask in gates:
      w=1+(gamma-1)*mask
      pred=predict(rt[:,:k],ct[:,:k],w,lam)
      result,_=score(pred,rr[te],cc[te],lam,qt,F,ut)
      rows.append({'seed':int(seed),'lambda':lam,'k':k,'method':f'{tag}_x{gamma:g}_k{k}',
         'active_rate':float(mask.any(1).mean()),'approved_pairs':int(pairgood.sum()),
         'approved_families':int(famgood.sum()),**result})
   print(a.panel,seed,'k',k,'ready',round(time.time()-t0,1),flush=True)
 report={**p.provenance(),'experiment':'exp69_calibrated_cross','representation':REP,
  'valid_rows':int(valid.sum()),'family_label':'released benchmark family supplied on held-out queries (oracle-family diagnostic)',
  'inner_selection':'70/30 family-stratified inner split of outer training set; directed pair gain >0.0005 with at least 25 exposed inner-val queries; family at least 40',
  'rows':rows,'seconds':time.time()-t0}
 (out/'result.json').write_text(json.dumps(report,indent=2))
 print('SAVED',out/'result.json',flush=True)
if __name__=='__main__':main()
