"""Recompute descriptive original-space neighbour statistics on frozen cohorts."""
import argparse,json
import numpy as np
from common import ROOT,load,nearest,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args();dest=ROOT/'results/native_neighbors'/a.panel;dest.mkdir(parents=True,exist_ok=False);out={};arr={}
 for rep in ('gte','reader'):
  d=load(a.panel,rep);x=d['x'];f=d['fam'];rng=np.random.default_rng(20260923);res=[]
  for g in np.unique(f):
   own=np.flatnonzero(f==g);src=np.flatnonzero(f!=g);q=np.sort(rng.choice(own,min(256,len(own)),replace=False))
   ni,nd=nearest(x[q],x[src],100);oi,od=nearest(x[q],x,100,exclude=q);nb=src[ni]
   stats={}
   for k in (5,20,100):
    shares=np.stack([(f[nb[:,:k]]==h).mean(1) for h in np.unique(f)],axis=1)
    stats[str(k)]=dict(source_shares=shares.mean(0).tolist(),largest_source_share=float(shares.max(1).mean()),same_family_ordinary=float((f[oi[:,:k]]==g).mean()),random_same_family_chance=float((len(own)-1)/(len(x)-1)),mean_ood_cosine=float(1-nd[:,:k].mean()),mean_ordinary_cosine=float(1-od[:,:k].mean()))
   res.append(dict(family=d['meta']['families'][g],g=int(g),queries=len(q),stats=stats));arr[f'{rep}|{g}|query_rows']=q;arr[f'{rep}|{g}|neighbors']=nb;arr[f'{rep}|{g}|cosine']=1-nd;arr[f'{rep}|{g}|ordinary']=oi
  out[rep]=res
 overlap=[]
 for g in np.unique(f):
  assert np.array_equal(arr[f'gte|{g}|query_rows'],arr[f'reader|{g}|query_rows'])
  aa=arr[f'gte|{g}|neighbors'][:,:20];bb=arr[f'reader|{g}|neighbors'][:,:20]
  overlap.append(dict(family=d['meta']['families'][g],overlap_k20=float(np.mean([len(set(u)&set(v))/20 for u,v in zip(aa,bb)]))))
 out['overlap']=overlap;out['panel']=a.panel;out['code_sha256']=sha(__file__)
 np.savez_compressed(dest/'neighbors.npz',**arr);(dest/'summary.json').write_text(json.dumps(out,indent=2));print('DONE',a.panel,flush=True)
if __name__=='__main__':main()
