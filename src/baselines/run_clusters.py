"""Pinned Avengers-Pro algebra with source-only selection of K, top-k and alpha.
Source: LLMRouterBench c77cb050 / AvengersPro balance_cluster_router.py and
simple_cluster_router.py. Native unnormalized centroids and reciprocal-rank
aggregation are intentionally retained. All candidate pools are fixed in advance.
"""
import argparse,json,time,itertools
import numpy as np
from sklearn.cluster import KMeans
from common import ROOT,load,score,sha,split_path,destination
KS=(32,64,128);TOP=(1,3);ALPHA=tuple(np.linspace(0,1,101));BETA=9.

def table(x,r,c,tr,k,seed):
 km=KMeans(k,n_init=10,random_state=seed).fit(x[tr]);lab=km.labels_
 rt=np.stack([r[tr[lab==g]].mean(0) if np.any(lab==g) else np.zeros(r.shape[1]) for g in range(k)])
 ct=np.stack([c[tr[lab==g]].mean(0) if np.any(lab==g) else np.zeros(c.shape[1]) for g in range(k)])
 lo=rt.min(1,keepdims=True);span=rt.max(1,keepdims=True)-lo
 rn=np.divide(rt-lo,span,out=np.ones_like(rt),where=span>0)
 cn=1-np.divide(ct,ct.max(1,keepdims=True),out=np.zeros_like(ct),where=ct.max(1,keepdims=True)>0)
 return km.cluster_centers_,rt,ct,rn,cn

def weights(x,centers,top):
 dd=1-x@centers.T;ids=np.argsort(dd,axis=1)[:,:top];z=-BETA*np.take_along_axis(dd,ids,axis=1);w=np.exp(z-z.max(1,keepdims=True));return ids,w/w.sum(1,keepdims=True)
def choose(key,ids,w,model_order):
 order=np.argsort(-key[:,model_order],axis=1,kind='stable');ranks=np.empty_like(order);ranks[np.arange(len(key))[:,None],order]=np.arange(key.shape[1]);rr=np.empty_like(key);rr[:,model_order]=1/(ranks+1)
 pred=(rr[ids]*w[:,:,None]).sum(1)
 return model_order[pred[:,model_order].argmax(1)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);a=ap.parse_args();tasks=[]
 for p,nf in [('routerbench-0shot',7),('sprout-sixfam',6)]:
  for split in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(nf)]:
   for rep in ('gte','reader'):tasks.append((p,split,rep))
 panel,split,rep=tasks[a.task];cfg=json.loads((ROOT/'configs/baselines.json').read_text());s=json.loads((split_path(panel)).read_text())[split];d=load(panel,rep)
 x,r,c,f=d['x'],d['r'],d['c'],d['fam'];tr=np.array(s['train']);te=np.array(s['test']);lams=cfg['lambdas'][panel];mo=np.argsort(d['meta']['models']);seed=42
 candidates=list(itertools.product(KS,TOP,ALPHA));vt=np.zeros((len(lams),len(candidates)));vc=np.zeros((len(lams),len(KS)*len(TOP)));nval=0;t=time.time()
 dest=destination('clusters',panel,split,rep);dest.mkdir(parents=True,exist_ok=False)
 for fold in s['inner']:
  fit=np.array(fold['fit']);val=np.array(fold['val']);nval+=len(val)
  for ik,k in enumerate(KS):
   centers,rt,ct,rn,cn=table(x,r,c,fit,k,seed)
   for it,top in enumerate(TOP):
    ids,w=weights(x[val],centers,top)
    choices=np.stack([choose(al*rn+(1-al)*cn,ids,w,mo) for al in ALPHA]);vr=r[val][np.arange(len(val))[None],choices];vco=c[val][np.arange(len(val))[None],choices]
    offset=(ik*len(TOP)+it)*len(ALPHA)
    for il,lam in enumerate(lams):
     vt[il,offset:offset+len(ALPHA)]+=(vr-lam*vco).sum(1)
     ch=choose(rt-lam*ct,ids,w,mo);vc[il,ik*len(TOP)+it]+=(r[val]-lam*c[val])[np.arange(len(val)),ch].sum()
   print('INNER K',k,panel,split,rep,round(time.time()-t),flush=True)
 vt/=nval;vc/=nval;best=vt.argmax(1);bc=vc.argmax(1);fits={k:table(x,r,c,tr,k,seed) for k in KS}
 results={};arrays={'test_rows':te,'query_ids':d['query_ids'][te]};sel={}
 for il,lam in enumerate(lams):
  k,top,al=candidates[best[il]];centers,rt,ct,rn,cn=fits[k];ids,w=weights(x[te],centers,top);av=choose(al*rn+(1-al)*cn,ids,w,mo)
  ck,ctop=list(itertools.product(KS,TOP))[bc[il]];centers,rt,ct,rn,cn=fits[ck];ids,w=weights(x[te],centers,ctop);control=choose(rt-lam*ct,ids,w,mo)
  methods={'avengerspro_validated':av,'cluster_utility_validated':control};results[str(lam)]={n:score(ch,r[te],c[te],f[te],lam) for n,ch in methods.items()}
  sel[str(lam)]=dict(K=k,top=top,alpha=float(al),cluster_control_K=ck,cluster_control_top=ctop,validation_utility=float(vt[il,best[il]]))
  for name,ch in methods.items():arrays[f'{lam:g}|{name}']=ch.astype(np.int16)
 np.savez_compressed(dest/'predictions.npz',**arrays);np.savez_compressed(dest/'validation_scores.npz',avengers=vt,cluster_control=vc)
 (dest/'result.json').write_text(json.dumps(dict(panel=panel,split=split,representation=rep,results=results,selections=sel,validation_queries=nval,candidates=candidates,n_train=len(tr),n_test=len(te),split_sha256=sha(split_path(panel)),code_sha256=sha(__file__),seconds=time.time()-t),indent=2))
 print('DONE',panel,split,rep,flush=True)
if __name__=='__main__':main()
