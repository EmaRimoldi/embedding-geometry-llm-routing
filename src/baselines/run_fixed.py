"""Exact kNN and fixed Family Adaptation; nested source validation selects k per lambda."""
import argparse,json,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import ROOT,load,nearest,score,sha,split_path,destination

def predict_knn(x,r,c,tr,te,ks):
 ids,dist=nearest(x[te],x[tr],min(max(ks),len(tr)))
 preds={}
 for k in ks:
  kk=min(k,len(tr));rr=np.empty((len(te),r.shape[1]));cc=np.empty_like(rr)
  for st in range(0,len(te),512):
   nb=tr[ids[st:st+512,:kk]];rr[st:st+512]=r[nb].mean(1);cc[st:st+512]=c[nb].mean(1)
  preds[k]=(rr,cc)
 return preds

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--kind',choices=['knn','prior'],required=True);a=ap.parse_args()
 cfg=json.loads((ROOT/'configs/baselines.json').read_text());tasks=[]
 for panel,nf in [('routerbench-0shot',7),('sprout-sixfam',6)]:
  for split in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(nf)]:
   for rep in ('gte','reader'):tasks.append((panel,split,rep))
 panel,split,rep=tasks[a.task];d=load(panel,rep);s=json.loads((split_path(panel)).read_text())[split]
 tr=np.array(s['train']);te=np.array(s['test']);x,r,c,f=d['x'],d['r'],d['c'],d['fam'];lams=cfg['lambdas'][panel]
 dest=destination(a.kind,panel,split,rep);dest.mkdir(parents=True,exist_ok=False);t=time.time()
 arrays={'test_rows':te,'query_ids':d['query_ids'][te]};results={};selections={}
 if a.kind=='knn':
  ks=cfg['knn_candidates'];vtot=np.zeros((len(lams),len(ks)));vn=0
  for inner in s['inner']:
   fit=np.array(inner['fit']);val=np.array(inner['val']);pr=predict_knn(x,r,c,fit,val,ks);vn+=len(val)
   for il,lam in enumerate(lams):
    for ik,k in enumerate(ks):
     rr,cc=pr[k];ch=(rr-lam*cc).argmax(1);vtot[il,ik]+=(r[val]-lam*c[val])[np.arange(len(val)),ch].sum()
   print('INNER',panel,split,rep,'rows',len(val),'seconds',round(time.time()-t),flush=True)
  vtot/=vn;best=vtot.argmax(1);pr=predict_knn(x,r,c,tr,te,ks)
  for k,(rr,cc) in pr.items():arrays[f'knn{k}|r']=rr;arrays[f'knn{k}|c']=cc
  for il,lam in enumerate(lams):
   rr,cc=pr[100];fixed=(rr-lam*cc).argmax(1);k=ks[best[il]];rr,cc=pr[k];validated=(rr-lam*cc).argmax(1)
   methods={'knn100':fixed,'knn_validated':validated}
   selections[str(lam)]=dict(k=int(k),validation_queries=vn,candidate_utilities=vtot[il].tolist())
   results[str(lam)]={name:score(ch,r[te],c[te],f[te],lam) for name,ch in methods.items()}
   for name,ch in methods.items():arrays[f'{lam:g}|{name}']=ch.astype(np.int16)
 else:
  prior=cfg['family_adaptation'];clf=LogisticRegression(C=prior['C'],class_weight='balanced',max_iter=prior['max_iter'],random_state=42).fit(x[tr],f[tr])
  rt=np.stack([r[tr][f[tr]==g].mean(0) for g in clf.classes_]);ct=np.stack([c[tr][f[tr]==g].mean(0) for g in clf.classes_]);p=clf.predict_proba(x[te]);rr=p@rt;cc=p@ct
  arrays['prior|r']=rr;arrays['prior|c']=cc;arrays['posterior']=p
  for lam in lams:
   ch=(rr-lam*cc).argmax(1);static=np.full(len(te),np.argmax((r[tr]-lam*c[tr]).mean(0)));oracle=(r[te]-lam*c[te]).argmax(1)
   methods={'family_adaptation':ch,'best_single_train':static,'realized_oracle':oracle}
   results[str(lam)]={name:score(v,r[te],c[te],f[te],lam) for name,v in methods.items()}
   for name,v in methods.items():arrays[f'{lam:g}|{name}']=v.astype(np.int16)
  selections['iterations']=clf.n_iter_.tolist()
 np.savez_compressed(dest/'predictions.npz',**arrays)
 (dest/'result.json').write_text(json.dumps(dict(panel=panel,split=split,representation=rep,results=results,selections=selections,n_train=len(tr),n_test=len(te),split_sha256=sha(split_path(panel)),code_sha256=sha(__file__),seconds=time.time()-t),indent=2))
 print('DONE',panel,split,rep,a.kind,round(time.time()-t),flush=True)
if __name__=='__main__':main()
