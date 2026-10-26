"""KM-Prox reproduction: K=32, prior n_i/s_i and cosine exponential tilt.
Reference: https://patel-shivam.github.io/papers/proxrouter/
Report both published inverse-temperature 20 and source-validated temperature.
"""
import argparse,json,time
import numpy as np
from sklearn.cluster import KMeans
from common import ROOT,load,score,sha
TEMPS=[0.,1.,5.,10.,20.,50.,100.]

def fit(x,r,c,tr):
 km=KMeans(n_clusters=32,n_init=10,random_state=42).fit(x[tr]);lab=km.labels_;cent=km.cluster_centers_;cent=cent/np.maximum(np.linalg.norm(cent,axis=1,keepdims=True),1e-12)
 n=np.bincount(lab,minlength=32);assert (n>0).all()
 dd=np.clip(1-np.sum(x[tr]*cent[lab],axis=1),0,2);spread=np.bincount(lab,weights=dd,minlength=32)/n
 prior=n/np.maximum(spread,1e-12)
 rt=np.stack([r[tr[lab==g]].mean(0) for g in range(32)]);ct=np.stack([c[tr[lab==g]].mean(0) for g in range(32)])
 return cent,prior,rt,ct
def predict(x,model,temp):
 cent,prior,rt,ct=model;dd=np.clip(1-x@cent.T,0,2);logits=np.log(prior)[None]-temp*dd
 logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
 assert np.allclose(p.sum(1),1);return p@rt,p@ct
def run(task):
 tasks=[(panel,split,rep) for panel,nf in [('routerbench-0shot',7),('sprout-sixfam',6)] for split in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(nf)] for rep in ['gte','reader']]
 panel,split,rep=tasks[task];cfg=json.loads((ROOT/'config/protocol.json').read_text());lams=cfg['lambdas'][panel];d=load(panel,rep);s=json.loads((ROOT/'inputs'/panel/'splits.json').read_text())[split]
 tr=np.array(s['train']);te=np.array(s['test']);x,r,c,f=d['x'],d['r'],d['c'],d['fam'];dest=ROOT/'results/prox'/panel/split/rep;dest.mkdir(parents=True,exist_ok=False);t=time.time()
 vs=np.zeros((len(lams),len(TEMPS)));nval=0
 for inner in s['inner']:
  fi=np.array(inner['fit']);va=np.array(inner['val']);model=fit(x,r,c,fi);nval+=len(va)
  for it,temp in enumerate(TEMPS):
   rr,cc=predict(x[va],model,temp)
   for il,lam in enumerate(lams):
    ch=(rr-lam*cc).argmax(1);vs[il,it]+=(r[va,ch]-lam*c[va,ch]).sum()
  print('INNER',len(va),'seconds',round(time.time()-t),flush=True)
 vs/=nval;best=vs.argmax(1);model=fit(x,r,c,tr);preds=[predict(x[te],model,temp) for temp in TEMPS];results={};arrays={'test_rows':te,'query_ids':d['query_ids'][te]};selection={}
 for il,lam in enumerate(lams):
  rr,cc=preds[TEMPS.index(20.)];fixed=(rr-lam*cc).argmax(1);rr,cc=preds[best[il]];ch=(rr-lam*cc).argmax(1)
  methods={'km_prox32_temp20':fixed,'km_prox32_validated':ch};results[str(lam)]={name:score(v,r[te],c[te],f[te],lam) for name,v in methods.items()}
  for name,v in methods.items():arrays[f'{lam:g}|{name}']=v.astype(np.int16)
  selection[str(lam)]=dict(inverse_temperature=TEMPS[best[il]],validation_utilities=vs[il].tolist())
 np.savez_compressed(dest/'predictions.npz',**arrays)
 (dest/'result.json').write_text(json.dumps(dict(panel=panel,split=split,representation=rep,results=results,selections=selection,n_train=len(tr),n_test=len(te),validation_queries=nval,split_sha256=sha(ROOT/'inputs'/panel/'splits.json'),code_sha256=sha(__file__),seconds=time.time()-t,reference='https://patel-shivam.github.io/papers/proxrouter/',K=32),indent=2))
 print('DONE',panel,split,rep,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pack',type=int,required=True);a=ap.parse_args()
 run(2*a.pack);run(2*a.pack+1)
