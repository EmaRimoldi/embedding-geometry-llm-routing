"""Layer sweep under the definitive protocol: one representation (reader, layer, pooling) and one router
family, every requested outer split. Routers, retrieval, penalties, splits and k candidates are those of
astra_definitive_paper_20260923_v1/code/run_fixed.py; only the representation changes.

For every outer split this stores, per penalty, the test utility SUMS (so folds pool query-weighted) and the
inner source-validation utility sums. The inner sums are what code/select_layers.py uses to choose a layer
without touching the outer test family. Nothing here selects anything on test outcomes.

  knn    kNN with k in the protocol candidates (kNN-100 and the k-validated router are both recoverable),
         plus two outcome-free diagnostics: the share of the k=20/100 neighbours whose family has the same
         response format as the query (config/formats.json), and, on OOD splits, kNN-100 restricted to
         source families of the query's format.
  prior  the soft family prior (balanced logistic regression, family-mean outcome tables), with the
         train-selected single model, the realized oracle and, on OOD splits, a representation-free
         format lookup (mean source outcomes of families with the query's format; pooled and family-balanced).

Usage:  sweep.py --panel sprout-sixfam --reader qwen2.5-7b-instruct --layer 21 --pool last --method knn
        sweep.py --plan config/plan.json --pack 3 --npacks 40      (what the slurm array runs)
"""
import argparse,json,os,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from common_ls import ROOT,PROTOCOL,family_format,feature_dir,load_rep,sha,nearest,split_names,splits,tag

KS=PROTOCOL['knn_candidates']

def knn_preds(x,r,c,tr,te,ks):
 """Identical to run_fixed.predict_knn, but also returns the neighbour rows."""
 ids,_=nearest(x[te],x[tr],min(max(ks),len(tr)));nb_all=tr[ids];preds={}
 for k in ks:
  kk=min(k,len(tr));rr=np.empty((len(te),r.shape[1]));cc=np.empty_like(rr)
  for st in range(0,len(te),512):
   nb=nb_all[st:st+512,:kk];rr[st:st+512]=r[nb].mean(1);cc[st:st+512]=c[nb].mean(1)
  preds[k]=(rr,cc)
 return preds,nb_all

def utility_sums(rr,cc,r,c,lams):
 ii=np.arange(len(r));return [float((r-l*c)[ii,(rr-l*cc).argmax(1)].sum()) for l in lams]

def constant_sums(rt,ct,r,c,lams):
 return [float((r-l*c)[:,int(np.argmax(rt-l*ct))].sum()) for l in lams]

def run_knn(d,s,lams,fmt):
 x,r,c,f=d['x'],d['r'],d['c'],d['fam'];tr=np.array(s['train']);te=np.array(s['test']);out={}
 pr,nb=knn_preds(x,r,c,tr,te,KS)
 out['test_sum']={str(k):utility_sums(*pr[k],r[te],c[te],lams) for k in KS}
 qf=fmt[f[te]]
 for k in (20,100):out[f'same_format_share_k{k}']=float((fmt[f[nb[:,:k]]]==qf[:,None]).mean())
 out['same_family_share_k20']=float((f[nb[:,:20]]==f[te][:,None]).mean())
 inner=np.zeros((len(KS),len(lams)));vn=0
 for q in s['inner']:
  fit=np.array(q['fit']);val=np.array(q['val']);p,_=knn_preds(x,r,c,fit,val,KS);vn+=len(val)
  for ik,k in enumerate(KS):inner[ik]+=utility_sums(*p[k],r[val],c[val],lams)
 out['inner_sum']={str(k):inner[ik].tolist() for ik,k in enumerate(KS)};out['inner_n']=vn
 best=inner.argmax(0);out['validated_k']=[int(KS[b]) for b in best]
 out['test_sum']['validated']=[out['test_sum'][str(KS[b])][il] for il,b in enumerate(best)]
 if len(np.unique(f[te]))==1:   # OOD: format-matched donors only
  keep=tr[fmt[f[tr]]==fmt[f[te][0]]]
  if len(keep)>=100:
   p,_=knn_preds(x,r,c,keep,te,[100]);out['test_sum']['format_restricted_100']=utility_sums(*p[100],r[te],c[te],lams)
  out['format_restricted_pool']=int(len(keep))
 return out

def fit_prior(x,f,tr):
 sp=PROTOCOL['soft_prior']
 return LogisticRegression(C=sp['C'],class_weight='balanced',max_iter=sp['max_iter'],random_state=42).fit(x[tr],f[tr])

def prior_sums(x,r,c,f,tr,te,lams):
 clf=fit_prior(x,f,tr)
 rt=np.stack([r[tr][f[tr]==g].mean(0) for g in clf.classes_]);ct=np.stack([c[tr][f[tr]==g].mean(0) for g in clf.classes_])
 p=clf.predict_proba(x[te]);return utility_sums(p@rt,p@ct,r[te],c[te],lams),int(max(clf.n_iter_))

def run_prior(d,s,lams,fmt):
 x,r,c,f=d['x'],d['r'],d['c'],d['fam'];tr=np.array(s['train']);te=np.array(s['test']);out={'test_sum':{}}
 out['test_sum']['soft_prior'],out['iterations']=prior_sums(x,r,c,f,tr,te,lams)
 out['test_sum']['best_single_train']=constant_sums(r[tr].mean(0),c[tr].mean(0),r[te],c[te],lams)
 out['test_sum']['realized_oracle']=[float((r[te]-l*c[te]).max(1).sum()) for l in lams]
 if len(np.unique(f[te]))==1:
  src=tr[fmt[f[tr]]==fmt[f[te][0]]]
  if len(src):
   out['test_sum']['format_lookup_pooled']=constant_sums(r[src].mean(0),c[src].mean(0),r[te],c[te],lams)
   gs=np.unique(f[src]);out['test_sum']['format_lookup_balanced']=constant_sums(np.mean([r[src][f[src]==g].mean(0) for g in gs],0),np.mean([c[src][f[src]==g].mean(0) for g in gs],0),r[te],c[te],lams)
 inner=np.zeros(len(lams));vn=0
 for q in s['inner']:
  fit=np.array(q['fit']);val=np.array(q['val']);u,_=prior_sums(x,r,c,f,fit,val,lams);inner+=u;vn+=len(val)
 out['inner_sum']={'soft_prior':inner.tolist()};out['inner_n']=vn
 return out

def claim(path,stale_hours=13):
 """Atomic per-group claim so concurrent packs never compute the same cell. A claim older than the job
 time limit belongs to a dead job and may be taken over; finished splits are skipped regardless."""
 try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:
  if time.time()-path.stat().st_mtime<stale_hours*3600:return False
  path.unlink(missing_ok=True);return claim(path,stale_hours)
 os.write(fd,f"{os.environ.get('SLURM_JOB_ID','local')} {time.time()}".encode());os.close(fd);return True

def run_group(panel,reader,layer,pool,method,kind):
 name=tag(reader,layer,pool);dest=ROOT/'results'/method/panel/reader/name;todo=[sp for sp in split_names(panel,kind) if not (dest/f'{sp}.json').exists()]
 if not todo:print('SKIP complete',panel,reader,name,method,flush=True);return
 if reader!='gte' and not (feature_dir(panel,reader)/'meta.json').exists():print('MISSING features (rerun after extraction)',panel,reader,flush=True);return
 dest.mkdir(parents=True,exist_ok=True)
 if not claim(dest/f'.claim_{kind}'):print('SKIP claimed by another job',panel,reader,name,method,flush=True);return
 t=time.time();d=load_rep(panel,reader,layer,pool);S=splits(panel);lams=PROTOCOL['lambdas'][panel];fmt=family_format(panel,d['meta']['families']);code=sha(__file__)
 for sp in todo:
  t1=time.time();s=S[sp];res=(run_knn if method=='knn' else run_prior)(d,s,lams,fmt)
  te=np.array(s['test']);res.update(panel=panel,reader=reader,layer=layer,pool=pool,representation=name,method=method,split=sp,lambdas=lams,
   n_test=len(te),test_families=[d['meta']['families'][g] for g in np.unique(d['fam'][te])],feature_path=d.get('feature_path','definitive gte'),code_sha256=code,seconds=time.time()-t1)
  tmp=dest/f'{sp}.json.tmp';tmp.write_text(json.dumps(res));tmp.rename(dest/f'{sp}.json')
  print('DONE',panel,reader,name,method,sp,round(time.time()-t1),flush=True)
 print('GROUP',panel,reader,name,method,round(time.time()-t),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel');ap.add_argument('--reader');ap.add_argument('--layer',type=int);ap.add_argument('--pool',default='last')
 ap.add_argument('--method',choices=['knn','prior']);ap.add_argument('--splits',default='ood',help="ood | iid | all | explicit list joined by '+', e.g. iid_0+iid_1")
 ap.add_argument('--plan');ap.add_argument('--pack',type=int);ap.add_argument('--npacks',type=int);a=ap.parse_args()
 if a.plan:
  groups=json.loads((ROOT/a.plan).read_text())['groups']
  for g in groups[a.pack::a.npacks]:run_group(g['panel'],g['reader'],g['layer'],g['pool'],g['method'],g['splits'])
 else:run_group(a.panel,a.reader,a.layer,a.pool,a.method,a.splits)

if __name__=='__main__':main()
