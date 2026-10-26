"""Independently recompute saved scores and matched OOD uncertainty.
Bootstrap intervals condition on the observed benchmark families and frozen fits;
they do not estimate generalization to a population of new benchmark families.
"""
import json
import numpy as np
from common import ROOT,load,sha

def interval(v,fam,seed=20260923,B=2000):
 rng=np.random.default_rng(seed);out=np.zeros(B);n=len(v)
 for g in np.unique(fam):
  w=v[fam==g]
  for st in range(0,B,32):
   b=min(32,B-st);out[st:st+b]+=w[rng.integers(len(w),size=(b,len(w)))].sum(1)/n
 return np.quantile(out,[.025,.975]).tolist()

def main():
 cfg=json.loads((ROOT/'config/protocol.json').read_text());report={};matched=[]
 for panel in cfg['panels']:
  d=load(panel);r,c,f=d['r'],d['c'],d['fam'];splits=json.loads((ROOT/'inputs'/panel/'splits.json').read_text());count=0;choices={};selectionchecks=0
  for kind in ['knn','prior']:
   for file in sorted((ROOT/'results'/kind/panel).glob('*/*/result.json')):
    j=json.loads(file.read_text());z=np.load(file.parent/'predictions.npz');te=np.array(splits[j['split']]['test'])
    assert np.array_equal(te,z['test_rows']) and np.array_equal(d['query_ids'][te],z['query_ids'])
    assert j['split_sha256']==sha(ROOT/'inputs'/panel/'splits.json')
    for lam in cfg['lambdas'][panel]:
     for method,record in j['results'][str(lam)].items():
      ch=z[f'{lam:g}|{method}'];assert len(ch)==len(te) and ch.min()>=0 and ch.max()<r.shape[1]
      rv=r[te,ch];cv=c[te,ch];uv=rv-lam*cv
      for k,v in [('reward',rv.mean()),('cost',cv.mean()),('utility',uv.mean())]:assert np.isclose(record[k],v,atol=1e-12,rtol=1e-10),(file,lam,method,k)
      if j['split'].startswith('ood'):
       key=(j['representation'],method,lam)
       if key not in choices:choices[key]=np.full(len(r),-1,dtype=np.int16)
       assert (choices[key][te]==-1).all();choices[key][te]=ch
      count+=1
     if kind=='knn':
      sel=j['selections'][str(lam)];assert sel['k']==cfg['knn_candidates'][np.argmax(sel['candidate_utilities'])];selectionchecks+=1
  for v in choices.values():assert (v>=0).all()
  ii=np.arange(len(r))
  for method in ['knn100','knn_validated','soft_prior']:
   for lam in cfg['lambdas'][panel]:
    ga=choices[('gte',method,lam)];la=choices[('reader',method,lam)];ug=r[ii,ga]-lam*c[ii,ga];ul=r[ii,la]-lam*c[ii,la];delta=ul-ug
    per={str(g):float(delta[f==g].mean()) for g in np.unique(f)}
    matched.append(dict(panel=panel,method=method,lambda_value=lam,n=len(r),gte_utility=float(ug.mean()),l21_utility=float(ul.mean()),delta=float(delta.mean()),ci95_fixed_family_query_bootstrap=interval(delta,f),family_deltas=per,positive_families=sum(v>0 for v in per.values()),choice_agreement=float((ga==la).mean())))
  report[panel]=dict(recomputed_score_records=count,validated_k_selections=selectionchecks,matched_query_ids=True,ood_coverage_exactly_once=True)
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 (out/'prediction_audit.json').write_text(json.dumps(dict(audit=report,code_sha256=sha(__file__),bootstrap_replicates=2000,interval_scope='Conditional on these benchmark families and fitted models; paired within-family query bootstrap. No multiplicity adjustment; exploratory individual-lambda intervals.'),indent=2))
 (out/'matched_encoder_audited.json').write_text(json.dumps(matched,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
