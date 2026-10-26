"""Predeclared final neighbour explanation, using frozen OOD kNN-100 runs.
No target outcome participates in neighbour choice, random draws or fitting.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parent
sys.path.insert(0,str(BASE/'code'))
from common import load,sha
from metrics import profiles,routing,summarize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);a=ap.parse_args();t=time.time()
 d=load(a.panel);r,c,f=d['r'],d['c'],d['fam'];n,m=r.shape;lams=json.loads((BASE/'config/protocol.json').read_text())['lambdas'][a.panel]
 dest=HERE/'results'/a.panel;dest.mkdir(parents=True,exist_ok=False)
 source_files=[BASE/'inputs'/a.panel/'splits.json',BASE/'config/protocol.json',BASE/'results/native_neighbors'/a.panel/'neighbors.npz']
 splits=json.loads(source_files[0].read_text());pop={int(g):int((f==g).sum()) for g in np.unique(f)};pred={rep:(np.empty_like(r),np.empty_like(c)) for rep in ['gte','reader']};seen=np.zeros(n,int)
 for split,s in splits.items():
  if not split.startswith('ood'):continue
  te=np.array(s['test']);tr=np.array(s['train']);assert not np.intersect1d(f[tr],f[te]).size;seen[te]+=1
  for rep in pred:
   path=BASE/'results/knn'/a.panel/split/rep/'predictions.npz';z=np.load(path);source_files.append(path)
   assert np.array_equal(z['test_rows'],te) and np.array_equal(z['query_ids'],d['query_ids'][te]);pred[rep][0][te]=z['knn100|r'];pred[rep][1][te]=z['knn100|c']
 assert (seen==1).all();full_arrays={};contrasts={};metrics={}
 for rep,(rr,cc) in pred.items():
  metrics[rep]=profiles(r,c,rr,cc)
  for k,v in metrics[rep].items():full_arrays[f'{rep}|{k}']=v
 for k in metrics['gte']:contrasts[f'Qwen_minus_GTE|{k}']=metrics['reader'][k]-metrics['gte'][k]
 rg,cg=pred['gte'];rq,cq=pred['reader']
 for lam in lams:
  policies={}
  for name,rr,cc in [('gte',rg,cg),('reader',rq,cq),('gte_reward_qwen_cost',rg,cq),('qwen_reward_gte_cost',rq,cg)]:
   vals,ch=routing(r,c,rr,cc,lam);policies[name]=vals
   full_arrays[f'{lam:g}|{name}|choice']=ch
   for k,v in vals.items():full_arrays[f'{lam:g}|{name}|{k}']=v
  for k in policies['gte']:contrasts[f'{lam:g}|Qwen_minus_GTE|{k}']=policies['reader'][k]-policies['gte'][k]
  gg=policies['gte']['utility'];qq=policies['reader']['utility'];gc=policies['gte_reward_qwen_cost']['utility'];qg=policies['qwen_reward_gte_cost']['utility']
  ce=.5*(gc-gg+qq-qg);re=.5*(qg-gg+qq-gc);assert np.allclose(ce+re,qq-gg,atol=1e-12)
  contrasts[f'{lam:g}|cost_swap_contribution']=ce;contrasts[f'{lam:g}|reward_swap_contribution']=re
 np.savez_compressed(dest/'full_query_metrics.npz',query_ids=d['query_ids'],family=f,**full_arrays,**contrasts)
 print('FULL_PREDICTIONS',a.panel,'seconds',round(time.time()-t),flush=True)
 # Fixed outcome-independent geometry sample already used in native-neighbour plots.
 z=np.load(BASE/'results/native_neighbors'/a.panel/'neighbors.npz');rows=[];neighbors={rep:[] for rep in pred}
 for g in np.unique(f):
  qa=z[f'gte|{g}|query_rows'];qb=z[f'reader|{g}|query_rows'];assert np.array_equal(qa,qb);rows.extend(qa.tolist())
  for rep in pred:
   nb=z[f'{rep}|{g}|neighbors'];assert nb.shape==(len(qa),100) and (f[nb]!=g).all()
   assert all(len(set(v))==100 for v in nb)
   # Independent equality with definitive saved routing predictions.
   assert np.allclose(r[nb].mean(1),pred[rep][0][qa],atol=1e-12,rtol=0)
   assert np.allclose(c[nb].mean(1),pred[rep][1][qa],atol=1e-12,rtol=0)
   neighbors[rep].append(nb)
 rows=np.asarray(rows);neighbors={rep:np.concatenate(v) for rep,v in neighbors.items()};q=len(rows);rs,cs,fs=r[rows],c[rows],f[rows]
 exclusive={rep:(np.full((q,m),np.nan),np.full((q,m),np.nan)) for rep in pred};shared_count=np.empty(q,int)
 for i,(aa,bb) in enumerate(zip(neighbors['gte'],neighbors['reader'])):
  shared=np.intersect1d(aa,bb);shared_count[i]=len(shared);ea=aa[~np.isin(aa,shared)];eb=bb[~np.isin(bb,shared)];assert len(ea)==len(eb)
  if not len(ea):continue
  for rep,nb in [('gte',ea),('reader',eb)]:exclusive[rep][0][i]=r[nb].mean(0);exclusive[rep][1][i]=c[nb].mean(0)
 informative=shared_count<100;sample_arrays={'shared_count':shared_count};sample_contrasts={};groupmetrics={}
 for rep in pred:
  rr,cc=pred[rep];groupmetrics[rep]=profiles(rs,cs,rr[rows],cc[rows])
  er,ec=exclusive[rep];vv=profiles(rs,cs,er,ec)
  for k,v in vv.items():sample_arrays[f'{rep}|exclusive|{k}']=v
  for lam in lams:
   rv,ch=routing(rs,cs,rr[rows],cc[rows],lam)
   for k,v in rv.items():groupmetrics[rep][f'{lam:g}|{k}']=v
   rv,ch=routing(rs[informative],cs[informative],er[informative],ec[informative],lam)
   for k,v in rv.items():tmp=np.full(q,np.nan);tmp[informative]=v;sample_arrays[f'{rep}|exclusive|{lam:g}|{k}']=tmp
 for key in [k.split('|exclusive|',1)[1] for k in sample_arrays if k.startswith('gte|exclusive|')]:sample_contrasts['exclusive_Qwen_minus_GTE|'+key]=sample_arrays['reader|exclusive|'+key]-sample_arrays['gte|exclusive|'+key]
 # Common random numbers: same per-query/repeat/family draw, first n_f used for each encoder.
 repeats=30;random_metrics={rep:{} for rep in pred};random_r={rep:np.empty((repeats,q,m)) for rep in pred};random_c={rep:np.empty((repeats,q,m)) for rep in pred}
 pools={g:np.flatnonzero(f==g) for g in np.unique(f)}
 for b in range(repeats):
  for i,row in enumerate(rows):
   rr={rep:np.zeros(m) for rep in pred};cc={rep:np.zeros(m) for rep in pred}
   counts={rep:np.bincount(f[neighbors[rep][i]],minlength=len(pop)) for rep in pred}
   for g,pool in pools.items():
    nn=max(counts[rep][g] for rep in pred)
    if not nn:continue
    assert g!=f[row] and nn<=len(pool)
    rng=np.random.default_rng(np.random.SeedSequence([20260923,b,int(row),int(g)]));draw=rng.choice(pool,int(nn),replace=False)
    for rep in pred:
     k=counts[rep][g]
     if k:rr[rep]+=r[draw[:k]].sum(0);cc[rep]+=c[draw[:k]].sum(0)
   for rep in pred:random_r[rep][b,i]=rr[rep]/100;random_c[rep][b,i]=cc[rep]/100
  for rep in pred:
   rr=random_r[rep][b];cc=random_c[rep][b];vals=profiles(rs,cs,rr,cc)
   for lam in lams:
    vv,_=routing(rs,cs,rr,cc,lam)
    for k,v in vv.items():vals[f'{lam:g}|{k}']=v
   for k,v in vals.items():random_metrics[rep].setdefault(k,[]).append(v)
  print('RANDOM_REPEAT',b+1,'seconds',round(time.time()-t),flush=True)
 mcse={}
 sampleweights=np.array([pop[int(g)]/(fs==g).sum() for g in fs])
 for rep in pred:
  sample_arrays[f'{rep}|random_r']=random_r[rep];sample_arrays[f'{rep}|random_c']=random_c[rep]
  for k,v in random_metrics[rep].items():
   v=np.asarray(v);valid=np.isfinite(v);avg=np.divide(np.nansum(v,axis=0),valid.sum(0),out=np.full(q,np.nan),where=valid.sum(0)>0)
   sample_arrays[f'{rep}|random_mean_metric|{k}']=avg
   sample_contrasts[f'{rep}|actual_minus_family_matched_random|{k}']=groupmetrics[rep][k]-avg
   sums=(np.nan_to_num(v)*sampleweights).sum(1);den=(valid*sampleweights).sum(1);pooled=np.divide(sums,den,out=np.full(repeats,np.nan),where=den>0)
   mcse[f'{rep}|{k}']=float(np.nanstd(pooled,ddof=1)/np.sqrt(repeats)) if np.isfinite(pooled).any() else None
 for k in groupmetrics['gte']:
  sample_contrasts['random_family_mixture_Qwen_minus_GTE|'+k]=sample_arrays['reader|random_mean_metric|'+k]-sample_arrays['gte|random_mean_metric|'+k]
  sample_contrasts['selection_gain_Qwen_minus_GTE|'+k]=sample_contrasts['reader|actual_minus_family_matched_random|'+k]-sample_contrasts['gte|actual_minus_family_matched_random|'+k]
 np.savez_compressed(dest/'sample_metrics.npz',query_ids=d['query_ids'][rows],rows=rows,family=fs,**sample_arrays,**sample_contrasts)
 summary=dict(panel=a.panel,n_queries=n,n_sample=q,population=pop,k=100,lambdas=lams,random_repeats=repeats,identical_neighborhood_queries=int((~informative).sum()),
 full_absolute={k:float(np.nanmean(v)) if np.isfinite(v).any() else None for k,v in full_arrays.items() if not k.endswith('|choice')},
 full_contrasts=summarize(contrasts,f,pop),sample_contrasts=summarize(sample_contrasts,fs,pop),randomization_mcse=mcse,
 per_family_full={d['meta']['families'][g]:{k:float(np.nanmean(v[f==g])) if np.isfinite(v[f==g]).any() else None for k,v in contrasts.items()} for g in pop},
 per_family_sample={d['meta']['families'][g]:{k:float(np.nanmean(v[fs==g])) if np.isfinite(v[fs==g]).any() else None for k,v in sample_contrasts.items()} for g in pop},
 interpretation='All contrasts Qwen minus GTE or actual minus random as named. Lower errors/regret better; higher utility/pair accuracy better. Bootstrap conditions on observed families and saved fitted predictors; it does not quantify unseen-family variability or response-generation noise. Pair accuracy excludes observed target utility ties; prediction ties score one half. CIs are descriptive, pointwise, not multiplicity-adjusted.',
 cost_reward_swap='Symmetric two-order decomposition of prediction-based decisions; not causal attribution of encoder training.',
 sample_note='Fixed random sample up to 256 queries per family. Weighted to full-cohort family proportions. Random controls retain exact counts from each source family, 100 distinct source neighbours, 30 repeats. Random decisions evaluated separately before averaging metrics; no unintended ensemble.',
 source_hashes={str(p.relative_to(BASE)):sha(p) for p in source_files},code_hashes={p.name:sha(p) for p in (HERE/'code').glob('*.py')},seconds=time.time()-t)
 (dest/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',a.panel,'seconds',round(time.time()-t),flush=True)
if __name__=='__main__':main()
