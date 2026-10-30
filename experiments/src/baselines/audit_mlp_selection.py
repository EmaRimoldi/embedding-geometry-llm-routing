"""Independently reconstruct MLP selection from saved inner-fold candidate scores."""
import json
import numpy as np
from common import ROOT,split_path,destination,sha
from run_mlp import tasks,configs,done_path,P,CFG,atomic_json
def main():
 checked=[];missing=[]
 for panel,sp,rep in tasks():
  root=destination('mlp',panel,sp,rep);p=root/'result.json'
  if not p.exists():missing.append([panel,sp,rep]);continue
  s=json.loads(split_path(panel).read_text())[sp];r=json.loads(p.read_text());te=np.array(s['test']);tr=np.array(s['train'])
  assert r['split_sha256']==sha(split_path(panel))
  final=np.load(root/'predictions.npz')
  np.testing.assert_array_equal(final['test_rows'],te)
  nv=sum(len(q['val']) for q in s['inner'])
  for lam in CFG['lambdas'][panel]:
   candidates=[];scores=[]
   for ia,h,es in configs():
    total=np.zeros(len(P['alphas']))
    for i,q in enumerate(s['inner']):
     source=done_path(root,ia,es,lam,f'inner_{i}')
     v=json.loads(source.read_text());fi=np.array(q['fit']);va=np.array(q['val'])
     assert not np.intersect1d(fi,va).size
     assert np.isin(fi,tr).all() and np.isin(va,tr).all()
     assert v['split_sha256']==sha(split_path(panel))
     assert v['architecture']==h and v['early_stopping']==es and v['alphas']==P['alphas']
     assert v['n_fit']==len(fi) and v['n_evaluation']==len(va)
     assert np.isfinite(v['utility_sum']).all()
     np.testing.assert_array_equal(np.load(source.with_suffix('.npz'))['rows'],va)
     total+=v['utility_sum']
    for j,alpha in enumerate(P['alphas']):
     candidates.append((ia,h,es,j,alpha));scores.append(float(total[j]/nv))
   best=int(np.argmax(scores));ia,h,es,j,alpha=candidates[best]
   sel=r['selections'][str(lam)]
   assert sel['architecture']==h and sel['early_stopping']==es and sel['alpha']==alpha
   assert abs(sel['validation_utility']-scores[best])<1e-12
   np.testing.assert_allclose([v['validation_utility'] for v in sel['candidates']],scores,rtol=0,atol=1e-12)
   ref=np.load(done_path(root,ia,es,lam,'refit').with_suffix('.npz'))
   np.testing.assert_array_equal(ref['rows'],te)
   np.testing.assert_array_equal(final[f'{lam:g}|mlp'],ref['choice'][:,j])
   np.testing.assert_array_equal(final[f'{lam:g}|predicted_utility'],ref['predicted_utility'][:,j,:])
   assert np.isfinite(final[f'{lam:g}|predicted_utility']).all()
   np.testing.assert_array_equal(final[f'{lam:g}|predicted_utility'].argmax(1),final[f'{lam:g}|mlp'])
   checked.append(dict(panel=panel,split=sp,representation=rep,lambda_value=lam,architecture=h,alpha=alpha,early_stopping=es))
 out=ROOT/'results/baselines/_summary/mlp_selection_audit.json'
 atomic_json(out,dict(pass_completed=True,checked=checked,missing=missing,all_complete=not missing))
 print('MLP_SELECTION_AUDIT',len(checked),'selections;',len(missing),'unfinished folds',flush=True)
if __name__=='__main__':main()
