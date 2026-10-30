"""Reconstruct layer selection, ensembles and every reported utility from saved outputs."""
import csv,json
from collections import defaultdict
import numpy as np
from common_nv import *
from run import select_layer,completed,fit_path

def main():
 verify();c=cfg();rows=[];missing=[];prediction_checks=0;validation_checks=0
 for panel,sp in c['tasks']:
  p=OUT/'results'/base.DATASETS[panel]/sp
  if not (p/'result.json').exists():missing.append([panel,sp]);continue
  d=data(panel);rec=json.loads((p/'result.json').read_text());assert rec['config_sha256']==sha(CONFIG)
  # Audit every candidate source-validation routing score before checking selected layer.
  for layer in c['layers']:
   for i in folds(panel,sp):
    stage=f'inner_{i}';tr,va=rowsets(panel,sp,stage);assert not np.intersect1d(tr,va).size
    transform=transform_dir(panel,tr,layer)
    with np.load(transform/'transform.npz') as z:np.testing.assert_array_equal(z['train_rows'],tr)
    for lam in c['lambdas'][panel]:
     fit=completed(panel,sp,layer,lam,c['screen_seed'],stage)
     with np.load(fit_path(panel,sp,layer,lam,c['screen_seed'],stage).with_suffix('.npz')) as z:
      np.testing.assert_array_equal(z['rows'],va);pred=z['prediction'];assert np.isfinite(pred).all();ch=pred.argmax(1)
      total=float((d['r'][va]-lam*d['c'][va])[np.arange(len(va)),ch].sum());assert abs(total-fit['utility_sum'])<1e-8
     validation_checks+=1
  assert select_layer(panel,sp)==rec['selection']
  te=np.asarray(splits(panel)[sp]['test']);tr=np.asarray(splits(panel)[sp]['train'])
  with np.load(p/'predictions.npz') as z:
   np.testing.assert_array_equal(z['test_rows'],te);np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
   for name,layer in [('selected_layer',rec['selection']['layer']),('fixed_l21',21)]:
    with np.load(transform_dir(panel,tr,layer)/'transform.npz') as t:np.testing.assert_array_equal(t['train_rows'],tr)
    for lam in c['lambdas'][panel]:
     records=[completed(panel,sp,layer,lam,s,'final') for s in c['ensemble_seeds']]
     keep=sorted(c['ensemble_seeds'],key=lambda s:(records[c['ensemble_seeds'].index(s)]['validation_mse'],s))[:c['keep_seeds']]
     assert keep==rec['ensembles'][name][str(lam)]['retained_seeds']
     predictions=[]
     for seed in c['ensemble_seeds']:
      with np.load(fit_path(panel,sp,layer,lam,seed,'final').with_suffix('.npz')) as f:
       np.testing.assert_array_equal(f['rows'],te);assert np.isfinite(f['prediction']).all()
       if seed in keep:predictions.append((seed,f['prediction']))
     predictions=dict(predictions);expected=np.mean(np.asarray([predictions[s] for s in keep]),axis=0)
     np.testing.assert_array_equal(z[f'{name}|{lam:g}|prediction'],expected)
     ch=z[f'{name}|{lam:g}|choice'];np.testing.assert_array_equal(ch,expected.argmax(1))
     score=base.score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)
     for key in ['utility','reward','cost','equal_family']:assert abs(score[key]-rec['results'][name][str(lam)][key])<1e-10
     assert score['per_family']==rec['results'][name][str(lam)]['per_family']
     rows.append(dict(dataset=panel,split=sp,protocol=sp.split('_')[0],method=name,layer=layer,lambda_value=lam,n_test=len(te),utility=score['utility'],reward=score['reward'],cost=score['cost']));prediction_checks+=1
 out=OUT/'summary';out.mkdir(exist_ok=True);groups=defaultdict(list)
 for v in rows:groups[(v['dataset'],v['protocol'],v['method'],v['lambda_value'])].append(v)
 agg=[]
 for key,vs in sorted(groups.items()):
  n=sum(v['n_test'] for v in vs);expected=5 if key[1]=='iid' else (7 if key[0]=='routerbench-0shot' else 6)
  agg.append(dict(zip(['dataset','protocol','method','lambda_value'],key),folds_completed=len(vs),folds_expected=expected,complete=len(vs)==expected,**{k:sum(v[k]*v['n_test'] for v in vs)/n for k in ['utility','reward','cost']}))
 for name,vs in [('per_split.csv',rows),('scores.csv',agg)]:
  if vs:
   with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(vs[0]));w.writeheader();w.writerows(vs)
 status=dict(completed_outer_evaluations=23-len(missing),expected=23,missing=missing,source_validation_scores_reconstructed=validation_checks,test_scores_reconstructed=prediction_checks,all_complete=not missing,comparison='Complete NVIDIA utility pipeline; not an isolated architecture or equal-compute comparison')
 atomic_json(out/'audit.json',status);print(status,flush=True)
if __name__=='__main__':main()
