"""Verify source-only validation and nested layer choices against frozen records."""
from pathlib import Path
import sys,json,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'layer_sweep'))
from common_layer import ROOT,CFG,BASE,OUT,baseline,verify_sources,fold_dir,result_dir,splits_for
verify_sources()
count=0;inner_count=0;selected=[]
per=list(csv.DictReader((OUT/'summary/per_split.csv').open()))
for panel in CFG['datasets']:
 d=baseline.load(panel);all_splits=json.loads(baseline.split_path(panel).read_text())
 for sp,s in splits_for(panel,21):
  tr=np.array(s['train']);te=np.array(s['test'])
  assert not set(tr)&set(te)
  assert set(d['fam'][tr]).isdisjoint(d['fam'][te])
  for inner in s['inner']:
   fit=set(inner['fit']);val=set(inner['val'])
   assert fit.isdisjoint(val) and (fit|val)<=set(tr)
   assert (fit|val).isdisjoint(te)
   assert set(d['fam'][list(fit)]).isdisjoint(d['fam'][list(val)])
   inner_count+=1
  entries={}
  for layer in CFG['layers']:
   j=json.loads((fold_dir(panel,layer,sp)/'result.json').read_text())
   kr=json.loads((result_dir(panel,layer,sp,'knn')/'result.json').read_text())
   assert kr['code_sha256']==baseline.sha(ROOT/'src/baselines/run_fixed.py')
   ks=BASE['knn_candidates'];lams=BASE['lambdas'][panel]
   v=np.array([kr['selections'][str(l)]['candidate_utilities'] for l in lams])
   assert np.isfinite(v).all()
   np.testing.assert_allclose(j['validation_utilities']['knn100'],v[:,ks.index(100)],atol=1e-12)
   np.testing.assert_allclose(j['validation_utilities']['knn_validated'],v.max(1),atol=1e-12)
   expected_n=sum(len(i['val']) for i in s['inner'])
   assert j['validation_queries']==expected_n
   for l in lams:assert kr['selections'][str(l)]['validation_queries']==expected_n
   prior=np.array([r['utility_sums'] for r in j['prior_inner']])
   assert [r['n'] for r in j['prior_inner']]==[len(i['val']) for i in s['inner']]
   np.testing.assert_allclose(j['validation_utilities']['family_adaptation'],prior.sum(0)/expected_n,atol=1e-12)
   entries[layer]=j
  for method in CFG['routers']:
   values=np.array([entries[l]['validation_utilities'][method] for l in CFG['layers']])
   common=CFG['layers'][values.mean(1).argmax()]
   each=np.array(CFG['layers'])[values.argmax(0)]
   for mode in ['nested','nested_lambda']:
    rows=[r for r in per if r['dataset']==panel and r['split']==sp and r['method']==method and r['layer']==mode]
    assert len(rows)==len(lams)
    for r in rows:
     index=lams.index(float(r['lambda_value']));expect=common if mode=='nested' else int(each[index])
     assert int(r['selected_layer'])==expect
     count+=1
   selected.append(dict(dataset=panel,split=sp,method=method,layer=int(common)))
assert count==1014
path=OUT/'summary/source_selection_audit.json'
path.write_text(json.dumps(dict(status='passed',nested_records_checked=count,source_only_inner_splits_checked=inner_count,source_hashes_verified=True,heldout_family_excluded=True,validation_aggregation_reconstructed=True,selections=selected),indent=2)+'\n')
print('SOURCE_SELECTION_AUDIT_PASSED',count,inner_count)
