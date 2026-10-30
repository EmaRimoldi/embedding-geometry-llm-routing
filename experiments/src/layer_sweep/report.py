"""Audit complete layer outputs and evaluate source-selected layers, preserving partial status."""
import json,csv
from collections import defaultdict
import numpy as np
from common_layer import ROOT,CFG,BASE,OUT,baseline,atomic,fold_dir,result_dir,splits_for,verify_sources

def main():
 verify_sources();records=[];entries={};missing=[];audit=0
 for panel in CFG['datasets']:
  d=baseline.load(panel);lams=BASE['lambdas'][panel];splits=dict(splits_for(panel,21))
  for layer in CFG['layers']:
   for sp,s in splits_for(panel,layer):
    path=fold_dir(panel,layer,sp)/'result.json'
    if not path.exists():missing.append([panel,layer,sp]);continue
    j=json.loads(path.read_text());assert j['split_sha256']==baseline.sha(baseline.split_path(panel))
    te=np.asarray(s['test']);entries[panel,layer,sp]=j
    for kind in ['knn','prior']:
     p=result_dir(panel,layer,sp,kind);r=json.loads((p/'result.json').read_text())
     ref=j['baseline_records'][kind]
     assert baseline.sha(p/'result.json')==ref['result_sha256']
     assert baseline.sha(p/'predictions.npz')==ref['predictions_sha256']
     z=np.load(p/'predictions.npz')
     np.testing.assert_array_equal(z['test_rows'],te);np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
     for lam in lams:
      methods=['knn100','knn_validated'] if kind=='knn' else ['family_adaptation']
      for method in methods:
       ch=z[f'{lam:g}|{method}'];v=baseline.score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)
       assert all(abs(v[k]-r['results'][str(lam)][method][k])<1e-10 for k in ['utility','reward','cost'])
       records.append(dict(dataset=panel,protocol=sp.split('_')[0],split=sp,layer=layer,
                           method=method,lambda_value=lam,n=len(te),**{k:v[k] for k in ['utility','reward','cost']}));audit+=1
  for sp,s in splits.items():
   if not all((panel,layer,sp) in entries for layer in CFG['layers']):continue
   for method in CFG['routers']:
    vals=np.asarray([entries[panel,layer,sp]['validation_utilities'][method] for layer in CFG['layers']])
    assert vals.shape==(len(CFG['layers']),len(lams)) and np.isfinite(vals).all()
    chosen=CFG['layers'][int(vals.mean(axis=1).argmax())]
    choices=[CFG['layers'][i] for i in vals.argmax(axis=0)]
    for mode in ['nested','nested_lambda']:
     for il,lam in enumerate(lams):
      layer=chosen if mode=='nested' else choices[il]
      match=[r for r in records if r['dataset']==panel and r['split']==sp and r['layer']==layer and r['method']==method and r['lambda_value']==lam]
      assert len(match)==1
      records.append(dict(match[0],layer=mode,selected_layer=layer))
 out=OUT/'summary';out.mkdir(parents=True,exist_ok=True)
 def write_csv(name,rows):
  if not rows:return
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (out/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 write_csv('per_split.csv',records)
 groups=defaultdict(list)
 for r in records:groups[r['dataset'],r['protocol'],str(r['layer']),r['method'],r['lambda_value']].append(r)
 agg=[]
 for (p,prot,layer,m,lam),rs in sorted(groups.items()):
  expected=5 if prot=='iid' else (7 if p=='routerbench-0shot' else 6);n=sum(r['n'] for r in rs)
  agg.append(dict(dataset=p,protocol=prot,layer=layer,method=m,lambda_value=lam,
          folds_completed=len(rs),folds_expected=expected,complete=len(rs)==expected,
          **{k:sum(r[k]*r['n'] for r in rs)/n for k in ['utility','reward','cost']}))
 write_csv('scores.csv',agg)
 atomic(out/'status.json',dict(audited_scores=audit,missing=missing,complete=not missing,
        nested_selection_complete=all((p,l,sp) in entries for p in CFG['datasets'] for l in CFG['layers'] for sp,s in splits_for(p,21))))
 print('AUDIT',audit,'MISSING',len(missing),flush=True)
if __name__=='__main__':main()
