"""Audit saved decisions and assemble query-weighted baseline scores."""
import json,csv
from collections import defaultdict
import numpy as np
from common import ROOT,load,score,split_path,destination,sha
from run_mlp import tasks
def main():
 cfg=json.loads((ROOT/'configs/baselines.json').read_text());records=[];missing=[];checks=0
 for task,(panel,sp,rep) in enumerate(tasks()):
  d=load(panel);s=json.loads(split_path(panel).read_text())[sp];te=np.array(s['test'])
  kinds=['knn','prior','clusters','prox','mlp']+(['lookups'] if sp.startswith('iid') else [])
  for kind in kinds:
   root=destination(kind,panel,sp,rep)
   if not (root/'result.json').exists():missing.append(dict(panel=panel,split=sp,representation=rep,kind=kind));continue
   rec=json.loads((root/'result.json').read_text());z=np.load(root/'predictions.npz')
   np.testing.assert_array_equal(z['test_rows'],te);np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
   for lam in cfg['lambdas'][panel]:
    for method,old in rec['results'][str(lam)].items():
     choice=z[f'{lam:g}|{method}'];assert ((choice>=0)&(choice<d['r'].shape[1])).all()
     new=score(choice,d['r'][te],d['c'][te],d['fam'][te],lam)
     for k in ['utility','reward','cost']:assert abs(new[k]-old[k])<1e-10,(panel,sp,rep,kind,lam,k)
     checks+=1;records.append(dict(dataset=panel,split=sp,protocol=sp.split('_')[0],encoder=rep,method=cfg['display_names'].get(method,method),lambda_value=lam,n_test=len(te),utility=new['utility'],reward=new['reward'],cost=new['cost']))
 out=ROOT/'results/baselines/_summary';out.mkdir(parents=True,exist_ok=True)
 groups=defaultdict(list)
 for v in records:groups[(v['dataset'],v['protocol'],v['encoder'],v['method'],v['lambda_value'])].append(v)
 agg=[]
 for key,rows in sorted(groups.items()):
  expected=5 if key[1]=='iid' else (7 if key[0]=='routerbench-0shot' else 6)
  n=sum(v['n_test'] for v in rows)
  agg.append(dict(zip(['dataset','protocol','encoder','method','lambda_value'],key),folds_completed=len(rows),folds_expected=expected,complete=len(rows)==expected,**{k:sum(v[k]*v['n_test'] for v in rows)/n for k in ['utility','reward','cost']}))
 for name,rows in [('per_split.csv',records),('scores.csv',agg)]:
  if rows:
   with (out/name).open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 status=dict(completed_result_files=sum(1 for _ in groups),audited_score_records=checks,missing=missing,all_complete=not missing,warning='Partial groups are explicitly marked; do not compare their pooled scores as complete results.')
 (out/'status.json').write_text(json.dumps(status,indent=2))
 print('AUDIT',checks,'scores;',len(missing),'unfinished method/fold combinations',flush=True)
if __name__=='__main__':main()
