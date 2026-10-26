"""Audit Ridge independently, keeping the already-running baseline source frozen."""
import json,csv
from collections import defaultdict
import numpy as np
from common import ROOT,load,score,destination,split_path
from run_mlp import tasks
def main():
 cfg=json.loads((ROOT/'configs/baselines.json').read_text());records=[];missing=[]
 for panel,sp,rep in tasks():
  root=destination('ridge',panel,sp,rep)
  if not (root/'result.json').exists():missing.append([panel,sp,rep]);continue
  d=load(panel);s=json.loads(split_path(panel).read_text())[sp];te=np.array(s['test'])
  rec=json.loads((root/'result.json').read_text());z=np.load(root/'predictions.npz')
  np.testing.assert_array_equal(z['test_rows'],te);np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
  for lam in cfg['lambdas'][panel]:
   old=rec['results'][str(lam)]['ridge'];ch=z[f'{lam:g}|ridge']
   q=score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)
   for k in ['utility','reward','cost']:assert abs(q[k]-old[k])<1e-10
   select=rec['selections'][str(lam)];i=int(np.argmax(select['validation_utilities']))
   assert select['alpha']==select['candidate_alphas'][i]
   records.append(dict(dataset=panel,split=sp,protocol=sp.split('_')[0],encoder=rep,method='Ridge',lambda_value=lam,n_test=len(te),utility=q['utility'],reward=q['reward'],cost=q['cost']))
 out=ROOT/'results/baselines/_summary';out.mkdir(parents=True,exist_ok=True)
 groups=defaultdict(list)
 for v in records:groups[(v['dataset'],v['protocol'],v['encoder'],v['lambda_value'])].append(v)
 agg=[]
 for key,rows in sorted(groups.items()):
  expected=5 if key[1]=='iid' else (7 if key[0]=='routerbench-0shot' else 6)
  n=sum(v['n_test'] for v in rows)
  agg.append(dict(zip(['dataset','protocol','encoder','lambda_value'],key),method='Ridge',folds_completed=len(rows),folds_expected=expected,complete=len(rows)==expected,**{k:sum(v[k]*v['n_test'] for v in rows)/n for k in ['utility','reward','cost']}))
 for name,rows in [('ridge_per_split.csv',records),('ridge_scores.csv',agg)]:
  if rows:
   with (out/name).open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'ridge_status.json').write_text(json.dumps(dict(audited_score_records=len(records),missing=missing,all_complete=not missing),indent=2))
 print('RIDGE_AUDIT',len(records),'scores;',len(missing),'unfinished folds',flush=True)
if __name__=='__main__':main()
