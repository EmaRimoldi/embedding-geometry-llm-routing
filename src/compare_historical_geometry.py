"""Compare full GTE/L21 results with provenance-recorded historical aggregates."""
import json
import numpy as np
from common import ROOT,write_json

def main():
 old=json.loads((ROOT/'reproducibility/manifests/historical_geometry_reference.json').read_text());root=ROOT/'results/geometry';records=[];concentration=[]
 for ds,ref in old.items():
  for rep,alias in [('gte-qwen2-7b','gte'),('qwen2.5-7b-instruct/layer_21','reader')]:
   paths=sorted((root/rep/ds/'primary').glob('ood_*/metrics.json'));ms=[json.loads(p.read_text()) for p in paths];required=7 if ds=='routerbench' else 6
   assert len(ms)==required,'Do not compare incomplete family totals'
   rr=[next(v for v in m['records'] if v['block']=='unrestricted' and v['k']==100) for m in ms];w=np.array([m['n_test'] for m in ms])
   for lam in ms[0]['records'][0]['utility_ordering']:
    value=lam['lambda_value'];key=f'{value:g}|{alias}|pair_accuracy';previous=ref['metrics'][key];current=np.average([next(v for v in r['utility_ordering'] if v['lambda_value']==value)['pair_accuracy'] for r in rr],weights=w)
    records.append(dict(dataset=ds,encoder=rep,metric='pairwise_utility_ordering_accuracy',lambda_value=value,old=previous,new=float(current),difference=float(current-previous)))
   for name,key in [('relative_reward','centered_reward_mse'),('cost','cost_mse_usd2')]:
    previous=ref['metrics'][alias+'|'+key];current=np.average([v['outcomes'][name]['neighbor_mean_mse'] for v in rr],weights=w);records.append(dict(dataset=ds,encoder=rep,metric=key,old=previous,new=float(current),difference=float(current-previous)))
   if alias=='gte':
    concentration.append(dict(dataset=ds,old=ref['previous_equal_family_concentration_k20'],new=float(np.mean([next(v for v in m['records'] if v['block']=='unrestricted' and v['k']==20)['largest_family_share'] for m in ms]))))
 notes=['Old and new RouterBench row sets differ (34,661 vs 34,771); SPROUT has 43,887 rows in both.', 'Historical family-concentration analysis sampled queries within families; the new version evaluates all eligible queries. Concentration comparison here weights families equally to match the old presentation, although new headline summaries weight queries equally.', 'Do not compare old absolute-distance difficulty percentages with new squared-distance percentages. An absolute-distance comparison is calculated separately.', 'The original balanced descriptive ordinary-k-means protocol has been reproduced separately. Additional clustering uses training-fitted ordinary k-means and held-out evaluation; do not mix its NMI values with the descriptive scores. MiniBatch results are withdrawn.', 'No historical generated labels are imported; only aggregate reference numbers and their provenance are stored.']
 write_json(root/'_summary/historical_comparison.json',dict(records=records,family_concentration=concentration,notes=notes))
 lines=['# Comparison with historical GTE / Qwen L21 geometry','', '| Benchmark | Encoder | Old ordering accuracy at lambda 50 | New |','|---|---|---:|---:|']
 for v in records:
  if v['metric']=='pairwise_utility_ordering_accuracy' and v['lambda_value']==50:lines.append(f"| {v['dataset']} | {v['encoder']} | {100*v['old']:.2f}% | {100*v['new']:.2f}% |")
 lines+=['','## Comparability','']+['- '+n for n in notes]
 (root/'_summary/historical_comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:8]))
if __name__=='__main__':main()
