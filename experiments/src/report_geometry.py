"""Collect all completed runs; missing runs remain explicitly missing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,write_json
from geometry import mean_ci

def main():
 root=ROOT/'results/geometry';out=root/'_summary';out.mkdir(exist_ok=True)
 expected=json.loads((ROOT/'configs/geometry_tasks.json').read_text());missing=[];not_applicable=[];runs=[];base=[];probes=[];transfers=[];clusters=[]
 for task in expected:
  p=root/task['encoder']/task['dataset']/task['subset']/task['fold']
  if (p/'not_applicable.json').exists():not_applicable.append(json.loads((p/'not_applicable.json').read_text()));continue
  if not (p/'complete.json').exists():missing.append(task);continue
  d=json.loads((p/'metrics.json').read_text());runs.append((task,p,d))
  for rec in d['records']:
   if rec['block']!='unrestricted':continue
   b={**task,'mode':d['mode'],'heldout_family':d['heldout_family'],'n_test':d['n_test'],'k':rec['k']}
   for key in ['same_family_share','random_same_family_share','largest_family_share','same_format_share','random_same_format_share','mean_cosine']:b[key]=rec[key]
   for target in ['mean_reward','solve_fraction','relative_reward','cost']:
    b[target+'_mse']=rec['outcomes'][target]['neighbor_mean_mse']
    b[target+'_local_reduction']=rec['outcomes'][target]['controls']['native_subgroup']['pair_mse_reduction']
   base.append(b)
  for row in d['linear_probes']:probes.append({**task,'mode':d['mode'],**row})
  for row in d['source_family_transfer']:transfers.append({**task,**row})
  cp=root/task['encoder']/task['dataset']/task['subset']/'additional_clustering_kmeans'/task['fold']/'metrics.json'
  if cp.exists():
   cd=json.loads(cp.read_text())
   for key,row in cd['clustering'].items():clusters.append({**task,'label':key,**row})
 for name,data in [('neighborhoods',base),('linear_probes',probes),('source_family_transfer',transfers),('additional_clustering_kmeans/cluster_recovery',clusters)]:
  if data:
   target=out/(name+'.csv');target.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(data).to_csv(target,index=False)
 comparisons=[]
 for task,path,d in runs:
  if task['encoder']=='gte-qwen2-7b':continue
  other=root/'gte-qwen2-7b'/task['dataset']/task['subset']/task['fold']
  if not (other/'complete.json').exists():continue
  a=np.load(path/'per_query.npz');b=np.load(other/'per_query.npz');np.testing.assert_array_equal(a['original_rows'],b['original_rows']);np.testing.assert_array_equal(a['test_rows'],b['test_rows'])
  for key in ['unrestricted/k100/relative_reward/mean_mse','unrestricted/k100/cost/mean_mse']:
   if key in a and key in b:
    diff=b[key].astype(float)-a[key].astype(float);comparisons.append({**task,'metric':key,'direction':'positive means lower error than GTE',**mean_ci(diff)})
  for key in a.files:
   if key.startswith('unrestricted/utility_ordering/') and key in b:
    diff=a[key].astype(float)-b[key].astype(float);comparisons.append({**task,'metric':key,'direction':'positive means higher pairwise ordering accuracy than GTE',**mean_ci(diff)})
 write_json(out/'paired_comparisons.json',comparisons)
 write_json(out/'status.json',dict(expected=len(expected),complete=len(runs),not_applicable=not_applicable,missing=missing))
 lines=['# Geometry reproduction status','',f'{len(runs)} complete; {len(not_applicable)} structurally inapplicable; {len(missing)} missing, out of {len(expected)} originally listed tasks.','', 'Only completed runs enter the tables. All comparisons use matching rows, split IDs and reference-query eligibility. The native full-context and shared-MiniLM subsets are reported separately.','', '## Files','', '- neighborhoods.csv: unrestricted neighbour composition and predictive errors for every fold and k.','- additional_clustering_kmeans/cluster_recovery.csv: ADDITIONAL train-fitted ordinary KMeans experiment; not the historical descriptive clustering reproduction.\n- Original descriptive family clustering is stored separately in each encoder/dataset/subset/descriptive_clustering_original_protocol folder.','- linear_probes.csv: every fixed regularization strength, including unfavorable results.','- source_family_transfer.csv: geometric proximity versus relative-reward usefulness, by held-out and source family.','- paired_comparisons.json: matched differences from GTE with query-bootstrap intervals conditional on the fitted training set.','- Individual fold folders additionally contain native-group restrictions, same-format restrictions, fine-group controls and per-query measurements.']
 if base:
  df=pd.DataFrame(base);num=['same_family_share','random_same_family_share','largest_family_share','same_format_share','relative_reward_mse','cost_mse']
  pooled=[]
  for keys,g in df.groupby(['encoder','dataset','subset','mode','k']):
   pooled.append(dict(zip(['encoder','dataset','subset','mode','k'],keys))|{v:float(np.average(g[v],weights=g.n_test)) for v in num}|{'folds':len(g)})
  pd.DataFrame(pooled).to_csv(out/'pooled_query_weighted.csv',index=False)
  for task,path,d in runs:
   if d['mode']!='ood':continue
   # One heatmap is produced once enough family rows are available below.
  tf=pd.DataFrame(transfers)
  if len(tf):
   for (rep,ds,sub),g in tf.groupby(['encoder','dataset','subset']):
    dest=root/rep/ds/sub/'figures';dest.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for a,val,title in zip(ax,['retrieval_share_in_unrestricted_100','relative_reward_mse'],['Which families supply the neighbours?','How well do their outcomes transfer?']):
     mat=g.pivot(index='heldout_family',columns='source_family',values=val);im=a.imshow(mat.values,aspect='auto',cmap='Blues' if val.startswith('retrieval') else 'magma_r');a.set_xticks(range(len(mat.columns)),mat.columns,rotation=60,ha='right',fontsize=7);a.set_yticks(range(len(mat.index)),mat.index,fontsize=7);a.set_title(title,fontsize=10);fig.colorbar(im,ax=a,shrink=.75)
    fig.savefig(dest/'heldout_family_structure.pdf',bbox_inches='tight');plt.close(fig)
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 print(f'Collected {len(runs)}/{len(expected)} completed runs',flush=True)
if __name__=='__main__':main()
