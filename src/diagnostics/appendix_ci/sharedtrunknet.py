"""Appendix comparison of frozen SharedTrunkNet variants; no refitting."""
import sys,json,os
from pathlib import Path
import numpy as np
from ci_common import ROOT,OUT,CFG,write,track,boot_sums,record
sys.path.insert(0,str(ROOT/'src/baselines'))
import common as base

def run():
 dest=OUT/'sharedtrunknet'
 assert not (dest/'report.json').exists(),'Preserve existing report'
 reports={};provenance={};audited=0
 methods=['fixed_l21','selected_layer','per_model','knn100','family_adaptation']
 for panel in ['routerbench-0shot','sprout-sixfam']:
  dataset=base.DATASETS[panel];d=base.load(panel);n=len(d['r'])
  splitfile=base.split_path(panel);splits=json.loads(splitfile.read_text())
  track(splitfile,provenance);track(ROOT/'datasets'/dataset/'prepared/panel.npz',provenance)
  lams=json.loads((ROOT/'configs/baselines.json').read_text())['lambdas'][panel]
  columns=[f'{m}|{l:g}' for m in methods for l in lams]
  matrix=np.zeros((n,len(columns)));seen=np.zeros(n,int);counts={m:[] for m in methods}
  for sp in sorted(s for s in splits if s.startswith('ood_')):
   te=np.asarray(splits[sp]['test']);tr=np.asarray(splits[sp]['train']);seen[te]+=1
   assert not np.intersect1d(tr,te).size
   p=ROOT/'results/nvidia/results'/dataset/sp
   j=json.loads((p/'result.json').read_text());z=np.load(p/'predictions.npz')
   track(p/'result.json',provenance);track(p/'predictions.npz',provenance)
   np.testing.assert_array_equal(te,z['test_rows'])
   np.testing.assert_array_equal(d['query_ids'][te],z['query_ids'])
   for m in methods[:2]:
    for lam in lams:
     ch=z[f'{m}|{lam:g}|choice'];u=(d['r'][te]-lam*d['c'][te])[np.arange(len(te)),ch]
     assert abs(u.mean()-j['results'][m][str(lam)]['utility'])<1e-10
     matrix[te,columns.index(f'{m}|{lam:g}')]=u/n;audited+=1
   p=ROOT/'results/nvidia_layer_validation/utility_v1'/dataset/sp
   j=json.loads((p/'result.json').read_text());z=np.load(p/'predictions.npz')
   track(p/'result.json',provenance);track(p/'predictions.npz',provenance)
   np.testing.assert_array_equal(te,z['test_rows'])
   np.testing.assert_array_equal(d['query_ids'][te],z['query_ids'])
   for lam in lams:
    ch=z[f'choice_{lam:g}'];u=(d['r'][te]-lam*d['c'][te])[np.arange(len(te)),ch]
    assert abs(u.mean()-j['scores'][str(lam)]['utility'])<1e-10
    matrix[te,columns.index(f'per_model|{lam:g}')]=u/n;audited+=1
   for kind,m in [('knn','knn100'),('prior','family_adaptation')]:
    p=base.destination(kind,panel,sp,'reader')
    j=json.loads((p/'result.json').read_text());z=np.load(p/'predictions.npz')
    track(p/'result.json',provenance);track(p/'predictions.npz',provenance)
    np.testing.assert_array_equal(te,z['test_rows'])
    np.testing.assert_array_equal(d['query_ids'][te],z['query_ids'])
    for lam in lams:
     ch=z[f'{lam:g}|{m}'];u=(d['r'][te]-lam*d['c'][te])[np.arange(len(te)),ch]
     assert abs(u.mean()-j['results'][str(lam)][m]['utility'])<1e-10
     matrix[te,columns.index(f'{m}|{lam:g}')]=u/n;audited+=1
  assert np.all(seen==1)
  weights=np.full(n,1/n);point=matrix.sum(0)/weights.sum()
  boot=boot_sums(matrix,weights,d['fam'])
  existing_path=OUT/'routing'/panel/'ood/report.json';track(existing_path,provenance)
  previous=json.loads(existing_path.read_text())
  rows={}
  for m in methods:
   ix=[columns.index(f'{m}|{lam:g}') for lam in lams]
   rows[m]={f'{lam:g}':record(point[i],boot[:,i]) for i,lam in zip(ix,lams)}
   rows[m]['mean']=record(point[ix].mean(),boot[:,ix].mean(1))
  # Reproduce all existing estimates/intervals before adding older variants.
  for m in methods[2:]:
   scope='nvidia' if m=='per_model' else 'base'
   method='nvidia' if m=='per_model' else m
   representation='utility' if m=='per_model' else 'reader'
   for lam in [f'{v:g}' for v in lams]+['mean']:
    old=next(v for v in previous['rows'] if v['scope']==scope and v['representation']==representation and v['method']==method and v['lambda_value']==lam)
    for k in ['mean','low','high']:assert abs(rows[m][lam][k]-old[k])<1e-10,(panel,m,lam,k)
  differences={};wins={}
  ref=[columns.index(f'per_model|{lam:g}') for lam in lams]
  for m in ['knn100','family_adaptation']:
   ix=[columns.index(f'{m}|{lam:g}') for lam in lams]
   differences[m]=record((point[ix]-point[ref]).mean(),(boot[:,ix]-boot[:,ref]).mean(1))
   old=next(v for v in previous['differences'] if v['comparison']=='L21 minus NVIDIA' and v['method']==m)
   for k in ['mean','low','high']:assert abs(differences[m][k]-old[k])<1e-10
   delta=point[ix]-point[ref]
   wins[m]={'ours_higher':int((delta>0).sum()),'sharedtrunknet_higher':int((delta<0).sum()),'ties':int((delta==0).sum())}
  reports[panel]={'lambdas':lams,'rows':rows,'differences':differences,'penalty_counts':wins}
  dest.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(dest/(panel+'.npz'),columns=np.array(columns),point=point,draws=boot)
  print('COMPLETE',panel,flush=True)
 track(Path(__file__),provenance);track(ROOT/'src/diagnostics/appendix_ci/ci_common.py',provenance)
 track(ROOT/'configs/appendix_ci.json',provenance)
 write(dest/'report.json',{'complete':True,'replicates':CFG['replicates'],'audit_score_count':audited,'reproduced_existing_intervals':True,'results':reports,'source_hashes':provenance,'slurm_job':os.environ.get('SLURM_JOB_ID')})
 print('AUDITED',audited,flush=True)
if __name__=='__main__':run()
