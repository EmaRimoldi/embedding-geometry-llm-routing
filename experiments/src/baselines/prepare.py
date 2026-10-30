"""Freeze the same final-paper split construction on the clean native datasets."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from common import ROOT,PANELS,REPS,DATASETS,load,split_path,sha

def main():
 records={}
 for panel in PANELS:
  d=load(panel);f=d['fam'];n=len(f);splits={}
  for seed in range(5):
   tr,te=train_test_split(np.arange(n),train_size=.7,stratify=f,random_state=seed)
   fit,val=train_test_split(tr,test_size=.2,stratify=f[tr],random_state=10000+seed)
   splits[f'iid_{seed}']=dict(train=np.sort(tr).tolist(),test=np.sort(te).tolist(),inner=[dict(fit=np.sort(fit).tolist(),val=np.sort(val).tolist())])
  for g in np.unique(f):
   tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g)
   splits[f'ood_{g}']=dict(train=tr.tolist(),test=te.tolist(),held_out_family=d['meta']['families'][g],inner=[dict(fit=tr[f[tr]!=h].tolist(),val=tr[f[tr]==h].tolist()) for h in np.unique(f[tr])])
  for name,s in splits.items():
   tr,te=np.array(s['train']),np.array(s['test'])
   assert not set(tr)&set(te) and len(tr)+len(te)==n
   for q in s['inner']:
    assert not set(q['fit'])&set(q['val']) and set(q['fit'])|set(q['val'])==set(tr)
    if name.startswith('ood'):assert not set(f[q['fit']])&set(f[q['val']])
   if name.startswith('ood'):assert not set(f[tr])&set(f[te])
   # Reuse the exact geometry outer populations. Inner split is the original routing protocol.
   gp=ROOT/'results/geometry/gte-qwen2-7b'/DATASETS[panel]/'primary'/name/'split.npz'
   if gp.exists():
    with np.load(gp) as z:
     for k,expected in [('train_rows',tr),('test_rows',te)]:
      np.testing.assert_array_equal(z[k],expected)
  p=split_path(panel);p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert json.loads(p.read_text())==splits
  else:p.write_text(json.dumps(splits))
  dataset=DATASETS[panel];pr=ROOT/'datasets'/dataset/'prepared'
  reps={}
  for rep,path in REPS.items():
   ep=ROOT/'artifacts/embeddings'/path/dataset;m=json.loads((ep/'manifest.json').read_text())
   assert sha(ep/'vectors.npy')==m['vector_sha256']
   assert sha(pr/'queries.jsonl')==m['query_sha256']
   reps[rep]=m
  records[panel]=dict(rows=n,models=d['meta']['models'],families=d['meta']['families'],family_counts=np.bincount(f).tolist(),panel_sha256=sha(pr/'panel.npz'),split_sha256=sha(p),representations=reps)
 out=ROOT/'results/baselines/_run';out.mkdir(parents=True,exist_ok=True)
 (out/'inputs.json').write_text(json.dumps(records,indent=2))
 sources={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/baselines').glob('*.py'))}
 sources['configs/baselines.json']=sha(ROOT/'configs/baselines.json')
 sources['configs/baseline_tasks.json']=sha(ROOT/'configs/baseline_tasks.json')
 (out/'source_hashes.json').write_text(json.dumps(sources,indent=2))
 print(json.dumps({p:dict(rows=v['rows'],families=v['families']) for p,v in records.items()},indent=2),flush=True)
if __name__=='__main__':main()
