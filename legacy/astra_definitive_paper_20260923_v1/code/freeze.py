"""Validate source snapshot hashes, preserve input manifests, freeze all split indices."""
import json,shutil,platform
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from common import ROOT,E4,PANELS,load,source,sha

def main():
 dest=ROOT/'inputs';dest.mkdir(exist_ok=False);record={}
 for panel in PANELS:
  d=load(panel);src=source(panel);meta=d['meta'];z={k:v for k,v in d.items() if isinstance(v,np.ndarray)}
  # Source manifests specify authoritative frozen hashes.
  expected=meta['snapshot_sha256'];got=sha(src/'panel.npz')
  if isinstance(expected,str):assert expected==got
  else:
   assert expected['panel.npz']==got
   for rep in ('gte','reader'):assert sha(src/f'{rep}.npy')==expected[f'{rep}.npy']
  if panel=='sprout-sixfam':
   for rep in ('gte','reader'):assert sha(Path(meta['representations'][rep]['path']))==meta['representations'][rep]['sha256']
  p=dest/panel;p.mkdir();np.savez_compressed(p/'panel.npz',**z);(p/'source_manifest.json').write_text(json.dumps(meta,indent=2))
  f=d['fam'];n=len(f);splits={}
  for seed in range(5):
   tr,te=train_test_split(np.arange(n),train_size=.7,stratify=f,random_state=seed)
   fit,val=train_test_split(tr,test_size=.2,stratify=f[tr],random_state=10000+seed)
   splits[f'iid_{seed}']=dict(train=np.sort(tr).tolist(),test=np.sort(te).tolist(),inner=[dict(fit=np.sort(fit).tolist(),val=np.sort(val).tolist())])
  for g in np.unique(f):
   tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g)
   splits[f'ood_{g}']=dict(train=tr.tolist(),test=te.tolist(),inner=[dict(fit=tr[f[tr]!=h].tolist(),val=tr[f[tr]==h].tolist()) for h in np.unique(f[tr])])
  for name,s in splits.items():
   assert not set(s['train'])&set(s['test']) and len(s['train'])+len(s['test'])==n
   for q in s['inner']:assert not set(q['fit'])&set(q['val']) and set(q['fit'])|set(q['val'])==set(s['train'])
  (p/'splits.json').write_text(json.dumps(splits))
  record[panel]=dict(rows=n,models=d['r'].shape[1],families=meta['families'],family_counts=np.bincount(f).tolist(),source_snapshot=got,split_sha256=sha(p/'splits.json'),native_topic_rows=int((d['topic']!='').sum()))
 (dest/'audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
