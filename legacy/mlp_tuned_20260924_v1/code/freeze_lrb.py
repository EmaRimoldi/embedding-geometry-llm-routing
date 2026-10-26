"""Freeze LLMRouterBench splits with the exact recipe of astra_definitive_paper_20260923_v1/code/freeze.py:
five family-stratified 70/30 IID splits (seeds 0-4) with one stratified 80/20 inner split (seed 10000+s), and
leave-one-family-out outer splits with leave-one-source-family-out inner splits. Outcome-independent.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from common_mlp import LRB,ROOT,load,sha

def main():
 dest=ROOT/'inputs'/LRB;dest.mkdir(parents=True,exist_ok=True);out=dest/'splits.json'
 assert not out.exists(),f'{out} already frozen'
 d=load(LRB,None);f=d['fam'];n=len(f);splits={}
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
 out.write_text(json.dumps(splits))
 audit=dict(panel=LRB,rows=n,models=d['r'].shape[1],families=d['meta']['families'],family_counts=np.bincount(f).tolist(),split_sha256=sha(out),
  source_panel_sha256=sha(ROOT.parent/'astra_reward_transfer_20260922_v1/inputs'/LRB/'panel.npz'))
 (dest/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
