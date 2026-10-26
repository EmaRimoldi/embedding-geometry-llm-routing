"""Frozen panels, splits and scoring for the tuned MLP baseline.

RouterBench and SPROUT: panels, representations and splits of astra_definitive_paper_20260923_v1 (read in place).
LLMRouterBench: the frozen 12,109-query common-context cohort of astra_reward_transfer_20260922_v1 (same
construction as the RouterBench cohort); its splits are frozen by code/freeze_lrb.py with the definitive recipe.
Writes only inside this folder.
"""
from pathlib import Path
import json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E4=Path(os.environ.get('E4_ROOT',ROOT.parent))
DEF=E4/'astra_definitive_paper_20260923_v1'
sys.path.insert(0,str(DEF/'code'))
import common as definitive  # noqa: E402
from common import score,sha,unit  # noqa: E402,F401
PROTOCOL=json.loads((ROOT/'config/protocol.json').read_text())
LRB='llmrouterbench-performance-cost'
PANELS=('routerbench-0shot','sprout-sixfam',LRB)
NFAM={'routerbench-0shot':7,'sprout-sixfam':6,LRB:10}

def load(panel,rep):
 if panel in definitive.PANELS:
  d=definitive.load(panel,rep)
 else:
  src=E4/'astra_reward_transfer_20260922_v1/inputs'/panel;z=np.load(src/'panel.npz',allow_pickle=False)
  d={k:z[k] for k in z.files};d['meta']=json.loads((src/'manifest.json').read_text());d['panel']=panel
  d['r']=np.asarray(d['r'],float);d['c']=np.asarray(d['c'],float)
  assert d['r'].shape==d['c'].shape and np.isfinite(d['r']).all() and np.isfinite(d['c']).all() and (d['c']>=0).all()
  if rep:d['x']=unit(np.load(src/f'{rep}.npy',mmap_mode='r'));assert len(d['x'])==len(d['r'])
 d['missing']=np.asarray(d.get('missing',np.zeros(d['r'].shape,bool)),bool)
 return d

def splits_path(panel):
 return (DEF/'inputs'/panel/'splits.json') if panel in definitive.PANELS else (ROOT/'inputs'/panel/'splits.json')

def splits(panel):
 return json.loads(splits_path(panel).read_text())

def split_names(panel):
 return [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(NFAM[panel])]

def evaluate(ch,r,c,fam,missing,lam):
 out=score(ch,r,c,fam,lam);out['missing_choice_rate']=float(missing[np.arange(len(ch)),ch].mean());return out
