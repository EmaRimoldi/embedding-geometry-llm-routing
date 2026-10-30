"""Frozen inputs and provenance for NVIDIA's utility-pipeline adaptation."""
import functools,hashlib,json,sys,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/baselines'))
import common as base
from run_mlp import atomic_json,atomic_npz
OUT=ROOT/'results/nvidia'
CONFIG=ROOT/'configs/nvidia.json'
@functools.lru_cache(maxsize=1)
def cfg():return json.loads(CONFIG.read_text())
@functools.lru_cache(maxsize=512)
def sha(p):return base.sha(p)
def sig(rows):return hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()[:20]
@functools.lru_cache(maxsize=2)
def data(panel):return base.load(panel)
@functools.lru_cache(maxsize=2)
def splits(panel):return json.loads(base.split_path(panel).read_text())
def task(i):return cfg()['tasks'][i]
def folds(panel,sp):return cfg()['validation_folds'][panel][sp]['inner_indices']
def transform_dir(panel,rows,layer):return OUT/'pca'/base.DATASETS[panel]/sig(rows)/f'layer_{layer}'
def features_dir(panel,layer):return ROOT/'artifacts/embeddings/qwen2.5-7b-instruct'/f'layer_{layer}'/base.DATASETS[panel]
def verify():
 for file in ['results/baselines/_run/source_hashes.json','results/nvidia/provenance/source_hashes.json']:
  for p,h in json.loads((ROOT/file).read_text()).items():assert base.sha(ROOT/p)==h,('Frozen source changed',p)
 assert sha(ROOT/'configs/mlp_reduced.json')==cfg()['reduced_mlp_config_sha256']
def trunk():
 p=ROOT/'src/nvidia/vendor/trunk.py';text=p.read_text();needle='loss_fn = nn.BCEWithLogitsLoss()'
 assert text.count(needle)==1
 adapted=text.replace(needle,'loss_fn = nn.MSELoss()');assert adapted.replace('loss_fn = nn.MSELoss()',needle)==text
 mod=types.ModuleType('nvidia_utility_trunk');exec(compile(adapted,str(p),'exec'),mod.__dict__);return mod

def fit_identity(panel,sp,layer,lam,seed,stage):
 return dict(panel=panel,split=sp,layer=layer,lambda_value=lam,seed=seed,stage=stage,config_sha256=sha(CONFIG),split_sha256=sha(base.split_path(panel)))
def fit_path(panel,sp,layer,lam,seed,stage):
 return OUT/'fits'/base.DATASETS[panel]/sp/stage/f'layer_{layer}'/f'lambda_{lam:g}'/f'seed_{seed}.json'
def rowsets(panel,sp,stage):
 s=splits(panel)[sp]
 if stage=='final':return np.asarray(s['train']),np.asarray(s['test'])
 f=s['inner'][int(stage.split('_')[1])];return np.asarray(f['fit']),np.asarray(f['val'])
