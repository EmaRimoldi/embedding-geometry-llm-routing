"""Isolated layer-sweep paths; frozen baseline implementations remain unchanged."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/baselines'))
import common as baseline
CFG=json.loads((ROOT/'configs/layer_sweep.json').read_text())
BASE=json.loads((ROOT/'configs/baselines.json').read_text())
ORDER=CFG['priority_layers']+[x for x in CFG['layers'] if x not in CFG['priority_layers']]
PACKS=[(p,l) for l in ORDER for p in CFG['datasets']]
OUT=ROOT/'results/layer_sweep'
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def fold_dir(panel,layer,split):
 return OUT/'qwen2.5-7b-instruct'/('layer_'+str(layer))/CFG['datasets'][panel]/split
def embedding_dir(panel,layer):
 return ROOT/'artifacts/embeddings/qwen2.5-7b-instruct'/('layer_'+str(layer))/CFG['datasets'][panel]
def result_dir(panel,layer,split,kind):
 if layer==21:return baseline.destination(kind,panel,split,'reader')
 return ROOT/'results/baselines/qwen2.5-7b-instruct'/('layer_'+str(layer))/CFG['datasets'][panel]/split/kind
def splits_for(panel,layer):
 splits=json.loads(baseline.split_path(panel).read_text())
 return [(sp,s) for sp,s in splits.items() if sp.startswith('ood') or layer in CFG['additional_iid_layers']]
def verify_sources():
 for name in ['results/baselines/_run/source_hashes.json','results/layer_sweep/provenance/source_hashes.json']:
  for f,h in json.loads((ROOT/name).read_text()).items():
   assert baseline.sha(ROOT/f)==h,('Frozen source changed',f)
