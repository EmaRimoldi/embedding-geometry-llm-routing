"""Reuse frozen raw extraction; extract missing NVIDIA candidate layers only."""
import argparse,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import encode
from timing import hardware
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
 encode.CONFIG=dict(encode.CONFIG,qwen_layers=json.loads((ROOT/'configs/nvidia.json').read_text())['layers'])
 cached=[l for l in encode.CONFIG['qwen_layers'] if (ROOT/'artifacts/embeddings/qwen2.5-7b-instruct'/f'layer_{l}'/a.dataset/'manifest.json').exists()]
 started=time.perf_counter();status='failed'
 try:
  encode.main('qwen2.5-7b-instruct',a.dataset);status='completed'
 finally:
  record=dict(dataset=a.dataset,layers=encode.CONFIG['qwen_layers'],cached_layers_before=cached,status=status,seconds=time.perf_counter()-started,hardware=hardware(cuda=True),scope='Incremental extraction including model load and I/O; existing layers are reused')
  out=ROOT/'results/nvidia/timing/extraction'/f'{os.environ.get("SLURM_JOB_ID","local")}-{a.dataset}.json';out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps(record,indent=2)+'\n')
