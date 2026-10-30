"""Extract missing sweep layers with the original, unchanged extraction pipeline."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import encode
cfg=json.loads((ROOT/'configs/layer_sweep.json').read_text())
# Process-local configuration; existing geometry config and encoder code are unchanged.
encode.CONFIG=dict(encode.CONFIG,qwen_layers=cfg['layers'])
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['routerbench','sprout'])
 a=ap.parse_args()
 encode.main('qwen2.5-7b-instruct',a.dataset)
