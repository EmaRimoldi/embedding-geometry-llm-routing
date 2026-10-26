"""Write config/plan.json: one sweep group per (panel, representation, router), held-out-family splits.
Scope (PI, 2026-09-24): justify the layer only -> emitted GTE (reference) and Qwen2.5-7B-Instruct last token
at every layer. One group per array task, so everything runs in parallel.
"""
import json
from pathlib import Path
from common_ls import ROOT,model_path

READER='qwen2.5-7b-instruct'

def main():
 n=json.loads((Path(model_path(READER))/'config.json').read_text())['num_hidden_layers']+1;groups=[]
 for m in ('knn','prior'):
  for p in ('routerbench-0shot','sprout-sixfam'):
   groups.append(dict(panel=p,reader='gte',layer=None,pool='emitted',method=m,splits='ood'))
   groups+=[dict(panel=p,reader=READER,layer=L,pool='last',method=m,splits='ood') for L in range(n)]
 (ROOT/'config/plan.json').write_text(json.dumps(dict(groups=groups),indent=0))
 print(len(groups),'groups written to config/plan.json')

if __name__=='__main__':main()
