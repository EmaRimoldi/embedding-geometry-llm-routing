"""Freeze a complete, outcome-independent geometry task list and submit it."""
import json,subprocess
from common import ROOT,write_json
reps=['gte-qwen2-7b']+['qwen2.5-7b-instruct/layer_'+str(l) for l in [21,23,25]]+['minilm-l12']
tasks=[]
for rep in reps:
 for dataset,nfamilies in [('routerbench',7),('sprout',6)]:
  for subset in (['shared_minilm'] if rep=='minilm-l12' else ['primary','shared_minilm']):
   for fold in ['iid_'+str(s) for s in range(5)]+['ood_'+str(s) for s in range(nfamilies)]:tasks.append(dict(encoder=rep,dataset=dataset,subset=subset,fold=fold))
write_json(ROOT/'configs/geometry_tasks.json',tasks)
print('Tasks:',len(tasks))
