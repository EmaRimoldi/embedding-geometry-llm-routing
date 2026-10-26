"""Pack a split's 13 independent lambda jobs to respect scheduler job-count limits."""
import argparse,json,subprocess,sys
from common import ROOT
ap=argparse.ArgumentParser();ap.add_argument('--pack',type=int,required=True);a=ap.parse_args()
cfg=json.loads((ROOT/'config/protocol.json').read_text());tasks=[];packs=[]
for panel,nf in [('routerbench-0shot',7),('sprout-sixfam',6)]:
 for split in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(nf)]:
  group=[]
  for lam in cfg['lambdas'][panel]:group.append(len(tasks));tasks.append((panel,split,lam))
  packs.append(group)
for task in packs[a.pack]:
 panel,split,lam=tasks[task];result=ROOT/'results/mlp_gpu'/panel/split/'reader'/f'lambda_{lam:g}'/'result.json'
 if result.exists():
  j=json.loads(result.read_text());assert j['panel']==panel and j['split']==split and j['lambda_value']==lam and j['representation']=='reader'
  print('ALREADY COMPLETE',task,flush=True);continue
 subprocess.run([sys.executable,'-u',str(ROOT/'code/run_mlp_gpu_l21.py'),'--task',str(task)],check=True)
