"""Known-family and known-native-cell references, IID only. No invented topics."""
import argparse,json
import numpy as np
from common import ROOT,load,split_path,destination,score
from run_mlp import atomic_json,atomic_npz,tasks
def main(task):
 panel,sp,rep=tasks()[task]
 if not sp.startswith('iid_'):return
 d=load(panel);s=json.loads(split_path(panel).read_text())[sp];tr=np.array(s['train']);te=np.array(s['test'])
 root=destination('lookups',panel,sp,rep);root.mkdir(parents=True,exist_ok=True)
 if (root/'result.json').exists():return
 cfg=json.loads((ROOT/'configs/baselines.json').read_text());results={};arrays=dict(test_rows=te,query_ids=d['query_ids'][te]);coverage={}
 for lam in cfg['lambdas'][panel]:
  u=d['r']-lam*d['c'];globalbest=u[tr].mean(0).argmax()
  fp={f:u[tr[d['fam'][tr]==f]].mean(0).argmax() for f in np.unique(d['fam'][tr])}
  family=np.array([fp.get(f,globalbest) for f in d['fam'][te]])
  cell=family.copy();seen=0
  for g in np.unique(d['subgroup'][te]):
   pool=tr[d['subgroup'][tr]==g];mask=d['subgroup'][te]==g
   if len(pool):cell[mask]=u[pool].mean(0).argmax();seen+=int(mask.sum())
  methods={'family_lookup':family,'family_topic_lookup':cell}
  results[str(lam)]={name:score(ch,d['r'][te],d['c'][te],d['fam'][te],lam) for name,ch in methods.items()}
  for name,ch in methods.items():arrays[f'{lam:g}|{name}']=ch
  coverage[str(lam)]=seen/len(te)
 atomic_npz(root/'predictions.npz',**arrays)
 atomic_json(root/'result.json',dict(panel=panel,split=sp,representation=rep,results=results,n_train=len(tr),n_test=len(te),known_cell_coverage=coverage,definition='Native subgroup where present; otherwise family. Missing training cell falls back to family, then global. No Qwen-generated labels. Diagnostic reference; known metadata required.'))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--task',type=int,required=True);main(a.parse_args().task)
