import json
import numpy as np
import torch
import run_graph as rg
from common import ROOT,load

torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
p='routerbench-0shot';d=load(p,'gte');des=np.load(ROOT/'inputs/graph_descriptions'/p/'descriptions.npz');splits=json.loads((ROOT/'inputs'/p/'splits.json').read_text());results=[]
for seed in [3,4]:
 s=splits[f'iid_{seed}'];fit=np.array(s['inner'][0]['fit']);val=np.array(s['inner'][0]['val'])[:5]
 g=rg.graph(d,des,fit,1.,0.);net=rg.init(d,des,seed);tq=rg.tensor(des['task_embedding'][d['fam'][val]]);xq=rg.tensor(d['x'][val])
 for precision in ['float32','float64']:
  if precision=='float64':
   net=net.double();g={k:(v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in g.items()};tq=tq.double();xq=xq.double()
  net.eval();n,m=g['n'],g['m'];q=len(val)
  with torch.no_grad():
   train=g['edges'].clone();train[1]+=q
   test=torch.stack([torch.arange(n,n+q,device=rg.DEVICE).repeat_interleave(m),n+q+torch.arange(m,device=rg.DEVICE).repeat(q)])
   edges=torch.cat([train,test],1);vis=torch.arange(edges.shape[1],device=rg.DEVICE)<train.shape[1]
   weights=torch.cat([g['weight'],torch.full((q*m,3),999.,device=rg.DEVICE,dtype=g['weight'].dtype)],0)
   direct=net(torch.cat([g['task'],tq]),torch.cat([g['query'],xq]),g['llm'],edges,~vis,vis,weights).reshape(q,m).cpu().numpy()
   fast=rg.predict(net,g,tq,xq)
   v=dict(seed=seed,precision=precision,max_abs=float(abs(direct-fast).max()),choices_equal=bool(np.array_equal(direct.argmax(1),fast.argmax(1))),tf32=False);results.append(v);print(v,flush=True)
   if precision=='float64':assert np.allclose(direct,fast,atol=1e-10,rtol=1e-10)
(ROOT/'reports/graph_numerical_diagnosis.json').write_text(json.dumps(results,indent=2))
