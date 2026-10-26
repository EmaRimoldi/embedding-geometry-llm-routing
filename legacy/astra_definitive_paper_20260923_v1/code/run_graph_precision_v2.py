"""Native GraphRouter architecture with strictly source-only graph training.

Public task descriptions are an explicit extra input. Outcome edges exist only
for fit queries. Validation selects checkpoints using the actual USD objective.
At test time the graph has no target outcome edges and frozen BatchNorm stats.
"""
import argparse,json,time,copy
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,load,score,sha
from graph_arch import EncoderDecoderNet

SCENARIOS=[('performance_first',1.,0.),('balance',.5,.5),('cost_first',.2,.8)]
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
INFERENCE_AUDITS=[]
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)

def graph(d,des,rows,ws,wc):
 x=d['x'][rows];r=d['r'][rows];c=d['c'][rows];f=d['fam'][rows];n,m=r.shape;cn=np.empty_like(c)
 for g in np.unique(f):
  ix=f==g;lo,hi=c[ix].min(),c[ix].max();cn[ix]=(c[ix]-lo)/(hi-lo) if hi>lo else 0
 u=ws*r-wc*cn
 edges=torch.as_tensor(np.stack([np.repeat(np.arange(n),m),n+np.tile(np.arange(m),n)]),dtype=torch.long,device=DEVICE)
 return dict(task=tensor(des['task_embedding'][f]),query=tensor(x),llm=tensor(des['llm_embedding']),
  edges=edges,weight=tensor(np.stack([u.ravel(),cn.ravel(),r.ravel()],1)),
  label=tensor(np.isclose(u,u.max(1,keepdims=True),atol=1e-8).astype(np.float32).ravel()),n=n,m=m)

def predict(net,g,tq,xq):
 """Algebraically identical to appending isolated test queries to native eval."""
 net.eval()
 with torch.no_grad():
  ini=net.model_align(g['task'],g['query'],g['llm']);ew=F.relu(net.edge_mlp(g['weight']))
  h=F.relu(net.bn1(net.encoder_conv_1(ini,g['edges'],edge_attr=ew)))
  h=net.bn2(net.encoder_conv_2(h,g['edges'],edge_attr=ew));llm=h[g['n']:]
  qi=torch.cat([net.model_align.task_transform(tq),net.model_align.query_transform(xq)],1)
  return torch.sigmoid(qi@llm.T/qi.shape[1]).cpu().numpy()

def check_inference(net,g,tq,xq):
 net.eval();q=min(5,len(xq));tq,xq=tq[:q],xq[:q];n,m=g['n'],g['m']
 with torch.no_grad():
  train_edges=g['edges'].clone();train_edges[1]+=q
  test_edges=torch.stack([torch.arange(n,n+q,device=DEVICE).repeat_interleave(m),n+q+torch.arange(m,device=DEVICE).repeat(q)])
  edges=torch.cat([train_edges,test_edges],1);visible=torch.arange(edges.shape[1],device=DEVICE)<train_edges.shape[1]
  # Deliberately extreme dummy test attributes must be inaccessible.
  weights=torch.cat([g['weight'],torch.full((q*m,3),999.,device=DEVICE)],0)
  direct=net(torch.cat([g['task'],tq]),torch.cat([g['query'],xq]),g['llm'],edges,~visible,visible,weights).reshape(q,m).cpu().numpy()
  fast=predict(net,g,tq,xq)
  error=float(np.max(abs(direct-fast)))
  alone=predict(net,g,tq[:1],xq[:1]);batch_error=float(np.max(abs(alone-fast[:1])))
  audit=dict(float32_max_abs=error,float32_batch_max_abs=batch_error,
             float32_choices_equal=bool(np.array_equal(direct.argmax(1),fast.argmax(1))))
  if not (np.allclose(direct,fast,rtol=2e-5,atol=2e-6) and np.allclose(alone,fast[:1],rtol=2e-5,atol=2e-6)):
   # Long GPU scatter sums change reduction order when isolated nodes are appended.
   # Establish algebraic identity independently in double precision, without
   # weakening the failed float32 threshold or changing native graph architecture.
   net64=copy.deepcopy(net).double()
   g64={k:(v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in g.items()}
   direct64=net64(torch.cat([g64['task'],tq.double()]),torch.cat([g64['query'],xq.double()]),g64['llm'],edges,~visible,visible,weights.double()).reshape(q,m).cpu().numpy()
   fast64=predict(net64,g64,tq.double(),xq.double())
   alone64=predict(net64,g64,tq[:1].double(),xq[:1].double())
   assert np.allclose(direct64,fast64,rtol=1e-10,atol=1e-10),np.max(abs(direct64-fast64))
   assert np.allclose(alone64,fast64[:1],rtol=1e-10,atol=1e-10)
   audit['float64_max_abs']=float(np.max(abs(direct64-fast64)))
   audit['float64_batch_max_abs']=float(np.max(abs(alone64-fast64[:1])))
  INFERENCE_AUDITS.append(audit)

def init(d,des,seed):
 torch.manual_seed(seed);return EncoderDecoderNet(d['x'].shape[1],des['llm_embedding'].shape[1],8,3).to(DEVICE)

def epoch(net,g,opt):
 net.train();opt.zero_grad();loss_sum=0
 for _ in range(32):
  hide=torch.rand(len(g['label']),device=DEVICE)<.5
  pred=net(g['task'],g['query'],g['llm'],g['edges'],hide,~hide,g['weight']);loss=F.binary_cross_entropy(pred,g['label'][hide]);assert torch.isfinite(loss)
  (loss/32).backward();loss_sum+=float(loss.detach())
 opt.step();return loss_sum/32

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--epochs',type=int,default=2000);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 # IID only until the user settles GraphRouter's additional OOD input policy.
 tasks=[(panel,f'iid_{seed}',sc) for panel in ['routerbench-0shot','sprout-sixfam'] for seed in range(5) for sc in range(3)]
 panel,split,sc=tasks[a.task];name,ws,wc=SCENARIOS[sc];d=load(panel,'gte');des=np.load(ROOT/'inputs/graph_descriptions'/panel/'descriptions.npz')
 assert list(des['models'])==list(d['meta']['models']) and list(des['tasks'])==list(d['meta']['families'])
 s=json.loads((ROOT/'inputs'/panel/'splits.json').read_text())[split];assert len(s['inner'])==1
 fit=np.array(s['inner'][0]['fit']);val=np.array(s['inner'][0]['val']);tr=np.array(s['train']);te=np.array(s['test'])
 lams=json.loads((ROOT/'config/protocol.json').read_text())['lambdas'][panel]
 dest=ROOT/'results'/('graph_precision_v2_smoke' if a.smoke else 'graph_precision_v2')/panel/split/name;dest.mkdir(parents=True,exist_ok=False)
 g=graph(d,des,fit,ws,wc);net=init(d,des,int(split.split('_')[1]));opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
 tq=tensor(des['task_embedding'][d['fam'][val]]);xq=tensor(d['x'][val]);check_inference(net,g,tq,xq)
 best=np.full(len(lams),-np.inf);bestep=np.zeros(len(lams),int);hist=[];bad=0;t=time.time()
 for ep in range(1,a.epochs+1):
  loss=epoch(net,g,opt);p=predict(net,g,tq,xq);ch=p.argmax(1)
  vu=np.array([(d['r'][val,ch]-lam*d['c'][val,ch]).mean() for lam in lams]);improve=vu>best
  best[improve]=vu[improve];bestep[improve]=ep;bad=0 if improve.any() else bad+1;hist.append(dict(epoch=ep,loss=loss,utilities=vu.tolist()))
  if ep%10==0:print('VALIDATE',ep,'seconds',round(time.time()-t),'loss',loss,flush=True)
  if bad>=200:break
 check_inference(net,g,tq,xq)
 record=dict(panel=panel,split=split,scenario=name,validation_utilities=best.tolist(),selected_epochs=bestep.tolist(),history=hist,seconds=time.time()-t,source_only_batchnorm=True,native_inference_equivalence=True,inference_audits=INFERENCE_AUDITS,tf32=False,code_sha256=sha(__file__),architecture_sha256=sha(ROOT/'code/graph_arch.py'),split_sha256=sha(ROOT/'inputs'/panel/'splits.json'))
 if a.smoke:
  (dest/'result.json').write_text(json.dumps(record,indent=2));print('SMOKE_DONE',round(time.time()-t),flush=True);return
 del g,tq,xq,net,opt
 g=graph(d,des,tr,ws,wc);net=init(d,des,int(split.split('_')[1]));opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
 tq=tensor(des['task_embedding'][d['fam'][te]]);xq=tensor(d['x'][te]);arrays={'test_rows':te,'query_ids':d['query_ids'][te]};results={}
 for ep in range(1,int(bestep.max())+1):
  epoch(net,g,opt)
  if ep in bestep:
   p=predict(net,g,tq,xq);ch=p.argmax(1)
   for il in np.flatnonzero(bestep==ep):
    lam=lams[il];arrays[f'{lam:g}|choice']=ch;results[str(lam)]={name:score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)}
  if ep%10==0:print('REFIT',ep,'seconds',round(time.time()-t),flush=True)
 np.savez_compressed(dest/'predictions.npz',**arrays);record.update(results=results,n_train=len(tr),n_test=len(te),seconds=time.time()-t)
 (dest/'result.json').write_text(json.dumps(record,indent=2));print('DONE',panel,split,name,round(time.time()-t),flush=True)
if __name__=='__main__':main()
