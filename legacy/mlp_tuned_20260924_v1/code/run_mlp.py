"""Tuned MLP baseline: one (panel, outer split, representation, architecture, early_stopping) per task.

Every (penalty, alpha, model) is an independent sklearn-style regression head trained in one GPU pass, so a
task covers the whole penalty grid and alpha grid for its architecture/stopping rule. For each inner source
fold and for the final outer fit it stores utility SUMS per (penalty, alpha); code/assemble.py selects
(architecture, alpha, early_stopping) per penalty on the inner sums only and scores the stored test choices.

Usage: run_mlp.py --pack 0      (a pack = one panel/split/representation, its 6 configurations in sequence)
       run_mlp.py --list        (print the pack table)
"""
import argparse,json,time
import numpy as np,torch
from common_mlp import PANELS,PROTOCOL,ROOT,load,sha,split_names,splits,splits_path
from batched_mlp import BatchedMLP

def packs():
 out=[]
 for rep in PROTOCOL['representations']:   # GTE (the paper's baseline) first, then the matched L21 comparison
  for panel in PANELS:
   for sp in split_names(panel):out.append((panel,sp,rep))
 return out

def configs():
 return [(ia,h,es) for ia,h in enumerate(PROTOCOL['architectures']) for es in PROTOCOL['early_stopping']]

def heads(r,c,lams,alphas):
 """Targets (n, L*A*M) and per-head alpha; head = (penalty, alpha, model), model fastest."""
 L,A,M=len(lams),len(alphas),r.shape[1]
 u=(r[:,None,:]-np.asarray(lams)[None,:,None]*c[:,None,:]).astype(np.float32)          # n,L,M
 Y=np.broadcast_to(u[:,:,None,:],(len(r),L,A,M)).reshape(len(r),-1)
 return np.ascontiguousarray(Y),np.broadcast_to(np.asarray(alphas)[None,:,None],(L,A,M)).reshape(-1)

def utility_sums(pred,r,c,lams,A):
 n,M=r.shape;L=len(lams);ch=pred.reshape(n,L,A,M).argmax(-1)                            # n,L,A
 U=r[:,None,:]-np.asarray(lams)[None,:,None]*c[:,None,:]                                 # n,L,M
 got=np.take_along_axis(np.broadcast_to(U[:,:,None,:],(n,L,A,M)),ch[...,None],-1)[...,0]
 return got.sum(0),ch

def run(panel,sp,rep,ia,h,es,d,s):
 dest=ROOT/'results/runs'/panel/sp/rep/f'arch{ia}_es{int(es)}'
 if (dest/'result.json').exists():print('ALREADY',panel,sp,rep,ia,es,flush=True);return
 dest.mkdir(parents=True,exist_ok=True);t=time.time();lams=PROTOCOL['lambdas'][panel];alphas=PROTOCOL['alphas'];A=len(alphas)
 x,r,c=d['x'],d['r'],d['c'];Y,al=heads(r,c,lams,alphas);tr=np.array(s['train']);te=np.array(s['test'])
 inner=np.zeros((len(lams),A));vn=0;idiag=[]
 for k,q in enumerate(s['inner']):
  fit=np.array(q['fit']);val=np.array(q['val'])
  net=BatchedMLP(h,al,early_stopping=es).fit(x[fit],Y[fit]);u,_=utility_sums(net.predict(x[val]),r[val],c[val],lams,A);inner+=u;vn+=len(val)
  idiag.append(dict(inner=k,epochs_median=float(np.median(net.iterations)),epochs_max=int(net.iterations.max())))
  print('INNER',panel,sp,rep,h,es,k,round(time.time()-t),flush=True)
 net=BatchedMLP(h,al,early_stopping=es).fit(x[tr],Y[tr]);test,ch=utility_sums(net.predict(x[te]),r[te],c[te],lams,A)
 it=net.iterations.reshape(len(lams),A,-1);ls=net.losses.reshape(len(lams),A,-1)
 np.savez_compressed(dest/'choices.npz',test_rows=te,choice=ch.astype(np.int8))
 rec=dict(panel=panel,split=sp,representation=rep,architecture=list(h),early_stopping=es,lambdas=lams,alphas=alphas,
  inner_sum=inner.tolist(),inner_n=vn,test_sum=test.tolist(),n_test=len(te),
  final_epochs=it.tolist(),final_train_loss=ls.tolist(),inner_diagnostics=idiag,
  split_sha256=sha(splits_path(panel)),code_sha256=sha(__file__),backend_sha256=sha(ROOT/'code/batched_mlp.py'),
  gpu=torch.cuda.get_device_name(0),seconds=time.time()-t)
 (dest/'result.json.tmp').write_text(json.dumps(rec));(dest/'result.json.tmp').rename(dest/'result.json')
 print('DONE',panel,sp,rep,h,es,round(time.time()-t),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pack',type=int);ap.add_argument('--list',action='store_true');a=ap.parse_args()
 P=packs()
 if a.list:
  for i,p in enumerate(P):print(i,*p)
  return
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 panel,sp,rep=P[a.pack];d=load(panel,rep);s=splits(panel)[sp]
 for ia,h,es in configs():run(panel,sp,rep,ia,h,es,d,s)

if __name__=='__main__':main()
