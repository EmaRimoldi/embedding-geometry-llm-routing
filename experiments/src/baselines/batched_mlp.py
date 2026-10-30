"""Independent sklearn-style MLPRegressor heads trained in parallel on a GPU.

Extends the original batched MLP implementation (same initialisation stream, shuffling,
half-MSE, weight-only L2, Adam and training-loss stopping) with two things sklearn exposes and the original
fixed: a per-head `alpha` and `early_stopping` (random `validation_fraction` of the fit rows, R^2 score,
`tol`/`n_iter_no_change` patience, best weights restored at the end). Heads never share parameters; a head
here is one (penalty, alpha, model) regression. code/verify.py checks equivalence with the original class and
with sklearn.MLPRegressor.
"""
import math
import numpy as np
import torch

class BatchedMLP:
 def __init__(self,h,alphas,device='cuda',dtype=torch.float32,max_iter=200,batch_size=200,tol=1e-4,n_iter_no_change=10,
              early_stopping=False,validation_fraction=.1,seed=1234):
  self.h=tuple(h);self.alphas=np.asarray(alphas,float);self.device=device;self.dtype=dtype;self.max_iter=max_iter
  self.batch_size=batch_size;self.tol=tol;self.patience=n_iter_no_change;self.early=early_stopping;self.vf=validation_fraction;self.seed=seed

 def _forward(self,x,weights,biases):
  """x: (batch, dim) shared by every head -> (heads, batch)."""
  a=torch.einsum('bd,hdk->hbk',x,weights[0])+biases[0][:,None,:]
  for w,b in zip(weights[1:],biases[1:]):a=torch.bmm(torch.relu(a),w)+b[:,None,:]
  return a[:,:,0]

 def fit(self,x,y):
  if y.ndim==1:y=y[:,None]
  n,dim=x.shape;heads=y.shape[1];assert len(self.alphas)==heads
  rng=np.random.RandomState(self.seed);widths=[dim,*self.h,1];self.weights=[];self.biases=[]
  for fanin,fanout in zip(widths[:-1],widths[1:]):   # sklearn _init_coef: glorot uniform, coef then intercept
   bound=np.sqrt(6/(fanin+fanout));w=rng.uniform(-bound,bound,(fanin,fanout));b=rng.uniform(-bound,bound,fanout)
   self.weights.append(torch.tensor(np.repeat(w[None],heads,axis=0),dtype=self.dtype,device=self.device,requires_grad=True))
   self.biases.append(torch.tensor(np.repeat(b[None],heads,axis=0),dtype=self.dtype,device=self.device,requires_grad=True))
  if self.early:   # sklearn: train_test_split(X, y, random_state=<same RandomState>, test_size=validation_fraction)
   perm=rng.permutation(n);nv=math.ceil(self.vf*n);vi,ti=perm[:nv],perm[nv:nv+(n-nv)]
   xv=torch.as_tensor(x[vi],dtype=self.dtype,device=self.device);yv=torch.as_tensor(y[vi].T,dtype=self.dtype,device=self.device)
   x=x[ti];y=y[ti];n=len(ti)
   sst=((yv-yv.mean(1,keepdim=True))**2).sum(1)
   best_w=[w.detach().clone() for w in self.weights];best_b=[b.detach().clone() for b in self.biases]
  bs=min(self.batch_size,n)
  params=self.weights+self.biases;mom=[torch.zeros_like(p) for p in params];var=[torch.zeros_like(p) for p in params]
  xx=torch.as_tensor(x,dtype=self.dtype,device=self.device);yy=torch.as_tensor(y.T,dtype=self.dtype,device=self.device)
  alpha=torch.as_tensor(self.alphas,dtype=self.dtype,device=self.device)
  active=np.ones(heads,bool);best=np.full(heads,np.inf);bad=np.zeros(heads,int);self.iterations=np.zeros(heads,int);self.losses=np.zeros(heads)
  self.best_scores=np.full(heads,-np.inf);step=0;indices=np.arange(n)
  for ep in range(1,self.max_iter+1):
   rng.shuffle(indices);accum=np.zeros(heads)
   for st in range(0,n,bs):
    ix=torch.as_tensor(indices[st:st+bs],device=self.device);b=len(ix)
    out=self._forward(xx[ix],self.weights,self.biases)
    loss=.5*((out-yy[:,ix])**2).mean(1)+.5*alpha/b*sum((w*w).sum((1,2)) for w in self.weights)
    accum+=loss.detach().cpu().numpy()*b
    mask=torch.as_tensor(active,dtype=self.dtype,device=self.device);(loss*mask).sum().backward();step+=1
    lr=.001*np.sqrt(1-.999**step)/(1-.9**step)
    with torch.no_grad():
     for p,mm,vv in zip(params,mom,var):
      g=p.grad;mm.mul_(.9).add_(g,alpha=.1);vv.mul_(.999).addcmul_(g,g,value=.001)
      upd=-lr*mm/(torch.sqrt(vv)+1e-8);upd*=mask.reshape((heads,)+(1,)*(p.ndim-1));p.add_(upd);p.grad=None
   cur=accum/n;self.iterations[active]=ep;self.losses[active]=cur[active]
   if self.early:
    with torch.no_grad():
     pv=torch.cat([self._forward(xv[s:s+4096],self.weights,self.biases) for s in range(0,len(xv),4096)],1)
     sse=((pv-yv)**2).sum(1)
     r2=torch.where(sst>0,1-sse/sst.clamp_min(1e-30),torch.where(sse==0,torch.ones_like(sse),torch.zeros_like(sse))).cpu().numpy()
    act=active.copy()
    bad[act]=np.where(r2[act]<self.best_scores[act]+self.tol,bad[act]+1,0)
    imp=act&(r2>self.best_scores);self.best_scores[imp]=r2[imp]
    if imp.any():
     m=torch.as_tensor(imp,device=self.device)
     with torch.no_grad():
      for bw,w in zip(best_w+best_b,self.weights+self.biases):bw[m]=w.detach()[m]
   else:
    act=active.copy();bad[act]=np.where(cur[act]>best[act]-self.tol,bad[act]+1,0);best[act]=np.minimum(best[act],cur[act])
   active&=bad<=self.patience
   if not active.any():break
  if self.early:
   with torch.no_grad():
    for w,bw in zip(self.weights+self.biases,best_w+best_b):w.copy_(bw)
  return self

 def predict(self,x):
  out=[]
  with torch.no_grad():
   for st in range(0,len(x),2048):
    out.append(self._forward(torch.as_tensor(x[st:st+2048],dtype=self.dtype,device=self.device),self.weights,self.biases).T.cpu().numpy())
  return np.concatenate(out)
