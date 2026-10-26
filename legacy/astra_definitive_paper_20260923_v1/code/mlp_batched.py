"""Independent sklearn-style MLP heads evaluated in parallel on a GPU.

Same initialization stream, shuffled batches, half-MSE, weight-only L2,
Adam epsilon convention and training-loss stopping as MLPRegressor. No shared
parameters or multi-task objective. Float64 reference mode is for equivalence QA.
"""
import numpy as np
import torch

class BatchedMLP:
 def __init__(self,h,device='cuda',dtype=torch.float32,max_iter=200,batch_size=200,tol=1e-4,n_iter_no_change=10):
  self.h=h;self.device=device;self.dtype=dtype;self.max_iter=max_iter;self.batch_size=batch_size;self.tol=tol;self.patience=n_iter_no_change
 def fit(self,x,y):
  if y.ndim==1:y=y[:,None]
  n,dim=x.shape;heads=y.shape[1];rng=np.random.RandomState(1234);widths=[dim,*self.h,1]
  self.weights=[];self.biases=[]
  for fanin,fanout in zip(widths[:-1],widths[1:]):
   bound=np.sqrt(6/(fanin+fanout));w=rng.uniform(-bound,bound,(fanin,fanout));b=rng.uniform(-bound,bound,fanout)
   self.weights.append(torch.tensor(np.repeat(w[None],heads,axis=0),dtype=self.dtype,device=self.device,requires_grad=True))
   self.biases.append(torch.tensor(np.repeat(b[None],heads,axis=0),dtype=self.dtype,device=self.device,requires_grad=True))
  params=self.weights+self.biases;mom=[torch.zeros_like(p) for p in params];var=[torch.zeros_like(p) for p in params]
  xx=torch.as_tensor(x,dtype=self.dtype,device=self.device);yy=torch.as_tensor(y.T,dtype=self.dtype,device=self.device)
  active=np.ones(heads,bool);best=np.full(heads,np.inf);bad=np.zeros(heads,int);self.iterations=np.zeros(heads,int);self.losses=np.zeros(heads);self.curves=[];step=0;indices=np.arange(n)
  for ep in range(1,self.max_iter+1):
   rng.shuffle(indices);accum=np.zeros(heads)
   for st in range(0,n,self.batch_size):
    ix=torch.as_tensor(indices[st:st+self.batch_size],device=self.device);bs=len(ix)
    a=xx[ix].unsqueeze(0).expand(heads,-1,-1)
    for il,(w,b) in enumerate(zip(self.weights,self.biases)):
     a=torch.bmm(a,w)+b[:,None,:]
     if il<len(self.weights)-1:a=torch.relu(a)
    loss=.5*((a[:,:,0]-yy[:,ix])**2).mean(1)
    loss=loss+.5*.0001/bs*sum((w*w).sum((1,2)) for w in self.weights)
    accum+=loss.detach().cpu().numpy()*bs
    mask=torch.as_tensor(active,dtype=self.dtype,device=self.device);(loss*mask).sum().backward();step+=1
    lr=.001*np.sqrt(1-.999**step)/(1-.9**step)
    with torch.no_grad():
     for p,mm,vv in zip(params,mom,var):
      grad=p.grad;mm.mul_(.9).add_(grad,alpha=.1);vv.mul_(.999).addcmul_(grad,grad,value=.001)
      update=-lr*mm/(torch.sqrt(vv)+1e-8)
      update*=mask.reshape((heads,)+(1,)*(p.ndim-1));p.add_(update);p.grad=None
   current=accum/n;self.curves.append(current.copy());self.iterations[active]=ep;self.losses[active]=current[active]
   bad[active]=np.where(current[active]>best[active]-self.tol,bad[active]+1,0);best[active]=np.minimum(best[active],current[active]);active &= bad<=self.patience
   if not active.any():break
  return self
 def predict(self,x):
  out=[];heads=self.weights[0].shape[0]
  with torch.no_grad():
   for st in range(0,len(x),2048):
    a=torch.as_tensor(x[st:st+2048],dtype=self.dtype,device=self.device).unsqueeze(0).expand(heads,-1,-1)
    for il,(w,b) in enumerate(zip(self.weights,self.biases)):
     a=torch.bmm(a,w)+b[:,None,:]
     if il<len(self.weights)-1:a=torch.relu(a)
    out.append(a[:,:,0].T.cpu().numpy())
  return np.concatenate(out)

def verify():
 from sklearn.neural_network import MLPRegressor
 import warnings,json
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(99);x=rng.normal(size=(73,9));y=rng.normal(size=(73,3));xe=rng.normal(size=(19,9));records=[]
 for h in [(7,),(7,5),(7,5,3)]:
  for mode,dtype,ndtype,tolerance in [('float64',torch.float64,np.float64,1e-9),('float32',torch.float32,np.float32,2e-5)]:
   for epochs in [1,6,50]:
    xx=x.astype(ndtype);yy=y.astype(ndtype);xt=xe.astype(ndtype)
    net=BatchedMLP(h,dtype=dtype,max_iter=epochs,batch_size=11,tol=0,n_iter_no_change=100).fit(xx,yy);pred=net.predict(xt)
    diffs=[]
    for m in range(3):
     native=MLPRegressor(hidden_layer_sizes=h,random_state=1234,max_iter=epochs,batch_size=11,tol=0,n_iter_no_change=100)
     with warnings.catch_warnings():warnings.simplefilter('ignore');native.fit(xx,yy[:,m])
     diffs.append(float(abs(native.predict(xt)-pred[:,m]).max()))
    assert max(diffs)<tolerance,(h,mode,epochs,diffs)
    records.append(dict(architecture=h,precision=mode,epochs=epochs,max_prediction_difference=max(diffs)))
 # Test native per-head stopping, including a nearly constant target.
 yy=np.stack([y[:,0],np.zeros(len(x)),y[:,2]],1)
 net=BatchedMLP((7,),dtype=torch.float64).fit(x,yy)
 for m in range(3):
  native=MLPRegressor(hidden_layer_sizes=(7,),random_state=1234,max_iter=200)
  with warnings.catch_warnings():warnings.simplefilter('ignore');native.fit(x,yy[:,m])
  assert net.iterations[m]==native.n_iter_,(m,net.iterations[m],native.n_iter_)
  assert np.allclose(net.predict(xe)[:,m],native.predict(xe),atol=1e-9,rtol=1e-9)
 print(json.dumps(dict(equivalence_tests=records,stopping_iterations=net.iterations.tolist(),pass_all=True),indent=2),flush=True)
if __name__=='__main__':verify()
