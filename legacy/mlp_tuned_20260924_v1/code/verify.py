"""Equivalence gate (GPU, float64): the extended BatchedMLP must reproduce
 (a) the definitive study's BatchedMLP at alpha=1e-4 without early stopping, and
 (b) sklearn.MLPRegressor head by head for several alphas, with and without early stopping (predictions and
     number of epochs), including a constant target.
Writes results/verify.json; exits non-zero on any failure so dependent jobs never start.
"""
import json,sys,warnings
import numpy as np,torch
from sklearn.neural_network import MLPRegressor
from common_mlp import DEF,ROOT
from batched_mlp import BatchedMLP
sys.path.insert(0,str(DEF/'code'));from mlp_batched import BatchedMLP as Original

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(7);x=rng.normal(size=(130,9));xe=rng.normal(size=(23,9));y=rng.normal(size=(130,3));y[:,1]=0;recs=[];ok=True
 # (a) original class, default alpha, no early stopping
 for h in [(7,),(7,5),(7,5,3)]:
  for epochs in (1,6,200):
   old=Original(h,dtype=torch.float64,max_iter=epochs).fit(x,y);new=BatchedMLP(h,[1e-4]*3,dtype=torch.float64,max_iter=epochs).fit(x,y)
   d=float(np.abs(old.predict(xe)-new.predict(xe)).max());it=bool((old.iterations==new.iterations).all());ok&=d<1e-9 and it
   recs.append(dict(test='original',arch=h,epochs=epochs,max_diff=d,same_iterations=it))
 # (b) sklearn, per-head alphas, both stopping rules
 alphas=[1e-4,1e-2,1.,10.,100.]
 for h in [(7,),(7,5)]:
  for early in (False,True):
   A=np.array([a for _ in range(3) for a in alphas]);Y=np.stack([y[:,m] for m in range(3) for _ in alphas],1)   # head = (target, alpha)
   net=BatchedMLP(h,A,dtype=torch.float64,early_stopping=early).fit(x,Y);pred=net.predict(xe);j=0
   for m in range(3):
    for a in alphas:
     ref=MLPRegressor(hidden_layer_sizes=h,alpha=a,random_state=1234,max_iter=200,early_stopping=early)
     with warnings.catch_warnings():warnings.simplefilter('ignore');ref.fit(x,y[:,m])
     d=float(np.abs(ref.predict(xe)-pred[:,j]).max());it=int(ref.n_iter_)==int(net.iterations[j]);ok&=d<1e-8 and it
     recs.append(dict(test='sklearn',arch=h,early_stopping=early,alpha=a,target=m,max_diff=d,sklearn_epochs=int(ref.n_iter_),epochs=int(net.iterations[j]),same_iterations=it));j+=1
 (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/verify.json').write_text(json.dumps(dict(pass_all=bool(ok),tests=recs),indent=1))
 bad=[r for r in recs if not (r['max_diff']<1e-8 and (r.get('same_iterations',True)))]
 print('PASS' if ok else 'FAIL',len(recs),'tests; failures:',json.dumps(bad[:10],indent=1),flush=True)
 sys.exit(0 if ok else 1)

if __name__=='__main__':main()
