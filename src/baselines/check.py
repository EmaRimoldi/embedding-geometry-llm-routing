"""Prelaunch checks of the reused MLP trainer and baseline algebra."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from common import ROOT,nearest,score
from batched_mlp import BatchedMLP
from run_fixed import predict_knn
from run_prox import predict
def main():
 rng=np.random.RandomState(71);x=rng.normal(size=(97,9));y=rng.normal(size=(97,2));xe=rng.normal(size=(13,9))
 records=[]
 for h in [(8,8),(8,8,8)]:
  for es in [False,True]:
   for alpha in [.0001,10.]:
    gpu=BatchedMLP(h,[alpha,alpha],device='cpu',dtype=torch.float64,max_iter=18,batch_size=23,early_stopping=es).fit(x,y)
    a=gpu.predict(xe)
    for m in range(2):
     ref=MLPRegressor(hidden_layer_sizes=h,alpha=alpha,random_state=1234,batch_size=23,max_iter=18,early_stopping=es).fit(x,y[:,m])
     err=float(abs(a[:,m]-ref.predict(xe)).max())
     assert err<1e-9,(h,es,alpha,m,err)
     assert gpu.iterations[m]==ref.n_iter_
     records.append(dict(architecture=h,early_stopping=es,alpha=alpha,head=m,max_prediction_difference=err))
 # Deterministic neighbour ties and correctly matched outcome rows.
 xx=np.eye(4,dtype=np.float32);r=rng.uniform(size=(4,3));c=rng.uniform(size=(4,3))
 ix,_=nearest(xx[[0]],xx[[1,2,3]],3);np.testing.assert_array_equal(ix,[[0,1,2]])
 pr=predict_knn(xx,r,c,np.array([1,2,3]),np.array([0]),[2,3])
 np.testing.assert_allclose(pr[2][0],r[[1,2]].mean(0)[None])
 cent=np.eye(2);prior=np.array([1.,2.]);rt=np.array([[1.,0.],[0.,1.]]);ct=rt*.1
 rr,cc=predict(np.eye(2),(cent,prior,rt,ct),0)
 np.testing.assert_allclose(rr,[[1/3,2/3]]*2)
 out=ROOT/'results/baselines/_run';out.mkdir(parents=True,exist_ok=True)
 (out/'prelaunch_checks.json').write_text(json.dumps(dict(pass_all=True,mlp_equivalence=records,knn_ties=True,prox_zero_temperature=True),indent=2))
 print('PASS: 16 sklearn-equivalence cases, retrieval ties, KM-Prox limiting case',flush=True)
if __name__=='__main__':main()
