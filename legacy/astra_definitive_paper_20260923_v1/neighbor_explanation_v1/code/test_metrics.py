import numpy as np
from metrics import routing,profiles,summarize
r=np.array([[1.,0.,.5],[.2,.2,.2],[0.,1.,.8]]);c=np.array([[.1,.01,.02],[.1,.1,.1],[.01,.1,.05]])
v,ch=routing(r,c,r,c,0);assert np.all(v['regret']==0) and np.all(v['pair_accuracy'][[0,2]]==1) and np.isnan(v['pair_accuracy'][1])
v,ch=routing(r,c,np.zeros_like(r),np.zeros_like(c),0);assert np.all(v['pair_accuracy'][[0,2]]==.5)
v,ch=routing(r,c,-r,c,0);assert np.all(v['pair_accuracy'][[0,2]]==0)
for k,x in profiles(r,c,r,c).items():assert (x==0).all(),k
for lam in [0,1,10]:
 rr=np.flip(r,axis=1);cc=np.flip(c,axis=1);gg=routing(r,c,r,c,lam)[0]['utility'];qq=routing(r,c,rr,cc,lam)[0]['utility'];gc=routing(r,c,r,cc,lam)[0]['utility'];qg=routing(r,c,rr,c,lam)[0]['utility'];assert np.allclose(.5*(gc-gg+qq-qg)+.5*(qg-gg+qq-gc),qq-gg)
s=summarize({'contrast':np.array([1.,1.,-1.]),'undefined':np.full(3,np.nan)},np.array([0,0,1]),{0:80,1:20},boot=100)
assert abs(s['contrast']['mean']-.6)<1e-12 and np.allclose(s['contrast']['ci95'],[.6,.6]);assert s['undefined']['mean'] is None
print('PASS: target ties, predicted ties, perfect/inverted ordering, regret, swap identity, population weights, paired bootstrap, undefined metrics')
