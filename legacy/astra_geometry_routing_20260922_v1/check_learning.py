import numpy as np
from sklearn.linear_model import Ridge
from learning_common import unit
from predictors import ridge,kernel_fit,weights
from nonlinear import select,decide
rng=np.random.default_rng(42);x=rng.normal(size=(75,12));xt=rng.normal(size=(8,12));y=rng.normal(size=(75,4));f=np.repeat(np.arange(3),25);w=weights(f)
for a,p in ridge(x,y,w,xt).items():
    ref=Ridge(alpha=a).fit(x,y,sample_weight=w).predict(xt)
    np.testing.assert_allclose(p,ref,atol=1e-9)
r=rng.random((75,3));c=rng.random((75,3))*.01;ctrl=rng.normal(size=(75,2));ct=rng.normal(size=(8,2))
pred,info=kernel_fit(unit(x),ctrl,r,c,f,unit(xt),ct,42)
assert len(pred)==18 and all(a.shape==(8,3) and b.shape==(8,3) for a,b in pred.values())
assert all(np.isfinite(a).all() and np.isfinite(b).all() and (a>=0).all() and (a<=1).all() and (b>=0).all() for a,b in pred.values())
pr=np.ones((8,3))*.5;pc=np.ones((8,3))*.01;pc[:,1]=0
ch=decide(pred,pr,pc,('prior',0.,0.),1.)
assert (ch==1).all()
mutated={k:(a+999,b+999) for k,(a,b) in pred.items()}
assert np.array_equal(ch,decide(mutated,pr,pc,('prior',0.,0.),1.))
inner={g:dict(pred=pred,prior_r=pr,prior_c=pc,r=pr,c=pc) for g in (0,1,2)}
selected,scores=select(inner,(0.,1.))
assert selected['1']['one_se_blend']==('prior',0.,0.)
print('PASS: weighted ridge reference, kernel outputs, exact fallback, conservative selection')
