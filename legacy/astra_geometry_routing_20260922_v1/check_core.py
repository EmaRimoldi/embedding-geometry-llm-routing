import numpy as np
from common import unit,nearest,contrast
x=unit(np.full((2,3584),30,dtype=np.float16))
np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
rng=np.random.default_rng(4);x=unit(rng.normal(size=(30,8)));q=unit(rng.normal(size=(3,8)))
i,d=nearest(q,x,5);dd=((q[:,None]-x)**2).sum(2)
assert np.array_equal(i,np.argsort(dd,axis=1)[:,:5])
r=rng.random((3,4));c=rng.random((3,4))
v=contrast(np.array([0,1,2]),np.array([1,1,0]),r,c,2)
assert abs(v['utility_difference']-v['reward_contribution']-v['cost_contribution'])<1e-10
print('PASS: float16 input normalization, exact retrieval, utility decomposition')
