import numpy as np
from geometry import nearest,unit,transforms,behavior
rng=np.random.default_rng(3);x=rng.normal(size=(120,16)).astype(np.float32)
# Regression: normalization of valid half-precision cached vectors must not
# silently collapse to zero when squared values overflow float16.
big=np.full((3,3584),30,dtype=np.float16)
np.testing.assert_allclose(np.linalg.norm(unit(big),axis=1),1,atol=1e-6)
np.testing.assert_allclose(unit(big),unit(big.astype(np.float32)))
fam=np.repeat(np.arange(3),40); fmt=fam%2; train=np.flatnonzero(fam!=0);qi=np.arange(8)
modes,get,meta=transforms(x,fam,train,fmt,4)
for mode in modes:
    z=get(mode);ii,dd=nearest(z[qi],z[train],k=20);nb=train[ii]
    assert not np.any(fam[nb]==0)
    brute=((z[qi,None]-z[train][None])**2).sum(2)
    np.testing.assert_allclose(dd,np.take_along_axis(brute,ii,axis=1),atol=1e-4)
    assert np.array_equal(ii,np.argsort(brute,axis=1)[:,:20])
v=meta['axis_removed_variance_fraction']
for k,a in v.items():np.testing.assert_allclose(a,v['format_removed'],rtol=1e-4)
ii,dd=nearest(unit(x[qi]),unit(x),k=20,query_ids=qi,source_ids=np.arange(120))
assert np.all(ii!=qi[:,None])
y=rng.normal(size=(120,3)); arrays={}; out=behavior(ii,qi,np.arange(120),y,np.ones(120,bool),fam,np.zeros(120,int),arrays,'test')
manual=[]
for i,q in enumerate(qi):
    values=[]
    for donor in ii[i,:5]:
        pool=np.flatnonzero((fam==fam[donor])&(np.arange(120)!=q))
        values.append(((y[pool]-y[q])**2).mean())
    manual.append(np.mean(values))
np.testing.assert_allclose(out['5']['matched_random_pair_mse'],np.mean(manual))
print('PASS: true family removal; exact neighbors; variance-matched axis removal; matched donor expectation excludes self.')
