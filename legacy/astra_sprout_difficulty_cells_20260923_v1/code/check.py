"""Small numerical checks of denominators, fallback pools and decomposition."""
import numpy as np
from run import abs_expect, compress, evaluate, matched_donor_random, refine

rng=np.random.default_rng(817)
y=rng.uniform(size=(240,17));x=rng.normal(size=(240,20));x/=np.linalg.norm(x,axis=1,keepdims=True)
fam=np.repeat([0,1],120);cells=np.repeat(['a','b','c','d'],60)
tr=np.r_[np.arange(0,55),np.arange(60,62),np.arange(120,175),np.arange(180,235)]
te=np.setdiff1d(np.arange(240),tr)
assert np.allclose(abs_expect(y[te],y[tr]),np.abs(y[te,None]-y[tr][None]).mean(1))
z=evaluate(x,y,tr,te,cells,fam,threshold=50)
for i,q in enumerate(te):
 pool=tr[cells[tr]==cells[q]]
 if len(pool)<50: pool=tr[fam[tr]==fam[q]]
 assert z['pool_n'][i]==len(pool)
 assert np.allclose(z['base_abs'][i],compress(np.abs(y[q]-y[pool]).mean(0)))
 order=pool[np.argsort(-(x[q]@x[pool].T),kind='stable')[:10]]
 assert np.array_equal(z['nn10'][i],order)
 assert np.allclose(z['knn10_mse'][i],compress((y[q]-y[order].mean(0))**2))
# Specifically the rare b cell must receive all family 0 training rows, including a.
idx=np.flatnonzero(cells[te]=='b')[0]
assert z['pool_n'][idx]==57 and z['fallback'][idx]
cut=evaluate(x,y,tr,te,cells,fam,cutoff=.1)
for i,q in enumerate(te):
 pool=tr[cells[tr]==cells[q]];pool=pool[x[q]@x[pool].T <= .1]
 if len(pool)<10: assert np.isnan(cut['base_abs'][i]).all();continue
 assert np.allclose(cut['base_abs'][i],compress(np.abs(y[q]-y[pool]).mean(0)))
# Matched random-donor expectation agrees with direct enumeration.
glob=evaluate(x,y,tr,te,np.zeros(len(y),int),fam)
matched=matched_donor_random(y,tr,te,glob['nn10'],cells)
for i,q in enumerate(te):
 expected=np.mean([np.abs(y[q]-y[tr[cells[tr]==cells[j]]]).mean(0) for j in glob['nn10'][i]],axis=0)
 assert np.allclose(matched[i],compress(expected))
print('PASS: exact absolute baseline; full-family fallback; capped retrieval; matched exclusion pools; donor-mixture expectation')
