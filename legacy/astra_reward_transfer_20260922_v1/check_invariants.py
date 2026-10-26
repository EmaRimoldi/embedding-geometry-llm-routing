"""Small numerical and leakage tests; no benchmark outcome is used here."""
import numpy as np
from sklearn.linear_model import Ridge
from experiment import ridge_many, center, predict, tune, policy, avg

rng=np.random.default_rng(9)
x=rng.normal(size=(60,8)); y=rng.normal(size=(60,5)); xt=rng.normal(size=(17,8))
w=rng.uniform(.1,2,size=60); w/=w.mean()
ans=ridge_many(x,y,w,xt)
mu=avg(x,w); sc=np.sqrt(avg((x-mu)**2,w))
for a,(yp,_) in ans.items():
    ref=Ridge(alpha=a).fit((x-mu)/sc,y,sample_weight=w).predict((xt-mu)/sc)
    np.testing.assert_allclose(yp,ref,rtol=1e-7,atol=1e-8)

err=center(y[:17])-center(yp)
pair=np.stack([err[:,i]-err[:,j] for i in range(5) for j in range(i+1,5)],axis=1)
np.testing.assert_allclose(np.mean(pair**2),2*5/4*np.mean(err**2))

f=np.repeat(np.arange(5),12); r=rng.uniform(0,1,(60,4)); c=rng.uniform(0,.01,(60,4))
g=rng.normal(size=(60,6)); x=rng.normal(size=(60,7))
outer=4
def run(rr):
    inner={}; truth={}
    for h in range(4):
        tr=np.flatnonzero((f!=outer)&(f!=h)); va=np.flatnonzero(f==h)
        inner[h]=predict(x[tr],g[tr],rr[tr],c[tr],f[tr],x[va],g[va],False)
        truth[h]=(rr[va],c[va])
    pars,sel,_=tune(inner,truth,(0.,1.))
    tr=np.flatnonzero(f!=outer); te=np.flatnonzero(f==outer)
    pr=predict(x[tr],g[tr],rr[tr],c[tr],f[tr],x[te],g[te],False)
    ch=policy(pr,*sel['1.0']['selected_reward_cost'],pars,1.)
    return pars,sel,pr['heads']['direct']['100:4'],ch

first=run(r); changed=r.copy(); changed[f==outer]=rng.uniform(-9,9,(12,4)); second=run(changed)
assert first[0]==second[0] and first[1]==second[1]
np.testing.assert_array_equal(first[2],second[2]); np.testing.assert_array_equal(first[3],second[3])
print('PASS: weighted ridge matches sklearn; pairwise loss identity; outer rewards cannot change fitting, selection or routing.')
