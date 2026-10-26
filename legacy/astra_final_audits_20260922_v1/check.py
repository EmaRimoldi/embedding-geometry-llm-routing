import sys,os,ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'datasets'));from audit import hull
sys.path.insert(0,str(ROOT/'geometry'));from intervene import basis_fit
for p in ROOT.rglob('*.py'):
 if 'sources' not in p.parts and not p.name.startswith('._'):ast.parse(p.read_text())
rng=np.random.default_rng(127)
for _ in range(100):
 r=np.round(rng.uniform(size=13),1);c=np.round(rng.uniform(size=13),2);h,b=hull(r,c)
 for lam in np.r_[0,np.logspace(-4,4,80),b]:assert np.isclose((r-lam*c).max(),(r[h]-lam*c[h]).max())
z=rng.normal(size=(300,4));x=rng.normal(size=(300,16));b,meta=basis_fit(x,z,np.ones(300));assert np.allclose(b.T@b,np.eye(meta['rank']),atol=1e-6)
xm=x.mean(0);changed=x-(x-xm)@b@b.T;assert np.allclose(changed.mean(0),xm,atol=1e-7)
# Affine mean is preserved before renormalization; normalization can change the mean, disclosed in report.
# Positive scaling equivalence of CARROT objective and raw-dollar lambda.
r=rng.uniform(size=(30,13));c=rng.uniform(0,.01,size=(30,13))
for mu in [.001,.1,.5,.99]:assert np.array_equal(((1-mu)*r-100*mu*c).argmax(1),(r-(100*mu/(1-mu))*c).argmax(1))
print('PASS: script parsing, random convex-hull oracle checks, subspace orthonormality, pre-normalization source-mean preservation, objective conversion',flush=True)
