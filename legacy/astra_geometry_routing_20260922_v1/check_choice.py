import itertools
import numpy as np
from choice_signal import signal
rng=np.random.default_rng(4);r=rng.random((5,3));c=rng.random((5,3));ch=np.array([0,1,2,0,1]);groups=np.array([0,0,1,1,1]);lam=.7
values=[]
for a in itertools.permutations(ch[:2]):
    for b in itertools.permutations(ch[2:]):
        perm=np.array(a+b);values.append((r-lam*c)[np.arange(5),perm].mean())
s=signal(ch,r,c,lam,groups)
np.testing.assert_allclose(s['expected_permuted_utility'],np.mean(values))
assert abs(signal(np.zeros(5,int),r,c,lam,groups)['query_specific_utility'])<1e-12
print('PASS: exact conditional permutation expectation and zero signal for constant choice')
