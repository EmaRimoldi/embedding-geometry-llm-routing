import json
import numpy as np
from common import ROOT,load
d=load('sprout-sixfam','reader');s=json.loads((ROOT/'inputs/sprout-sixfam/splits.json').read_text())['ood_5'];tr=np.array(s['train']);te=np.array(s['test']);j=179;k=200
z=np.load(ROOT/'results/knn/sprout-sixfam/ood_5/reader/predictions.npz');x=d['x'];xx=x[tr];q=x[te[j]]
single=xx@q;batch=(x[te[128:256]]@xx.T)[j-128];double=xx.astype(np.float64)@q.astype(np.float64)
out={};sets={}
for name,sim in [('single32',single),('batch32',batch),('float64',double)]:
 order=np.argsort(-sim,kind='stable');ix=order[:k];sets[name]=set(ix.tolist());nb=tr[ix]
 out[name]=dict(max_reward_difference=float(abs(d['r'][nb].mean(0)-z['knn200|r'][j]).max()),boundary=[dict(position=p,source_row=int(tr[order[p]]),cosine=float(sim[order[p]])) for p in range(k-3,k+3)])
out['max_float32_dot_difference']=float(abs(single-batch).max())
out['symmetric_differences']={a+'_'+b:[dict(source_row=int(tr[i]),single32=float(single[i]),batch32=float(batch[i]),float64=float(double[i])) for i in sorted(sets[a]^sets[b])] for a,b in [('single32','batch32'),('single32','float64'),('batch32','float64')]}
(ROOT/'reports/retrieval_numerical_diagnostic.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
