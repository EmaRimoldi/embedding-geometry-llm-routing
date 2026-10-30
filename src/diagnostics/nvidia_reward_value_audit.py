"""Describe released numeric reward values; no fitting or threshold selection."""
from pathlib import Path
import numpy as np,json,os
r=Path(__file__).resolve().parents[2]
report={}
for dataset in ['routerbench','sprout']:
 with np.load(r/'datasets'/dataset/'prepared/panel.npz',allow_pickle=False) as z:
  y=z['reward'];fam=z['family']
 values={}
 for f in sorted(set(fam)):
  a=y[fam==f].ravel();v,c=np.unique(a,return_counts=True);ix=np.argsort(-c,kind='stable')[:12]
  frac=a[(a>0)&(a<1)]
  values[str(f)]={'top_values':[[float(v[i]),int(c[i])] for i in ix],'distinct_values':len(v),'fractional_quantiles':np.quantile(frac,[0,.1,.5,.9,1]).tolist() if len(frac) else [],'within_1e_6_of_binary':int((np.minimum(abs(a),abs(a-1))<1e-6).sum()),'total':len(a)}
 report[dataset]=values
report['job_id']=os.environ.get('SLURM_JOB_ID')
(r/'results/nvidia_layer_validation/reward_value_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
