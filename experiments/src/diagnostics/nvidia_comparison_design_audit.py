"""Read-only label/input-design audit; no fitting, selection or routing changes."""
from pathlib import Path
import hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/nvidia_layer_validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'purpose':'Reference-comparison design audit only; no model fitting or selection','datasets':{}}
for dataset in ['routerbench','sprout']:
 p=ROOT/'datasets'/dataset/'prepared/panel.npz'
 with np.load(p,allow_pickle=False) as z:
  reward=z['reward'];family=z['family']
 meta=json.loads((p.parent/'manifest.json').read_text())
 fractional=(reward!=0)&(reward!=1)
 per_family={}
 for f in sorted(set(family)):
  x=reward[family==f];fr=(x!=0)&(x!=1)
  per_family[str(f)]={'queries':len(x),'fractional_rewards':int(fr.sum()),'total_outcomes':int(x.size),'fraction_fractional':float(fr.mean()),'reward_min':float(x.min()),'reward_max':float(x.max())}
 n=reward.shape[1]
 def params(d):return (d+1)*256+(256+1)*128+(128+1)*n
 report['datasets'][dataset]={'rows':len(reward),'models':n,'panel_sha256':sha(p),'fractional_rewards':int(fractional.sum()),'total_outcomes':int(reward.size),'fraction_fractional':float(fractional.mean()),'per_family':per_family,'proposed_threshold_0_5_positive_counts':(reward>=.5).sum(0).tolist(),'model_names':meta['models'],'PCA100_shared_once_input_dim':100,'current_repeated_input_dim':100*n,'shared_once_parameters':params(100),'repeated_parameters':params(100*n)}
report['source_sha256']=sha(Path(__file__))
report['slurm_job_id']=os.environ.get('SLURM_JOB_ID')
(OUT/'label_and_input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
