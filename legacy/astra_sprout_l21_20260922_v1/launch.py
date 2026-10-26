"""Submit isolated jobs with a strict preparation and extraction-validation gate."""
import subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'launch.json').exists(),'Already submitted; inspect existing jobs before any retry'
def submit(args):
    job=subprocess.check_output(['sbatch','--parsable']+args,cwd=ROOT,universal_newlines=True).strip().split(';')[0]
    print('submitted',job,args,flush=True);return job
jobs={}
def save(): (ROOT/'launch.json').write_text(json.dumps(jobs,indent=2))
jobs['prepare']=submit(['--job-name=astra-sprout-prepare','cpu.sbatch','prepare']);save()
jobs['canary']=submit(['--job-name=astra-sprout-canary','--dependency=afterok:'+jobs['prepare'],'canary.sbatch']);save()
dep='--dependency=afterok:'+jobs['canary']
jobs['baselines']=submit(['--job-name=astra-sprout-baselines','--array=0-17%3',dep,'cpu.sbatch','baselines']);save()
jobs['baseline_report']=submit(['--job-name=astra-sprout-report','--dependency=afterok:'+jobs['baselines'],'cpu.sbatch','report']);save()
for seed in (42,7):
    jobs['kernel_'+str(seed)]=submit(['--job-name=astra-sprout-kernel'+str(seed),'--array=0-17%2',dep,'cpu.sbatch','kernel',str(seed)]);save()
jobs['final_report']=submit(['--job-name=astra-sprout-report','--dependency=afterok:'+':'.join(jobs[k] for k in ('baselines','kernel_42','kernel_7')),'cpu.sbatch','report']);save()
(ROOT/'code_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.sbatch','.md')},indent=2))
