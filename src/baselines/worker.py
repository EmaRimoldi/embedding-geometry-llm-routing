"""Run resumable baseline work; release and resubmit allocated GPUs only for real unfinished work."""
import argparse,json,subprocess,sys,time,os,signal
from pathlib import Path
from common import ROOT,sha,destination
from run_mlp import tasks,next_unit,assemble
STOP=False
def stop(signum,frame):
 global STOP
 STOP=True
def verify():
 expected=json.loads((ROOT/'results/baselines/_run/source_hashes.json').read_text())
 for p,h in expected.items():assert sha(ROOT/p)==h,('Code changed after freeze',p)
def call(script,*args):
 subprocess.run([sys.executable,'-u',str(ROOT/'src/baselines'/script),*map(str,args)],check=True)
def continue_job(kind,index):
 cmd=['sbatch','--parsable','--export=ALL,PAPER_CONTINUATION=1',str(ROOT/'jobs'/f'baselines_{kind}.sbatch'),str(index)]
 result=subprocess.run(cmd,capture_output=True,text=True,check=True)
 record=dict(kind=kind,index=index,parent_job=os.environ.get('SLURM_JOB_ID'),job_id=result.stdout.strip(),submitted_at=time.time())
 p=ROOT/'results/baselines/_logs'/f'continuation-{record["parent_job"]}-{index}.json'
 p.write_text(json.dumps(record,indent=2));print('CONTINUED',record,flush=True)
def main(kind,index):
 verify();started=time.monotonic();signal.signal(signal.SIGUSR1,stop)
 if kind=='mlp':
  # Two disjoint workers cover all 46 panel/fold/encoder combinations.
  assigned=list(range(index,len(tasks()),2))
  for task in assigned:
   while next_unit(task) is not None:
    if STOP or time.monotonic()-started>4.75*3600:
     continue_job(kind,index);return
    call('run_mlp.py','--task',task)
   assemble(task)
 else:
  panel,sp,rep=tasks()[index]
  steps=[('knn','run_fixed.py',['--task',index,'--kind','knn']),
         ('prior','run_fixed.py',['--task',index,'--kind','prior']),
         ('clusters','run_clusters.py',['--task',index]),
         ('prox','run_cpu_prox.py',['--task',index])]
  if sp.startswith('iid'):steps.append(('lookups','run_lookups.py',['--task',index]))
  for method,script,args in steps:
   p=destination(method,panel,sp,rep)
   if (p/'result.json').exists():continue
   if STOP or time.monotonic()-started>9.5*3600:
    continue_job(kind,index);return
   if p.exists():
    # Preserve incomplete files before an identical-method retry.
    p.rename(p.with_name(p.name+'_incomplete_'+str(int(time.time()))))
   call(script,*args)
 done=ROOT/'results/baselines/_run/finished_workers';done.mkdir(exist_ok=True)
 (done/f'{kind}_{index}').write_text(os.environ.get('SLURM_JOB_ID','manual'))
 expected=[done/f'cpu_{i}' for i in range(len(tasks()))]+[done/f'mlp_{i}' for i in range(2)]
 if all(p.exists() for p in expected):
  import fcntl
  with (done/'report.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not (done/'report_job.json').exists():
    v=subprocess.run(['sbatch','--parsable',str(ROOT/'jobs/baselines_report.sbatch')],capture_output=True,text=True,check=True)
    (done/'report_job.json').write_text(json.dumps(dict(job_id=v.stdout.strip())))
 print('WORKER_DONE',kind,index,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['cpu','mlp']);ap.add_argument('index',type=int);a=ap.parse_args();main(a.kind,a.index)
