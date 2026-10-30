"""Report measured study costs, never use timing to select a routing model."""
import collections,json,re,subprocess,time
from common_nv import *

def aggregate(records):
 out={'records':len(records),'timed_records':0,'seconds':{},'hardware':{}}
 for rec in records:
  t=rec.get('timing')
  if not t:continue
  out['timed_records']+=1
  for k,v in t.items():
   if k.endswith('_seconds'):out['seconds'][k]=out['seconds'].get(k,0)+float(v)
  h=t.get('hardware',{});key=' | '.join(str(h.get(k,'')) for k in ['gpu_name','cpu_model','cpu_threads'])
  out['hardware'][key]=out['hardware'].get(key,0)+1
 return out

def main():
 verify();c=cfg();dest=OUT/'timing';dest.mkdir(exist_ok=True)
 records=[json.loads(p.read_text()) for p in (OUT/'fits').glob('**/seed_*.json')]
 groups=collections.defaultdict(list)
 for rec in records:
  stage='screening' if rec['identity']['stage']!='final' else 'final_all_controls_unique'
  groups[stage].append(rec)
 selections={}
 for p in (OUT/'results').glob('**/selection.json'):
  selections[(p.parent.parent.name,p.parent.name)]=json.loads(p.read_text())['layer']
 for rec in groups['final_all_controls_unique']:
  i=rec['identity'];selected=selections.get((base.DATASETS[i['panel']],i['split']))
  if i['layer']==21:groups['fixed_l21_final'].append(rec)
  if i['layer']==selected:groups['selected_layer_final'].append(rec)
  if i['layer']==21 and selected==21:groups['shared_final_between_controls'].append(rec)
 pca=[json.loads(p.read_text()) for p in (OUT/'pca').glob('**/ready.json')]
 extraction=[json.loads(p.read_text()) for p in (dest/'extraction').glob('*.json')]
 submission_path=OUT/'submission.json'
 submission=json.loads(submission_path.read_text()) if submission_path.exists() else {'training_workers':[]}
 jobs={}
 for k in ['preflight','timing_preflight','extraction_array','pca_array']:
  if submission.get(k):jobs[str(submission[k]).split(';')[0]]=k
 for j in submission['training_workers']:jobs[str(j).split(';')[0]]='training'
 for p in (OUT/'_logs').glob('continuation-*.json'):
  r=json.loads(p.read_text());jobs[str(r['job_id']).split(';')[0]]='training_continuation'
 for p in [OUT/'finished/report.json']:
  if p.exists():jobs[str(json.loads(p.read_text())['job_id']).split(';')[0]]='audit'
 fields=['JobIDRaw','JobName','State','ElapsedRaw','AllocCPUS','AllocTRES','CPUTimeRAW','TotalCPU','Submit','Eligible','Start','End','NodeList']
 scheduler=[];error=None
 try:
  if not submission_path.exists():
   raise ValueError('Scheduler accounting unavailable: new-run submission.json was not supplied; measured phase timings remain available')
  if not jobs:
   raise ValueError('Scheduler accounting unavailable: submission.json contains no study job IDs')
  raw=subprocess.check_output(['sacct','-X','-P','-n','-j',','.join(jobs),'--format='+','.join(fields)],text=True)
  for line in raw.splitlines():
   parts=line.split('|');r=dict(zip(fields,parts));jid=r.get('JobIDRaw','')
   if not jid or '.' in jid:continue
   r['category']=jobs.get(jid,jobs.get(jid.split('_')[0],'other'))
   seconds=float(r.get('ElapsedRaw') or 0);tres=dict(x.split('=',1) for x in r.get('AllocTRES','').split(',') if '=' in x)
   gpu=int(tres.get('gres/gpu',0));r['allocated_gpu_hours']=seconds*gpu/3600
   r['allocated_cpu_core_hours']=seconds*int(r.get('AllocCPUS') or 0)/3600
   scheduler.append(r)
 except (OSError,subprocess.CalledProcessError,ValueError) as e:error=str(e)
 atomic_json(dest/'scheduler.json',dict(captured_at=time.time(),jobs=jobs,records=scheduler,error=error))
 report=dict(captured_at=time.time(),config_sha256=sha(CONFIG),network_phases={k:aggregate(v) for k,v in groups.items()},pca_all_candidate_study=aggregate(pca),incremental_extraction=extraction,completed_outer_results=len(list((OUT/'results').glob('**/result.json'))),expected_outer_results=len(c['tasks']),planned_screening_fits=sum(len(folds(p,s))*len(c['layers'])*len(c['lambdas'][p]) for p,s in c['tasks']),scheduler_error=error)
 report['notes']=[
  'Network totals sum measured work; they are not calendar completion time. Queue waiting is excluded.',
  'Selected and fixed-L21 totals overlap when layer21 is selected; unique actual work is screening plus final_all_controls_unique.',
  'Final training includes all ten candidates, not only the retained five. The extra fixed-L21 control is study overhead.',
  'PCA totals include all precomputed candidate transforms, including unused outer transforms; this is total study work, not deployment cost.',
  'Extraction is incremental and reuses cached layers. Shared encoder cost must be disclosed separately in end-to-end comparisons.',
  'Scheduler GPU hours are allocated hours, not measured utilization. Failed jobs and continuations remain included. CPU core hours are allocated, not actual CPU usage.',
  'Layer search here uses our utility-validation adaptation, not the published Fisher-score selection. Do not attribute its cost to the published method.',
  'kNN-100 has no k search in the fixed-representation comparison. Historical L21 selection is separate. Validated kNN does search k.',
  'Existing bundled kNN runtime includes validated kNN and cannot be labelled fixed kNN-100 training time.',
  'Matched-hardware inference benchmark remains required; setup speed does not establish prediction speed or performance superiority.'
 ]
 atomic_json(dest/'summary.json',report)
 lines=['# NVIDIA compute accounting','',f"Completed outer evaluations: {report['completed_outer_results']}/{report['expected_outer_results']}",'','| Phase | Network fits | Training hours | Prediction hours |','|---|---:|---:|---:|']
 for k,v in report['network_phases'].items():lines.append(f"| {k} | {v['records']} | {v['seconds'].get('training_seconds',0)/3600:.4f} | {v['seconds'].get('prediction_seconds',0)/3600:.4f} |")
 if error:lines+=['','Scheduler accounting unavailable: '+error]
 lines+=['','## Interpretation','']+['- '+x for x in report['notes']]
 (dest/'summary.md').write_text('\n'.join(lines)+'\n')
 print('COMPUTE_REPORT',len(records),'network fits;',len(pca),'PCA transforms;',len(extraction),'extraction records;',len(scheduler),'scheduler rows')
if __name__=='__main__':main()
