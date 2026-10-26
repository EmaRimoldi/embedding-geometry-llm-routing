"""Complete performance comparison and supplemental Slurm/PCA cost accounting."""
import collections,csv,datetime,json,os,subprocess
from pathlib import Path
from common_nv import ROOT,OUT,cfg,base,splits,folds,rowsets,transform_dir,sha,verify,atomic_json
def read_csv(p):return list(csv.DictReader(p.open()))
def write_csv(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 verify()
 assert json.loads((OUT/'summary/audit.json').read_text())['all_complete']
 ip=json.loads((OUT/'inference_cpu_v1/latest_audit.json').read_text())
 ia=json.loads((ROOT/ip['path']).read_text())
 assert ia['all_complete'] and ia['audited_latency_rows']==1794
 dest=OUT/'final_comparison';dest.mkdir(exist_ok=True)
 groups=collections.defaultdict(list)
 paths=[OUT/'summary/scores.csv',ROOT/'results/baselines/_summary_reduced/scores.csv',ROOT/'results/baselines/_summary/ridge_scores.csv']
 hashes={}
 for p in paths:
  hashes[str(p.relative_to(ROOT))]=sha(p)
  for r in read_csv(p):
   assert r['complete']=='True',r
   enc=r.get('encoder','nvidia_pipeline')
   groups[(r['dataset'],r['protocol'],enc,r['method'])].append(r)
 means=[];perlam=[]
 for key,rs in sorted(groups.items()):
  assert len(rs)==13 and len({float(r['lambda_value']) for r in rs})==13
  means.append(dict(zip(['dataset','protocol','encoder','method'],key),mean_utility=sum(float(r['utility']) for r in rs)/13))
  for r in rs:perlam.append(dict(zip(['dataset','protocol','encoder','method'],key),lambda_value=float(r['lambda_value']),utility=float(r['utility'])))
 write_csv(dest/'all_methods_mean_utility.csv',means);write_csv(dest/'all_methods_per_lambda.csv',perlam)
 pairwise=[]
 for panel in base.PANELS:
  for protocol in ['iid','ood']:
   for control in ['selected_layer','fixed_l21']:
    nv={float(r['lambda_value']):float(r['utility']) for r in groups[panel,protocol,'nvidia_pipeline',control]}
    for (pa,pr,encoder,method),rs in sorted(groups.items()):
     if pa!=panel or pr!=protocol or encoder=='nvidia_pipeline':continue
     diff=[nv[float(r['lambda_value'])]-float(r['utility']) for r in rs]
     pairwise.append(dict(dataset=panel,protocol=protocol,nvidia_control=control,baseline_encoder=encoder,
       baseline_method=method,mean_nvidia_minus_baseline=sum(diff)/13,
       positive_penalties=sum(d>0 for d in diff),negative_penalties=sum(d<0 for d in diff),
       exact_equal_penalties=sum(d==0 for d in diff)))
 write_csv(dest/'all_pairwise_penalty_comparisons.csv',pairwise)
 # PCA study work and the transforms required by each deployable pipeline.
 required={'selected_pipeline_with_screening':set(),'fixed_l21_without_screening':set(),'shared_screening':set(),
           'selected_final':set(),'fixed_l21_final':set()}
 for panel,sp in cfg()['tasks']:
  tr,_=rowsets(panel,sp,'final')
  rec=json.loads((OUT/'results'/base.DATASETS[panel]/sp/'result.json').read_text())
  for i in folds(panel,sp):
   it,_=rowsets(panel,sp,'inner_'+str(i))
   for layer in cfg()['layers']:
    required['shared_screening'].add(str(transform_dir(panel,it,layer)/'ready.json'))
  required['selected_final'].add(str(transform_dir(panel,tr,rec['selection']['layer'])/'ready.json'))
  required['fixed_l21_final'].add(str(transform_dir(panel,tr,21)/'ready.json'))
 required['selected_pipeline_with_screening']=required['shared_screening']|required['selected_final']
 required['fixed_l21_without_screening']=required['fixed_l21_final']
 required['actual_study_all_precomputed']={str(p) for p in (OUT/'pca').glob('**/ready.json')}
 required['overlap_selected_pipeline_fixed_control']=required['selected_pipeline_with_screening']&required['fixed_l21_without_screening']
 pcatotals={}
 for name,files in required.items():
  totals=collections.defaultdict(float)
  for p in files:
   r=json.loads(Path(p).read_text())
   for k,v in r['timing'].items():
    if k.endswith('_seconds'):totals[k]+=float(v)
  pcatotals[name]=dict(transforms=len(files),seconds=dict(totals))
 # Refresh terminal scheduler states and correctly classify Slurm array task IDs.
 submission=json.loads((OUT/'submission.json').read_text());jobs={}
 for key in ['preflight','timing_preflight','timing_report_check','extraction_array','pca_array']:
  if submission.get(key):jobs[str(submission[key])]=key
 for j in submission['training_workers']:jobs[str(j)]='training'
 for p in (OUT/'_logs').glob('continuation-*.json'):jobs[str(json.loads(p.read_text())['job_id'])]='training_continuation'
 jobs[str(json.loads((OUT/'finished/report.json').read_text())['job_id'])]='performance_and_compute_audit'
 ins=json.loads((OUT/'inference_cpu_v1/submission.json').read_text())
 for key in ['job_id','resume_job','partial_audit_job','final_audit_job']:
  if ins.get(key):jobs[str(ins[key])]='inference_'+key
 fields=['JobID','JobIDRaw','JobName','State','ExitCode','ElapsedRaw','AllocCPUS','AllocTRES','Submit','Eligible','Start','End','NodeList']
 text=subprocess.check_output(['sacct','-X','-P','-n','-j',','.join(jobs),'--format='+','.join(fields)],text=True)
 (dest/'scheduler_raw.txt').write_text(text)
 scheduler=[];allocation=collections.defaultdict(lambda:dict(jobs=0,gpu_hours=0.,cpu_core_hours=0.,failed_gpu_hours=0.,failed_cpu_core_hours=0.))
 for line in text.splitlines():
  r=dict(zip(fields,line.split('|')))
  category=jobs[r['JobID'].split('_')[0]];r['category']=category
  tres=dict(v.split('=',1) for v in r['AllocTRES'].split(',') if '=' in v)
  hours=float(r['ElapsedRaw'])/3600;gh=hours*int(tres.get('gres/gpu',0));ch=hours*int(r['AllocCPUS'])
  r.update(allocated_gpu_hours=gh,allocated_cpu_core_hours=ch)
  for a,b in [('Submit','Start'),('Eligible','Start')]:
   try:r[a.lower()+'_to_start_seconds']=(datetime.datetime.fromisoformat(r[b])-datetime.datetime.fromisoformat(r[a])).total_seconds()
   except ValueError:r[a.lower()+'_to_start_seconds']=None
  assert r['State'] in ['COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL'],r
  v=allocation[category];v['jobs']+=1;v['gpu_hours']+=gh;v['cpu_core_hours']+=ch
  if r['State']!='COMPLETED':v['failed_gpu_hours']+=gh;v['failed_cpu_core_hours']+=ch
  scheduler.append(r)
 assert len([r for r in scheduler if r['category']=='extraction_array'])==2
 assert len([r for r in scheduler if r['category']=='pca_array'])==28
 timing=json.loads((OUT/'timing/summary.json').read_text())
 report=dict(all_complete=True,performance_audit=json.loads((OUT/'summary/audit.json').read_text()),
  inference_audit_path=ip['path'],inference_choice_disagreements=ia['numerical_choice_disagreements'],
  source_score_hashes=hashes,pca_accounting=pcatotals,allocation_by_category=dict(allocation),
  scheduler=scheduler,network_phases=timing['network_phases'],incremental_extraction=timing['incremental_extraction'],
  limitations=['Complete utility adaptation of the NVIDIA pipeline, not its published Fisher-based selection or an equal-compute architecture comparison.',
  'All mean utilities average the thirteen prescribed penalties equally after query-weighted fold pooling; pairwise sign counts are descriptive, not significance tests.',
  'Actual study work includes extra fixed-L21 controls and unused precomputed transforms. Selected and fixed-control work overlap and must not be added without deduplication.',
  'Feature extraction is incremental: cached L21/L23/L25 were reused. Encoder work is separate from CPU inference.',
  'Scheduler allocation and actual phase times differ; waiting includes holds/dependencies when measured from submission. Queue time is not compute.',
  'The final-comparison formatting job itself is outside the listed experiment/audit allocation totals.',
  'kNN100 has no fitted network or k-search; source preparation/retrieval is measured. Validated-kNN bundled historical runtimes include search and cannot be substituted.',
  'CPU latency uses up to128 fixed test queries per split, four threads, warmup and three repetitions; report per CPU model and batch, separate setup, and no inference about GPU latency.'])
 atomic_json(dest/'report.json',report)
 lines=['# Completed NVIDIA comparison','','All23 evaluations and both controls audited;598 final scores,6552 candidate validation scores. Full CPU timing coverage:1794 method/batch/penalty rows.','','## Mean utility across penalties','','| Dataset | Protocol | Encoder | Method | Utility |','|---|---|---|---|---:|']
 for r in means:
  if r['encoder']=='nvidia_pipeline' or (r['encoder']=='reader' and r['method'] in ['MLP','kNN-100','Family Adaptation','Validated kNN']):
   lines.append('| '+' | '.join(str(r[k]) for k in ['dataset','protocol','encoder','method'])+' | '+format(r['mean_utility'],'.6f')+' |')
 lines+=['','## Limits','']+['- '+x for x in report['limitations']]
 (dest/'report.md').write_text('\n'.join(lines)+'\n')
 print('FINAL_COMPARISON_COMPLETE',len(means),'means;',len(pairwise),'pairwise comparisons;',len(scheduler),'allocations')
if __name__=='__main__':main()
