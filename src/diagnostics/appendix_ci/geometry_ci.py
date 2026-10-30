"""Uncertainty for frozen locality and predicted-label lookup measurements."""
import sys,json,os
import numpy as np
from ci_common import ROOT,OUT,CFG,load_panel,write,track,boot_sums,record
def aggregate(folds,n,pooled=False,reps=None,seed=None,family=None):
 sums=np.zeros((n,folds[0][1].shape[1]));weights=np.zeros(n)
 total=sum(len(te) for te,v in folds)
 for te,v in folds:
  w=1/total if pooled else 1/(len(folds)*len(te))
  sums[te]+=v*w;weights[te]+=w
 use=np.flatnonzero(weights>0)
 boot=boot_sums(sums[use],weights[use],family[use] if family is not None else None,reps,seed)
 return sums.sum(0)/weights.sum(),boot,use
def run(task):
 dataset=['routerbench','sprout'][task];panel=['routerbench-0shot','sprout-sixfam'][task]
 out=OUT/'geometry'/dataset
 assert not (out/'report.json').exists()
 d=load_panel(dataset);n=len(d['reward']);prov={};track(ROOT/'datasets'/dataset/'prepared/panel.npz',prov)
 rows=[];arrays={}
 for rep,sub in [('gte-qwen2-7b','primary'),('gte-qwen2-7b','shared_minilm'),('minilm-l12','shared_minilm')]:
  folds=[]
  for k in range(5):
   p=ROOT/'results/geometry'/rep/dataset/sub/f'iid_{k}'
   z=np.load(p/'per_query.npz');s=np.load(p/'split.npz');j=json.loads((p/'metrics.json').read_text())
   orig=z['original_rows'];te=orig[z['unrestricted/query_rows']];nn=orig[z['unrestricted/neighbors'][:,:100]];tr=s['train_rows']
   np.testing.assert_array_equal(te,s['test_rows']);assert not np.intersect1d(te,tr).size
   share=(d['family'][nn]==d['family'][te,None]).mean(1)
   random=np.array([(d['family'][tr]==f).mean() for f in d['family'][te]])
   old=next(r for r in j['records'] if r['block']=='unrestricted' and r['k']==100)
   assert abs(share.mean()-old['same_family_share'])<1e-12 and abs(random.mean()-old['random_same_family_share'])<1e-12
   folds.append((te,np.column_stack([share,random])))
   track(p/'per_query.npz',prov);track(p/'metrics.json',prov)
  point,boot,use=aggregate(folds,n)
  for k,metric in enumerate(['same_family','random_same_family']):
   rows.append(dict(scope='family_locality',encoder=rep,subset=sub,metric=metric,**record(point[k],boot[:,k])))
  arrays['family|'+rep+'|'+sub]=boot
 for rep in ['gte-qwen2-7b','qwen2.5-7b-instruct/layer_21']:
  folds=[]
  for k in range(len(np.unique(d['family']))):
   p=ROOT/'results/geometry'/rep/dataset/'primary'/f'ood_{k}'
   z=np.load(p/'per_query.npz');s=np.load(p/'split.npz');j=json.loads((p/'metrics.json').read_text())
   orig=z['original_rows'];te=orig[z['unrestricted/query_rows']];nn=orig[z['unrestricted/neighbors'][:,:100]];tr=s['train_rows']
   np.testing.assert_array_equal(te,s['test_rows'])
   share=(d['format'][nn]==d['format'][te,None]).mean(1)
   random=np.array([(d['format'][tr]==f).mean() for f in d['format'][te]])
   old=next(r for r in j['records'] if r['block']=='unrestricted' and r['k']==100)
   assert abs(share.mean()-old['same_format_share'])<1e-12 and abs(random.mean()-old['random_same_format_share'])<1e-12
   folds.append((te,np.column_stack([share,random])))
   track(p/'per_query.npz',prov);track(p/'metrics.json',prov)
  point,boot,use=aggregate(folds,n,pooled=True,family=d['family'])
  for k,metric in enumerate(['same_format','random_same_format']):
   rows.append(dict(scope='answer_format',encoder=rep,subset='primary',metric=metric,**record(point[k],boot[:,k])))
  arrays['format|'+rep]=boot
 # Predicted-label deployment, paired with the exact same true-label lookup.
 lams=json.loads((ROOT/'configs/baselines.json').read_text())['lambdas'][panel]
 for kind in ['family','cell']:
  folds=[]
  for k in range(5):
   p=ROOT/'results/diagnostics/deployable_lookup'/panel/f'iid_{k}';z=np.load(p/'predictions.npz');j=json.loads((p/'result.json').read_text());te=z['test_rows'];ii=np.arange(len(te))
   pred=[];true=[]
   for lam in lams:
    u=d['reward'][te]-lam*d['cost'][te]
    pred.append(u[ii,z[f'{kind}|{lam:g}|predicted_choice']]);true.append(u[ii,z[f'{kind}|{lam:g}|true_choice']])
   pred=np.mean(pred,axis=0);true=np.mean(true,axis=0)
   acc=z[kind+'_predicted_labels']==z[kind+'_true_labels']
   assert abs(acc.mean()-j['classifier'][kind]['accuracy'])<1e-12
   original=[r for r in j['scores'] if r['kind']==kind]
   assert abs(pred.mean()-np.mean([r['predicted_utility'] for r in original]))<1e-10
   assert abs(true.mean()-np.mean([r['true_utility'] for r in original]))<1e-10
   folds.append((te,np.column_stack([acc,true,pred,pred-true])))
   track(p/'result.json',prov);track(p/'predictions.npz',prov)
  point,boot,use=aggregate(folds,n)
  for k,metric in enumerate(['accuracy','known_utility','predicted_utility','difference']):
   rows.append(dict(scope='deployability',group=kind,metric=metric,**record(point[k],boot[:,k])))
  arrays['deployability|'+kind]=boot
 # Preserve prior IID disagreement bootstrap, including its exact sample sequence.
 p=ROOT/'results/diagnostics/iid_lookup_disagreement';z=np.load(p/'paired_metrics.npz');j=json.loads((p/'report.json').read_text())
 track(p/'paired_metrics.npz',prov);track(p/'report.json',prov)
 for ref in ['family_lookup','family_topic_lookup']:
  folds=[]
  for k in range(5):
   prefix=f'{panel}|{ref}|iid_{k}|'
   folds.append((z[prefix+'rows'],np.column_stack([z[prefix+'delta'],z[prefix+'disagree']])))
  point,boot,use=aggregate(folds,n,reps=1000,seed=42)
  nl=len(lams);oldsummary=next(v for v in j['summaries'] if v['panel']==panel and v['reference']==ref)
  np.testing.assert_allclose(np.quantile(boot[:,:nl].mean(1),[.025,.975]),oldsummary['mean_gain_ci95'],atol=1e-12)
  for k,lam in enumerate(lams):
   old=next(v for v in j['per_penalty'] if v['panel']==panel and v['reference']==ref and v['lambda_value']==lam)
   np.testing.assert_allclose(np.quantile(boot[:,k],[.025,.975]),old['ci95'],atol=1e-12)
   for metric,value,samples in [('overall_gain',point[k],boot[:,k]),('different_choices',point[nl+k],boot[:,nl+k]),('conditional_gain',point[k]/point[nl+k],boot[:,k]/boot[:,nl+k])]:
    rows.append(dict(scope='disagreement',reference=ref,lambda_value=f'{lam:g}',metric=metric,**record(value,samples)))
  for metric,value,samples in [('overall_gain',point[:nl].mean(),boot[:,:nl].mean(1)),('different_choices',point[nl:].mean(),boot[:,nl:].mean(1)),('conditional_gain',point[:nl].mean()/point[nl:].mean(),boot[:,:nl].mean(1)/boot[:,nl:].mean(1))]:
   rows.append(dict(scope='disagreement',reference=ref,lambda_value='mean',metric=metric,**record(value,samples)))
  arrays['disagreement|'+ref]=boot
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'bootstrap.npz',**arrays)
 write(out/'report.json',dict(dataset=dataset,complete=True,rows=rows,source_hashes=prov,existing_disagreement_intervals_reproduced=True,slurm_job=os.environ.get('SLURM_JOB_ID')))
 print('COMPLETE',dataset,len(rows),flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
