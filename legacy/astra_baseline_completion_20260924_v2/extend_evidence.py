from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path('__E4_ROOT__/astra_definitive_paper_20260923_v1');OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'astra_baseline_completion_20260924_v1'
sys.path.insert(0,str(ROOT/'code'))
from common import load,sha
old=json.loads((OLD/'audit.json').read_text());records=json.loads((OLD/'records.json').read_text());added=[]
# Verify the saved snapshot was not changed, then independently check new results.
for p,h in old['sources'].items():assert sha(Path(p))==h,p
for panel in ['routerbench-0shot','sprout-sixfam']:
 d=load(panel);ss=json.loads((ROOT/'inputs'/panel/'splits.json').read_text())
 for p in sorted((ROOT/'results/mlp_gpu'/panel).rglob('result.json')):
  if str(p) in old['sources']:continue
  j=json.loads(p.read_text());assert j['split_sha256']==sha(ROOT/'inputs'/panel/'splits.json');te=np.asarray(ss[j['split']]['test']);z=np.load(p.parent/'predictions.npz');ch=z['choice']
  assert np.array_equal(te,z['test_rows']) and np.array_equal(d['query_ids'][te],z['query_ids']);assert np.array_equal(j['selected_architecture'],j['candidate_architectures'][int(np.argmax(j['validation_utilities']))])
  for lam,ms in j['results'].items():
   for m,v in ms.items():
    rr=d['r'][te,ch];cc=d['c'][te,ch];u=rr-float(lam)*cc
    vals={'utility':float(u.mean()),'reward':float(rr.mean()),'cost':float(cc.mean())}
    for k,a in vals.items():assert np.isclose(a,v[k],atol=1e-12,rtol=1e-10)
    records.append(dict(panel=panel,split=j['split'],rep=j['representation'],method=m,lam=float(lam),n=len(te),**vals))
  added.append(dict(file=str(p),sha256=sha(p),predictions_sha256=sha(p.parent/'predictions.npz')))
(OUT/'records.json').write_text(json.dumps(records));(OUT/'incremental_audit.json').write_text(json.dumps({'parent_audit_sha256':sha(OLD/'audit.json'),'unchanged_parent_files':len(old['sources']),'new_files':added},indent=2))
c=json.loads((OLD/'curves.json').read_text());cfg=json.loads((ROOT/'config/protocol.json').read_text());coverage={}
for panel in cfg['panels']:
 for rep in ['gte','reader']:
  for setting in ['iid','ood']:
   expected=5 if setting=='iid' else (7 if panel=='routerbench-0shot' else 6)
   rr=[r for r in records if r['panel']==panel and r['rep']==rep and r['split'].startswith(setting+'_') and r['method']=='mlp_utility_validated'];coverage[f'{panel}/{setting}/{rep}']={'complete':len(rr),'expected':13*expected,'counts_by_lambda':{str(l):sum(r['lam']==l for r in rr) for l in cfg['lambdas'][panel]}}
(OUT/'coverage.json').write_text(json.dumps(coverage,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
(OUT/'figures').mkdir(exist_ok=True);blue='#2166AC';red='#C5393C';panels=[('routerbench-0shot','RouterBench'),('sprout-sixfam','SPROUT')]
methods=[('knn100','kNN-100'),('knn_validated','Validated kNN'),('soft_prior','Soft family prior'),('avengerspro_validated','Avengers-Pro'),('km_prox32_validated','KM-Prox (validated)')]
def save(fig,n):
 for ext in ['pdf','png']:fig.savefig(OUT/'figures'/f'{n}.{ext}',bbox_inches='tight',dpi=180)
 plt.close(fig)
# Two datasets share columns; each method gets its own row for readable labels.
fig,axs=plt.subplots(5,2,figsize=(7.5,10.4),sharex=True);fig.subplots_adjust(top=.93,bottom=.08,hspace=.43,wspace=.32)
for row,(method,name) in enumerate(methods):
 for col,(panel,label) in enumerate(panels):
  ax=axs[row,col];g=np.asarray(c[f'{panel}/ood/gte/{method}']);q=np.asarray(c[f'{panel}/ood/reader/{method}']);delta=q[:,1]-g[:,1]
  ax.axhline(0,color=blue,lw=1);ax.plot(g[:,0],delta,color=red,marker='o',ms=3,lw=1.4);ax.fill_between(g[:,0],0,delta,where=delta>0,color=red,alpha=.09);ax.fill_between(g[:,0],0,delta,where=delta<0,color=blue,alpha=.1)
  ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,10,100,500],['0','1','10','100','500']);ax.grid(alpha=.13);ax.set_title(label+' | '+name,fontsize=10)
  if col==0:ax.set_ylabel('L21 minus GTE')
  if row==4:ax.set_xlabel('Cost penalty λ')
fig.suptitle('Held-out families: utility change at the same cost penalty',fontsize=12);save(fig,'ood_matched_utility_all')
fig,axs=plt.subplots(2,4,figsize=(11,5.7),sharey='row');fig.subplots_adjust(top=.9,bottom=.2,wspace=.25,hspace=.42)
counts=[]
for row,(panel,label) in enumerate(panels):
 for col,(setting,method,name) in enumerate([('iid','km_prox32_temp20','IID | KM-Prox (fixed)'),('ood','km_prox32_temp20','OOD | KM-Prox (fixed)'),('iid','cluster_utility_validated','IID | Utility clusters'),('ood','cluster_utility_validated','OOD | Utility clusters')]):
  ax=axs[row,col];aa=[]
  for rep,color in [('gte',blue),('reader',red)]:
   a=np.asarray(c[f'{panel}/{setting}/{rep}/{method}']);aa.append(a);ax.plot(a[:,3]*1000,a[:,2],color=color,lw=1.6,marker='o',ms=2.5)
  delta=aa[1][:,1]-aa[0][:,1];counts.append(dict(panel=panel,setting=setting,method=method,wins=int((delta>1e-12).sum()),losses=int((delta<-1e-12).sum())))
  ax.set_title(name,fontsize=9);ax.grid(alpha=.15)
  if col==0:ax.set_ylabel(label+'\nMean reward')
  if row==1:ax.set_xlabel('USD / 1,000 queries',fontsize=9)
fig.legend([Line2D([],[],color=blue,lw=2),Line2D([],[],color=red,lw=2)],['GTE','Qwen L21'],ncol=2,loc='lower center',bbox_to_anchor=(.5,.035),frameon=False)
fig.suptitle('Additional matched comparisons: fixed KM-Prox and utility-cluster control',fontsize=12);save(fig,'additional_matched_controls')
(OUT/'control_summary.json').write_text(json.dumps(counts,indent=2));print(json.dumps({'new_outputs':len(added),'coverage':coverage,'controls':counts},indent=2))
