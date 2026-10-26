from pathlib import Path
import json,numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'astra_definitive_paper_20260923_v1'
cfg=json.loads((ROOT/'config/protocol.json').read_text())
records=json.loads((OUT/'records.json').read_text())
audits=json.loads((OUT/'audit.json').read_text())['audits']
coverage={};curves={}
def curve(panel,setting,rep,method):
 key=f'{panel}/{setting}/{rep}/{method}';expected={f'{setting}_{i}' for i in range(5 if setting=='iid' else (7 if panel.startswith('router') else 6))};a=[];counts=[]
 for lam in cfg['lambdas'][panel]:
  rr=[x for x in records if x['panel']==panel and x['split'] in expected and x['rep']==rep and x['method']==method and x['lam']==lam];counts.append(len(rr))
  if {x['split'] for x in rr}==expected:a.append([lam,*[np.average([x[k] for x in rr],weights=[x['n'] for x in rr]) for k in ['utility','reward','cost']]])
 coverage[key]=dict(completed=sum(counts),expected=13*len(expected),per_lambda=counts)
 if len(a)==13:curves[key]=a;return np.asarray(a)
 return None
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
PANELS={'routerbench-0shot':'RouterBench','sprout-sixfam':'SPROUT'};blue='#2166AC';red='#C5393C'
methods=[('knn100','kNN-100'),('knn_validated','Validated kNN'),('soft_prior','Soft family prior'),('avengerspro_validated','Avengers-Pro'),('km_prox32_validated','KM-Prox (validated)')]
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight',dpi=180)
 plt.close(fig)
summary={}
for setting in ['ood','iid']:
 fig,axs=plt.subplots(2,5,figsize=(12.7,5.1),sharey='row');fig.subplots_adjust(top=.87,bottom=.19,wspace=.2,hspace=.4)
 for i,(panel,label) in enumerate(PANELS.items()):
  for k,(method,title) in enumerate(methods):
   ax=axs[i,k];aa=[]
   for rep,col in [('gte',blue),('reader',red)]:
    a=curve(panel,setting,rep,method);assert a is not None;aa.append(a);ax.plot(a[:,3]*1000,a[:,2],color=col,lw=1.7,marker='o',ms=2.5)
   delta=aa[1][:,1]-aa[0][:,1];summary[f'{panel}/{setting}/{method}']={'positive':int((delta>0).sum()),'negative':int((delta<0).sum()),'deltas':delta.tolist()}
   ax.set_title(title,fontsize=9);ax.grid(alpha=.14);ax.tick_params(labelsize=7)
   if k==0:ax.set_ylabel(label+'\nMean reward')
   if i==1:ax.set_xlabel('USD / 1,000 queries',fontsize=8)
 fig.legend([Line2D([],[],color=blue,lw=2),Line2D([],[],color=red,lw=2)],['GTE','Qwen L21'],ncol=2,loc='lower center',bbox_to_anchor=(.5,.01),frameon=False)
 fig.suptitle(('Held-out families' if setting=='ood' else 'In distribution')+': matched routers, different representations',fontsize=13)
 save(fig,setting+'_matched_five')
 main=[('gte','best_single_train','Best single (train)','#171717','-.','x'),('gte','mlp_utility_validated','MLP (utility)','#777777','--','s'),('gte','knn100','kNN-100','#aaaaaa',':','o'),('gte','knn_validated','Validated kNN','#333333','--','^'),('gte','avengerspro_validated','Avengers-Pro','#80648b','-.','D')]
 if setting=='iid':main.append(('gte','graph_validated','GraphRouter*','#63895b',':','v'))
 main += [('gte','km_prox32_validated','KM-Prox (validated)','#36988c','--','>'),('reader','knn100','L21 kNN-100',red,'-','o'),('reader','soft_prior','L21 soft prior','#db8a27','-','s')]
 for full in [False,True]:
  entries=main+([('gte','km_prox32_temp20','KM-Prox (fixed)','#3175a1',':','<'),('gte','soft_prior','GTE soft prior','#ae748e','--','x'),('gte','cluster_utility_validated','Utility clusters (control)','#9c9c29',':','+')] if full else [])
  fig,axs=plt.subplots(1,2,figsize=(10.5,5));fig.subplots_adjust(top=.89,bottom=.27,wspace=.24);present=set()
  for ax,(panel,label) in zip(axs,PANELS.items()):
   pending=[]
   for rep,m,name,col,ls,mk in entries:
    a=curve(panel,setting,rep,m)
    if a is None:pending.append(name);continue
    present.add(name);ax.plot(a[:,3]*1000,a[:,2],color=col,ls=ls,marker=mk,ms=3,lw=1.8 if rep=='reader' else 1.3)
   ax.set(title=label,xlabel='USD per 1,000 queries',ylabel='Mean reward');ax.grid(alpha=.15)
   if pending:ax.text(.02,.02,'Pending: '+', '.join(pending),transform=ax.transAxes,fontsize=8,bbox=dict(fc='white',ec='none',alpha=.9))
  fig.legend([Line2D([],[],color=c,ls=ls,marker=mk,ms=3) for rep,m,n,c,ls,mk in entries if n in present],[n for rep,m,n,c,ls,mk in entries if n in present],loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False,fontsize=8)
  fig.suptitle(('Held-out-family' if setting=='ood' else 'In-distribution')+' baseline comparison',fontsize=13)
  save(fig,setting+('_full' if full else '')+'_baselines')
# Record all methods and MLP coverage, even if not plotted.
for panel in PANELS:
 for setting in ['iid','ood']:
  for rep,m in sorted({(x['rep'],x['method']) for x in records if x['panel']==panel}):curve(panel,setting,rep,m)
(OUT/'coverage.json').write_text(json.dumps(coverage,indent=2));(OUT/'curves.json').write_text(json.dumps(curves,indent=2));(OUT/'matched_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'audit':audits,'matched_counts':{k:(v['positive'],v['negative']) for k,v in summary.items()},'mlp':{k:v for k,v in coverage.items() if 'mlp' in k}},indent=2))
