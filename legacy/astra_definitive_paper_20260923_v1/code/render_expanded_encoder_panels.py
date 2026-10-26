"""Six preselected matched classifiers, retaining every completed comparison.
Run render_routing_panels.py first to refresh validated plot records.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures'
records=json.loads((ROOT/'reports/routing_plot_records.json').read_text())
cfg=json.loads((ROOT/'config/protocol.json').read_text())
methods=[('knn100','kNN-100'),('knn_validated','Validated kNN'),('soft_prior','Soft family prior'),('mlp_utility_validated','MLP (utility regression)'),('avengerspro_validated','Avengers-Pro'),('km_prox32_validated','KM-Prox (validated)')]
blue='#2166AC';red='#C5393C';coverage={};complete=True
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':180})
fig,axs=plt.subplots(4,3,figsize=(10.5,11.2));fig.subplots_adjust(top=.9,bottom=.12,wspace=.17,hspace=.65)
for ip,(panel,pname,nfold) in enumerate([('routerbench-0shot','RouterBench',7),('sprout-sixfam','SPROUT',6)]):
 group=[];allarrays=[]
 for im,(method,label) in enumerate(methods):
  ax=axs[ip*2+im//3,im%3];group.append(ax);arrays={};count={}
  for rep in ['gte','reader']:
   vals=[];counts=[]
   for lam in cfg['lambdas'][panel]:
    rr=[r for r in records if r['panel']==panel and r['split'] in {f'ood_{f}' for f in range(nfold)} and r['rep']==rep and r['method']==method and r['lam']==lam]
    counts.append(len(rr));assert len({r['split'] for r in rr})==len(rr)
    if len(rr)==nfold:vals.append([lam,*[np.average([r[k] for r in rr],weights=[r['n'] for r in rr]) for k in ['utility','reward','cost']]])
   arrays[rep]=np.array(vals).reshape(-1,4);count[rep]=counts
  ready=all(len(a)==len(cfg['lambdas'][panel]) for a in arrays.values());coverage[f'{panel}/{method}']=dict(complete=ready,fold_counts=count,expected_folds=nfold)
  ax.set_title(label,fontsize=10);ax.grid(alpha=.15)
  if ready:
   for rep,color in [('gte',blue),('reader',red)]:
    a=arrays[rep];ax.plot(a[:,3]*1000,a[:,2],color=color,lw=1.8,marker='o',ms=2.8);allarrays.append(a)
  else:
   complete=False
   text='Matched runs in progress\n'+'\n'.join(f'{name}: {sum(count[rep])}/{nfold*len(cfg["lambdas"][panel])} fold–λ results' for rep,name in [('gte','GTE'),('reader','Qwen')])
   ax.text(.5,.5,text,transform=ax.transAxes,ha='center',va='center',fontsize=8,color='#555',bbox=dict(fc='white',ec='none',alpha=.9))
  if im%3==0:ax.set_ylabel('Mean observed reward')
  if im//3==1:ax.set_xlabel('USD per 1,000 queries')
 if allarrays:
  a=np.concatenate(allarrays);xmin,xmax=a[:,3].min()*1000,a[:,3].max()*1000;ymin,ymax=a[:,2].min(),a[:,2].max();dx=max(xmax-xmin,.01)*.05;dy=max(ymax-ymin,.01)*.07
  for ax in group:ax.set_xlim(max(-.1,xmin-dx),xmax+dx);ax.set_ylim(ymin-dy,ymax+dy)
 for im,ax in enumerate(group):
  if im%3:ax.tick_params(labelleft=False)
 y=axs[ip*2,0].get_position().y1+.035
 fig.text(.065,y,pname,fontsize=13,fontweight='bold')
fig.suptitle('Held-out family: same classifiers, two encoders',fontsize=16,y=.99)
fig.legend([Line2D([],[],color=blue,lw=2),Line2D([],[],color=red,lw=2)],['GTE semantic encoder','Qwen L21'],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.029))
fig.text(.5,.009,'Matched queries, folds, λ grids and tuning rules · '+('All runs complete' if complete else 'Progress layout: pending panels are not evidence'),ha='center',fontsize=8,color='#555')
name='ood_matched_encoder_six_methods'+('' if complete else '_progress')
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
plt.close(fig)
(ROOT/'reports/expanded_encoder_panel_status.json').write_text(json.dumps(dict(methods=methods,complete=complete,coverage=coverage,figure=name),indent=2))
print(name)
