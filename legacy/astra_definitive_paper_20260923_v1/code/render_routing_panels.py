"""Compact matched-encoder comparisons and honest baseline-completion frontiers.
Only complete five-seed IID or all-family OOD points enter curves. No outcome-based
lambda, method or seed filtering. Graph scenarios selected on source validation.
"""
from pathlib import Path
from collections import defaultdict
import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';CFG=json.loads((ROOT/'config/protocol.json').read_text())
PANELS={'routerbench-0shot':'RouterBench','sprout-sixfam':'SPROUT'}
BLUE='#2166AC';RED='#C5393C'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':190})
records=[]
def append(j,method,lam,v,rep=None):
 records.append(dict(panel=j['panel'],split=j['split'],rep=rep or j['representation'],method=method,lam=float(lam),n=j['n_test'],**{k:v[k] for k in ['utility','reward','cost']}))
for kind in ['knn','prior','clusters','mlp_gpu','prox']:
 for p in (ROOT/'results'/kind).rglob('result.json'):
  j=json.loads(p.read_text())
  for lam,methods in j['results'].items():
   for method,v in methods.items():append(j,method,lam,v)
# New graph precision version supersedes all earlier scenarios only once complete.
for panel in PANELS:
 for i in range(5):
  split=f'iid_{i}';js=None
  for version in ['graph_precision_v2','graph']:
   pp=list((ROOT/'results'/version/panel/split).glob('*/result.json'))
   if len(pp)==3:
    js=sorted([json.loads(p.read_text()) for p in pp],key=lambda j:['performance_first','balance','cost_first'].index(j['scenario']));break
  if js is None:continue
  assert len({j['split_sha256'] for j in js})==1
  for il,lam in enumerate(CFG['lambdas'][panel]):
   best=js[int(np.argmax([j['validation_utilities'][il] for j in js]))]
   v=next(iter(best['results'][str(lam)].values()));append(best,'graph_validated',lam,v,'gte')
cellpath=ROOT/'inputs/sprout_cells_native_openhermes_v1/iid_results.json'
if cellpath.exists():
 for v in json.loads(cellpath.read_text()):append(dict(panel='sprout-sixfam',split=v['split'],representation='metadata',n_test=v['n_test']),'cell_lookup',v['lambda_value'],v)
keys=[(r['panel'],r['split'],r['rep'],r['method'],r['lam']) for r in records];assert len(keys)==len(set(keys))
(ROOT/'reports/routing_plot_records.json').write_text(json.dumps(records,indent=2))
def curve(panel,setting,rep,method):
 expected={f'{setting}_{i}' for i in range(5 if setting=='iid' else (7 if panel.startswith('routerbench') else 6))};vals=[];coverage={}
 for lam in CFG['lambdas'][panel]:
  rr=[r for r in records if r['panel']==panel and r['split'] in expected and r['rep']==rep and r['method']==method and r['lam']==lam]
  coverage[str(lam)]=len(rr)
  if {r['split'] for r in rr}!=expected:continue
  vals.append([lam,*[np.average([r[k] for r in rr],weights=[r['n'] for r in rr]) for k in ['utility','reward','cost']]])
 return np.array(vals).reshape(-1,4),coverage,len(expected)
cards=[]
def save(fig,name,caption):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig);cards.append((name,caption))
methods=[('knn100','kNN-100'),('knn_validated','Validated kNN'),('soft_prior','Soft family prior')]
for setting in ['ood','iid']:
 title='Held-out family' if setting=='ood' else 'In distribution'
 # Same classifier in two spaces; all complete points on the predeclared grid.
 fig,axs=plt.subplots(2,3,figsize=(10.5,5.7),sharex='row',sharey='row');fig.subplots_adjust(top=.85,bottom=.16,wspace=.13,hspace=.42)
 for i,(panel,pname) in enumerate(PANELS.items()):
  for j,(method,label) in enumerate(methods):
   ax=axs[i,j]
   for rep,color in [('gte',BLUE),('reader',RED)]:
    a,_,_=curve(panel,setting,rep,method);assert len(a)==len(CFG['lambdas'][panel])
    ax.plot(a[:,3]*1000,a[:,2],color=color,lw=1.8,marker='o',ms=3,alpha=.95)
   ax.set_title(label,fontsize=10);ax.grid(alpha=.16)
   if j==0:ax.set_ylabel(pname+'\nMean reward')
   if i==1:ax.set_xlabel('USD per 1,000 queries')
 fig.suptitle(title+': same routing rule, different encoder',fontsize=15,y=.98)
 fig.legend([Line2D([],[],color=BLUE,lw=2),Line2D([],[],color=RED,lw=2)],['GTE semantic encoder','Qwen L21'],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.045))
 fig.text(.5,.012,'Higher reward and lower cost are preferable. All predeclared λ values retained.',ha='center',fontsize=8,color='#555555')
 save(fig,f'{setting}_matched_encoder_small_multiples','Matched query cohort and outer splits, identical classifier and tuning rule in both spaces. Source validation may select different k values. Each point is query weighted; IID pools five test splits and OOD all held-out families. Lines connect evaluated policies, not fitted or deployable mixtures. No claim of uniform encoder superiority.')
 # Direct utility advantage at matched lambda, complementary to cost/reward plots.
 fig,axs=plt.subplots(2,3,figsize=(10.5,4.8),sharex='row',sharey='row');fig.subplots_adjust(top=.84,bottom=.16,wspace=.13,hspace=.4)
 for i,(panel,pname) in enumerate(PANELS.items()):
  for j,(method,label) in enumerate(methods):
   ax=axs[i,j];g,_,_=curve(panel,setting,'gte',method);q,_,_=curve(panel,setting,'reader',method);assert np.array_equal(g[:,0],q[:,0]);delta=q[:,1]-g[:,1]
   ax.axhline(0,color=BLUE,lw=1.1);ax.plot(q[:,0],delta,color=RED,marker='o',ms=3,lw=1.6)
   ax.fill_between(q[:,0],0,delta,where=delta>=0,color=RED,alpha=.08,interpolate=True);ax.fill_between(q[:,0],0,delta,where=delta<0,color=BLUE,alpha=.1,interpolate=True)
   ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,10,100,500],['0','1','10','100','500']);ax.grid(alpha=.15);ax.set_title(label,fontsize=10)
   if j==0:ax.set_ylabel(pname+'\nUtility: Qwen − GTE')
   if i==1:ax.set_xlabel('Cost penalty λ')
 fig.suptitle(title+': utility change when replacing GTE with Qwen L21',fontsize=14,y=.98)
 fig.text(.5,.035,'Above zero: Qwen L21 performs better. Below zero: GTE performs better. Same λ and test queries.',ha='center',fontsize=9)
 save(fig,f'{setting}_matched_encoder_utility_differences','Complement to paired reward–cost panels. Utility is reward minus lambda times USD cost. Every predeclared lambda retained, including reversals. Descriptive pooled estimates; no pointwise significance is asserted by this plot.')
# Main fixed baseline set. Additional published baselines are retained in full evidence.
main=[('gte','mlp_utility_validated','MLP (utility)', '#757575','--','s'),('gte','knn100','kNN-100','#A6A6A6',':','o'),('gte','knn_validated','Validated kNN','#444444','--','^'),('gte','avengerspro_validated','Avengers-Pro','#7E668A','-.','D'),('gte','graph_validated','GraphRouter*','#718C69',':','v'),('reader','knn100','Qwen L21 kNN-100',RED,'-','o'),('reader','soft_prior','Qwen L21 soft prior','#E58D32','-','s')]
extra=[('gte','km_prox32_temp20','KM-Prox (published)', '#56A9A6','--','>'),('gte','km_prox32_validated','KM-Prox (validated)','#428080',':','<')]
status={}
for setting in ['iid','ood']:
 for full in [False,True]:
  choices=main+(extra if full else []);fig,axs=plt.subplots(1,2,figsize=(10.5,5.4));fig.subplots_adjust(top=.84,bottom=.3,wspace=.22);present=set()
  for ax,(panel,pname) in zip(axs,PANELS.items()):
   pending=[]
   for rep,method,label,color,ls,marker in choices:
    a,cov,expected=curve(panel,setting,rep,method);key=f'{setting}/{panel}/{rep}/{method}';status[key]=dict(expected_splits=expected,completed_splits_per_lambda=cov,complete_points=len(a),expected_points=len(CFG['lambdas'][panel]))
    if len(a)!=len(CFG['lambdas'][panel]):
     pending.append(label.replace(' (utility)',''));continue
    present.add(label);ax.plot(a[:,3]*1000,a[:,2],color=color,ls=ls,marker=marker,ms=3,lw=2 if rep=='reader' else 1.5,zorder=3 if rep=='reader' else 2)
   ax.set_title(pname);ax.set_xlabel('USD per 1,000 queries');ax.set_ylabel('Mean observed reward');ax.grid(alpha=.16)
   if pending:ax.text(.02,.02,'Pending full coverage:\n'+', '.join(pending),transform=ax.transAxes,fontsize=7,va='bottom',wrap=True,bbox=dict(fc='white',ec='none',alpha=.85))
  handles=[Line2D([],[],color=c,ls=ls,marker=mk,ms=4,lw=1.7) for rep,m,l,c,ls,mk in choices if l in present];labels=[l for rep,m,l,c,ls,mk in choices if l in present]
  fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.05),fontsize=8)
  fig.suptitle(('Held-out-family' if setting=='ood' else 'IID')+' routing: baseline frontier comparison',fontsize=15,y=.98)
  fig.text(.5,.92,'Progress figure · curves require every λ and every evaluation split',ha='center',fontsize=9,color='#555')
  fig.text(.5,.005,'*GraphRouter also uses public task descriptions. OOD inclusion remains undecided.',ha='center',fontsize=7,color='#555')
  save(fig,f'{setting}_'+('full_evidence_' if full else '')+'baseline_frontiers_progress','Published baselines plus the two fixed Qwen L21 candidates. Missing or incomplete methods are explicitly listed, never omitted based on performance. GraphRouter scenario/checkpoint selected solely on source validation; public task descriptions are extra input. MLP regresses utility, an adaptation of the published reward-regression baseline. This is not a final complete-baseline comparison.')
(ROOT/'reports/frontier_completion.json').write_text(json.dumps(status,indent=2))
(OUT/'routing_panels.html').write_text('<!doctype html><meta charset="utf-8"><title>Routing comparisons</title><style>body{font:16px system-ui;max-width:1150px;margin:32px auto;background:#f5f6f8}section{background:white;padding:24px;margin:24px 0}img{width:100%}p{line-height:1.5}</style><h1>Routing comparisons</h1>'+''.join(f'<section><h2>{html.escape(n)}</h2><img src="{n}.png"><p>{html.escape(c)}</p><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a></section>' for n,c in cards))
(OUT/'routing_captions.json').write_text(json.dumps([dict(name=n,caption=c) for n,c in cards],indent=2));print('Rendered',len(cards),'grouped figures')
