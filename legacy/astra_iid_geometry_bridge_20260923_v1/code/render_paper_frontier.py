"""Snapshot only complete existing IID curves; no training or experiment edits."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
E4=Path('__E4_ROOT__')
ROOT=E4/'astra_iid_geometry_bridge_20260923_v1'; D=E4/'astra_definitive_paper_20260923_v1'
OUT=ROOT/'paper_revision'; OUT.mkdir(exist_ok=True)
choices=[('cell_mean','Cell lookup','#182B49','-','s'),('knn100','kNN-100','#D66A35','-','o'),('knn_validated','Validated kNN','#337EA0','--','^'),('mlp_utility_validated','MLP','#777777','--','v'),('avengerspro_validated','Avengers-Pro','#98729C','-.','D'),('graph_validated','GraphRouter*','#559277',':','P')]
config=json.loads((D/'config/protocol.json').read_text()); records=[]; sources={}; missing={}
def read(p):
 sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
for panel in ['routerbench-0shot','sprout-sixfam']:
 for seed in range(5):
  sp=f'iid_{seed}'
  j=read(ROOT/'results'/panel/sp/'result.json')
  for v in j['results']:
   if v['method']=='cell_mean': records.append(dict(panel=panel,split=sp,method='cell_mean',lam=v['lambda_value'],n=j['n_test'],**{k:v[k] for k in ['utility','reward','cost']}))
  for kind in ['knn','clusters','mlp_gpu']:
   for p in (D/'results'/kind/panel/sp/'gte').rglob('result.json'):
    j=read(p)
    for lam,methods in j['results'].items():
     for method,v in methods.items():
      if method in [a[0] for a in choices]:records.append(dict(panel=panel,split=sp,method=method,lam=float(lam),n=j['n_test'],**{k:v[k] for k in ['utility','reward','cost']}))
  pp=list((D/'results/graph_precision_v2'/panel/sp).glob('*/result.json'))
  if len(pp)==3:
   js=sorted([read(p) for p in pp],key=lambda j:['performance_first','balance','cost_first'].index(j['scenario']))
   assert len({j['split_sha256'] for j in js})==1
   for i,lam in enumerate(config['lambdas'][panel]):
    j=js[int(np.argmax([q['validation_utilities'][i] for q in js]))]
    v=next(iter(j['results'][str(lam)].values()))
    records.append(dict(panel=panel,split=sp,method='graph_validated',lam=lam,n=j['n_test'],**{k:v[k] for k in ['utility','reward','cost']}))
assert len(records)==len({(v['panel'],v['split'],v['method'],v['lam']) for v in records})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(9.6,3.7));fig.subplots_adjust(bottom=.25,top=.91,wspace=.22)
present=set();coverage={}
for ax,panel,label in zip(axes,['routerbench-0shot','sprout-sixfam'],['RouterBench','SPROUT']):
 missing[panel]=[]
 for method,name,color,ls,marker in choices:
  vals=[];cov=[]
  for lam in config['lambdas'][panel]:
   rr=[v for v in records if v['panel']==panel and v['method']==method and v['lam']==lam]
   cov.append(len(rr))
   if {v['split'] for v in rr}!={f'iid_{i}' for i in range(5)}:continue
   vals.append([np.average([v[k] for v in rr],weights=[v['n'] for v in rr]) for k in ['cost','reward']])
  coverage[panel+'/'+method]=cov
  if len(vals)!=len(config['lambdas'][panel]):missing[panel].append(name);continue
  present.add(method);a=np.array(vals)
  ax.plot(a[:,0]*1000,a[:,1],label=name,color=color,ls=ls,marker=marker,ms=3,lw=1.7,zorder=4 if method=='cell_mean' else 3)
 ax.set_title(label,fontsize=12);ax.set_xlabel('USD per 1,000 queries');ax.set_ylabel('Mean reward');ax.grid(alpha=.14)
 if missing[panel]:ax.text(.02,.025,'Incomplete: '+', '.join(missing[panel]),transform=ax.transAxes,fontsize=7,color='#666')
handles=[Line2D([],[],color=c,ls=ls,marker=mk,ms=4,lw=1.7) for m,l,c,ls,mk in choices if m in present]
fig.legend(handles,[l for m,l,c,ls,mk in choices if m in present],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005),fontsize=8)
for ext in ['pdf','png']:fig.savefig(OUT/f'sec4_iid_frontiers_cells.{ext}',bbox_inches='tight',dpi=180)
(OUT/'frontier_snapshot.json').write_text(json.dumps(dict(records=records,input_sha256=sources,coverage=coverage,missing=missing),indent=2))
print(json.dumps(dict(coverage=coverage,missing=missing),indent=2))
