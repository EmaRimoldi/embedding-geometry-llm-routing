"""Paper figure from results/summary/summary.json -> figures/layer_sweep.{pdf,png}.
Rows: RouterBench, SPROUT. Columns:
 (a) held-out utility averaged over the penalty grid vs relative depth (kNN-100 solid, soft prior dashed);
     emitted GTE as horizontal references; the source-selected router as a horizontal dotted line
 (b) share of the 20 neighbours from a family with the query's response format (held-out family removed)
 (c) layers chosen by inner source validation, one per held-out family
Colours follow the paper (GTE blue, Qwen2.5-7B last token orange-red); categorical slots in fixed order.
"""
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_ls import ROOT

C={'gte':'#2a78d6','qwen2.5-7b-instruct|last':'#eb6834','qwen2.5-7b-instruct|mean':'#1baf7a',
   'qwen2.5-coder-1.5b-instruct|last':'#eda100','qwen2.5-coder-7b-instruct|last':'#e87ba4','qwen2.5-coder-14b-instruct|last':'#008300'}
LABEL={'qwen2.5-7b-instruct|last':'Qwen2.5-7B-Instr. last','qwen2.5-7b-instruct|mean':'Qwen2.5-7B-Instr. mean',
       'qwen2.5-coder-1.5b-instruct|last':'Coder-1.5B last','qwen2.5-coder-7b-instruct|last':'Coder-7B last','qwen2.5-coder-14b-instruct|last':'Coder-14B last'}
INK='#3d3d3a';MUTED='#8a8983';GRID='#e6e5df'
PANELS=[('routerbench-0shot','RouterBench'),('sprout-sixfam','SPROUT')]

def style(ax):
 for s in ('top','right'):ax.spines[s].set_visible(False)
 for s in ('left','bottom'):ax.spines[s].set_color(MUTED)
 ax.tick_params(colors=INK,labelsize=8);ax.grid(axis='y',color=GRID,lw=.6);ax.set_axisbelow(True)

def main():
 S=json.loads((ROOT/'results/summary/summary.json').read_text());cur=S['curves'];dg=S['diagnostics']
 fig,axes=plt.subplots(2,3,figsize=(12,6.2),gridspec_kw=dict(width_ratios=[1.35,1,.9]))
 for row,(panel,title) in enumerate(PANELS):
  a,b,c=axes[row]
  for key,col in C.items():
   if key=='gte':continue
   reader,pool=key.split('|')
   for v,ls,mk in (('knn100','-','o'),('soft_prior','--','s')):
    k=f'{panel}|{reader}|{pool}|{v}'
    if k not in cur:continue
    L=np.array(sorted(map(int,cur[k])));y=[np.mean(cur[k][str(l)]) for l in L];x=L/L.max()
    main_series=reader=='qwen2.5-7b-instruct'
    a.plot(x,y,ls=ls,color=col,lw=2 if main_series else 1.2,marker=mk if main_series else None,ms=4,alpha=1 if main_series else .8,label=f'{LABEL[key]} · {"kNN-100" if v=="knn100" else "soft prior"}')
    if key=='qwen2.5-7b-instruct|last' and '21' in cur[k]:a.plot(21/L.max(),np.mean(cur[k]['21']),marker='o',ms=9,mfc='none',mec=INK,mew=1.2,zorder=5)
  for v,ls in (('knn100','-'),('soft_prior','--')):
   k=f'{panel}|gte|emitted|{v}'
   if k in cur:a.axhline(np.mean(cur[k]['None']),color=C['gte'],ls=ls,lw=1.5,label=f'GTE · {"kNN-100" if v=="knn100" else "soft prior"}')
  for v,ls in (('knn100','-'),('soft_prior','--')):
   k=f'{panel}|qwen2.5-7b-instruct|last|{v}'
   if k in S['auto']:a.axhline(np.mean(S['auto'][k]['one_layer_per_fold']),color=INK,ls=':',lw=1.2)
  a.set_title(f'{title}: held-out utility vs depth',fontsize=9.5,color=INK,loc='left');a.set_xlabel('relative depth (layer / L)',fontsize=8.5,color=INK)
  a.set_ylabel('mean utility over λ grid',fontsize=8.5,color=INK);style(a)
  # (b) same-format share
  for key,col in C.items():
   if key=='gte':
    k=f'{panel}|gte|emitted|None'
    if k in dg:b.axhline(dg[k]['same_format_share_k20'],color=col,lw=1.5,label='GTE')
    continue
   reader,pool=key.split('|');pts=sorted((int(kk.split('|')[3]),v['same_format_share_k20']) for kk,v in dg.items() if kk.startswith(f'{panel}|{reader}|{pool}|'))
   if pts:
    L=np.array([p[0] for p in pts]);b.plot(L/L.max(),[p[1] for p in pts],color=col,lw=2 if reader=='qwen2.5-7b-instruct' else 1.2,label=LABEL[key])
  b.set_ylim(0,1.02);b.set_title('neighbours with the same response format',fontsize=9.5,color=INK,loc='left');b.set_xlabel('relative depth',fontsize=8.5,color=INK);style(b)
  # (c) selected layers
  width=.38
  for i,(v,name) in enumerate((('knn100','kNN-100'),('soft_prior','soft prior'))):
   k=f'{panel}|qwen2.5-7b-instruct|last|{v}'
   if k not in S['selected']:continue
   cnt=Counter(ch['layer'] for ch in S['selected'][k].values());ls=sorted(cnt)
   c.bar(np.array(ls)+(i-.5)*width,[cnt[l] for l in ls],width=width,color=C['qwen2.5-7b-instruct|last'],alpha=1 if i==0 else .55,label=name)
  c.axvline(21,color=INK,lw=.8,ls=':');c.set_xlim(-.5,28.5);c.set_title('layer chosen on source families',fontsize=9.5,color=INK,loc='left')
  c.set_xlabel('layer (Qwen2.5-7B-Instr., last token)',fontsize=8.5,color=INK);c.set_ylabel('held-out families',fontsize=8.5,color=INK);style(c)
  c.yaxis.get_major_locator().set_params(integer=True)
 axes[0,0].legend(fontsize=6.5,frameon=False,loc='lower right',ncol=2);axes[0,1].legend(fontsize=6.5,frameon=False,loc='lower right');axes[0,2].legend(fontsize=7,frameon=False)
 fig.text(.01,.005,'Open circle: layer 21. Dotted line in (a): router whose layer is selected per held-out family by leave-one-source-family-out validation.',fontsize=7.5,color=MUTED)
 fig.tight_layout(rect=(0,.02,1,1));out=ROOT/'figures';out.mkdir(exist_ok=True)
 for ext in ('pdf','png'):fig.savefig(out/f'layer_sweep.{ext}',dpi=200)
 print('wrote',out/'layer_sweep.pdf')

if __name__=='__main__':main()
