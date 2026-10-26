"""Rebuild the three-stage routing schematic for paper.tex."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(11.2,2.25))
fig.patch.set_facecolor('white')
ax.set_xlim(0,11.2);ax.set_ylim(0,2.25);ax.axis('off')

cards=[
 (0.12,0.32,2.62,1.57,'QUERY','Query text',r'$x$','#eef2f7','#344054'),
 (4.26,0.32,2.62,1.57,'EMBED','Embedding',r'$z=\mathcal{F}(x)$','#eee6fa','#7353a6'),
 (8.40,0.32,2.68,1.57,'CLASSIFY','Model selection',r'$m^*=g(z)$','#fce5e2','#b54d4c'),
]
for x,y,w,h,stage,title,detail,fill,accent in cards:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.055,rounding_size=0.16',
                             linewidth=1.4,edgecolor=accent,facecolor=fill))
 ax.text(x+0.18,y+h-0.20,stage,fontsize=13.0,fontweight='bold',color=accent,va='top')
 ax.text(x+w/2,y+0.78,title,fontsize=20.0,fontweight='semibold',color='#182230',ha='center',va='center')
 ax.text(x+w/2,y+0.32,detail,fontsize=17.0,color='#344054',ha='center',va='center')
for a,b in zip(cards,cards[1:]):
 ax.add_patch(FancyArrowPatch((a[0]+a[2]+0.10,1.05),(b[0]-0.12,1.05),
                              arrowstyle='-|>',mutation_scale=20,lw=2.0,color='#596579'))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
out_dir=Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=='figures' else Path('/private/tmp')
fig.savefig(out_dir/'routing_pipeline.pdf',bbox_inches='tight',pad_inches=0.02)
fig.savefig(out_dir/'routing_pipeline.png',dpi=180,bbox_inches='tight',pad_inches=0.02)
