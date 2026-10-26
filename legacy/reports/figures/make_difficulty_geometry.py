"""Plot frozen GTE difficulty-geometry headline values from appendix_difficulty.tex."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
labels=['RouterBench · solve fraction','RouterBench · mean reward',
        'LLMRouterBench · solve fraction','LLMRouterBench · mean reward']
y=np.array([3,2,1,0],dtype=float)
colors=['#376ba8','#376ba8','#d88936','#d88936']
markers=['o','D','o','D']
pref=np.array([51.15,51.25,51.49,51.64])
lo=np.array([51.04,51.14,51.27,51.43]);hi=np.array([51.26,51.36,51.69,51.85])
r2=np.array([0.51,0.61,0.84,0.86])
fig,(left,right)=plt.subplots(1,2,figsize=(9.1,2.9),gridspec_kw={'width_ratios':[1.22,1]})
fig.patch.set_facecolor('white')
for ax in (left,right):
 ax.set_ylim(-0.65,3.65)
 ax.spines[['top','right','left']].set_visible(False)
 ax.tick_params(axis='y',length=0)
 ax.set_axisbelow(True)
 ax.grid(axis='x',color='#e6e9ef',linewidth=.75)
left.axvline(50,color='#667085',lw=1.2,ls=(0,(3,3)))
for i in range(4):
 left.errorbar(pref[i],y[i],xerr=[[pref[i]-lo[i]],[hi[i]-pref[i]]],fmt=markers[i],
               markersize=7.8,color=colors[i],ecolor=colors[i],elinewidth=1.8,capsize=3,
               markeredgecolor='white',markeredgewidth=.7,zorder=4)
 left.text(pref[i]+0.23,y[i],f'{pref[i]:.2f}%',va='center',fontsize=9.4,color=colors[i],fontweight='bold')
left.set_xlim(49.8,54.3);left.set_xticks([50,51,52,53,54])
left.set_yticks(y,labels,fontsize=9.1)
left.tick_params(axis='x',labelsize=8.8)
left.set_title('Same-tail cosine preference',loc='left',fontsize=10.5,fontweight='bold',pad=8)
left.set_xlabel('Probability (%) · 50% is chance',fontsize=8.8,labelpad=4)
for i in range(4):
 right.barh(y[i],r2[i],height=.50,color=colors[i],alpha=.83,zorder=3)
 right.text(r2[i]+.07,y[i],f'{r2[i]:.2f}%',va='center',fontsize=9.4,color=colors[i],fontweight='bold')
right.set_xlim(0,4.0);right.set_xticks([0,1,2,3,4]);right.set_yticks(y,['']*4)
right.tick_params(axis='x',labelsize=8.8)
right.set_title('Embedding-only 2-cluster partition',loc='left',fontsize=10.5,fontweight='bold',pad=8)
right.set_xlabel('Additional held-out $R^2$ beyond cell mean (%)',fontsize=8.8,labelpad=4)
fig.subplots_adjust(left=.255,right=.98,bottom=.20,top=.86,wspace=.20)
out=Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=='figures' else Path('/private/tmp')
fig.savefig(out/'difficulty_geometry_headline.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(out/'difficulty_geometry_headline.png',dpi=180,bbox_inches='tight',pad_inches=.03)
