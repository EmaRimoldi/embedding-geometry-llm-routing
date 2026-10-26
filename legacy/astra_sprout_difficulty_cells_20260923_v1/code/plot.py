"""Plot all fixed grouping resolutions, with unchanged primary cohort."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
d=json.loads((root/'reports/summary.json').read_text())
assert json.loads((root/'reports/AUDIT.json').read_text())['passed']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7.1,3.1),gridspec_kw={'width_ratios':[1.12,1]})
colors=['#6A43A9','#177C98','#CB7731']; names=['Solve fraction','Mean reward','Reward profile']
keys=['family_k10','approved_cell_k10','cell_clusters2_k10','cell_clusters4_k10','cell_clusters8_k10']
for i,(name,color) in enumerate(zip(names,colors)):
 vals=[d['locality_percent_reduction'][key] for key in keys]
 p=np.array([v['estimate'][i] for v in vals]);ci=np.array([v['ci95'][i] for v in vals])
 axes[0].errorbar(np.arange(5),p,yerr=np.array([p-ci[:,0],ci[:,1]-p]),color=color,
                  marker='o',markersize=4,lw=1.6,capsize=2,label=name)
axes[0].set_xticks(range(5),['Family','Approved\ncell','Cell +\n2 groups','Cell +\n4 groups','Cell +\n8 groups'])
axes[0].set_ylabel('Local outcome agreement gain (%)')
axes[0].set_title('Finer groups reduce, but retain, locality',loc='left',fontweight='bold')
axes[0].set_ylim(bottom=0);axes[0].grid(axis='y',alpha=.18);axes[0].legend(frameon=False,fontsize=7)
terms=['family_composition','cell_beyond_family','clusters_beyond_cell','remaining_selection']
labels=['Family composition','Additional cell composition','Additional fine-group composition','Remaining neighbour selection']
palette=['#BBC9DB','#82A8BF','#78B7A5','#6A43A9'];left=np.zeros(3)
for term,label,color in zip(terms,labels,palette):
 vals=np.array(d['donor_composition']['k10']['terms'][term]['share_percent']['estimate'][:3])
 assert (vals>=0).all(), 'Use a signed plot if any decomposition term is negative'
 axes[1].barh(np.arange(3),vals,left=left,color=color,height=.58,label=label)
 left+=vals
assert np.allclose(left,100)
axes[1].set_yticks(range(3),names);axes[1].invert_yaxis();axes[1].set_xlim(0,100)
axes[1].set_xlabel('Unrestricted neighbour advantage (%)')
axes[1].set_title('Composition explains only part',loc='left',fontweight='bold')
axes[1].legend(frameon=False,fontsize=6.5,loc='upper left',bbox_to_anchor=(-.03,-.40),borderaxespad=0)
fig.subplots_adjust(left=.075,right=.985,top=.88,bottom=.34,wspace=.52)
fig.savefig(root/'reports/difficulty_grouping_sprout.pdf',bbox_inches='tight')
fig.savefig(root/'reports/difficulty_grouping_sprout.png',bbox_inches='tight',dpi=200)
