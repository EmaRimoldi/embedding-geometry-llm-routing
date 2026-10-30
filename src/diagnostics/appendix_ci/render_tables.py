"""Render appendix tables from audited bootstrap outputs, without editing manuscript prose."""
import json,sys
from pathlib import Path
from ci_common import ROOT,OUT,write
PANELS=['routerbench-0shot','sprout-sixfam'];DS=['routerbench','sprout'];NAMES=['RouterBench','SPROUT']
METHODS=[('Avengers-Pro','avengerspro_validated'),('Best single model','best_single_train'),('Family lookup','family_lookup'),('Family-cell lookup','family_topic_lookup'),('KM-Prox','km_prox32_temp20'),('KM-Prox (validated)','km_prox32_validated'),('MLP','mlp'),('Realized oracle','realized_oracle'),('Ridge','ridge'),('Family Adaptation','family_adaptation'),('Validated kNN','knn_validated'),('kNN-100','knn100')]
ROUTE={(ds,reg):json.loads((OUT/'routing'/panel/reg/'report.json').read_text()) for ds,panel in zip(DS,PANELS) for reg in ['iid','ood']}
GEO={ds:json.loads((OUT/'geometry'/ds/'report.json').read_text()) for ds in DS}
macros={};audit=[]
def find(rows,**kw):
 a=[v for v in rows if all(str(v.get(k))==str(x) for k,x in kw.items())];assert len(a)==1,(kw,len(a));return a[0]
def route(ds,method,rep='gte',reg='ood',lam='mean',scope='base'):
 return find(ROUTE[ds,reg]['rows'],scope=scope,representation=rep,method=method,lambda_value=str(lam))
def geo(ds,scope,metric,**kw):return find(GEO[ds]['rows'],scope=scope,metric=metric,**kw)
def fnum(x,d,sign=False):
 t=f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
 if float(t)==0:t=f'{0:.{d}f}'
 return t
def cell(r,d=4,pct=False,sign=False,bold=False,point=None):
 assert r['low']<=r['high']
 scale=100 if pct else 1
 a=point if point is not None else fnum(r['mean']*scale,d,sign)+(r'\%' if pct else '')
 b=fnum(r['low']*scale,d,sign);c=fnum(r['high']*scale,d,sign)
 if bold:a=r'\textbf{'+a+'}'
 audit.append(r)
 return r'\resultci{'+a+'}{'+b+'}{'+c+'}'
def table(label,caption,cols,header,rows,sep=4):
 return '\n'.join([r'\begin{table}[H]',r'\centering\small',r'\setlength{\tabcolsep}{'+str(sep)+'pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{@{}'+cols+'@{}}',r'\toprule',header,r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def row(v):return ' & '.join(v)+r' \\[2pt]'
rr=[]
for ds,name in zip(DS,NAMES):
 for group,label in [('family','Family'),('cell','Cell')]:
  vals=[cell(geo(ds,'deployability','accuracy',group=group),2,True)]
  vals += [cell(geo(ds,'deployability',m,group=group),5) for m in ['known_utility','predicted_utility','difference']]
  rr.append(row([name,label]+vals))
macros['CIDeployability']=table('tab:lookup-deployability',r'Lookup with predicted labels, with 95\% intervals. Utilities average five splits and thirteen penalties; $\Delta$ is predicted-label minus known-label utility.','llrrrr',row(['Benchmark','Label','Accuracy',r'\shortstack{Known-label\\utility}',r'\shortstack{Predicted-label\\utility}',r'$\Delta$']),rr,5)
order=[('Best single model','best_single_train'),('kNN-100','knn100'),('Validated kNN','knn_validated'),('Avengers-Pro','avengerspro_validated'),('KM-Prox (fixed)','km_prox32_temp20'),('KM-Prox (validated)','km_prox32_validated'),('Ridge','ridge'),('MLP','mlp'),('Family Adaptation','family_adaptation')]
rr=[r'\multicolumn{3}{@{}l}{\textit{GTE embeddings}} \\']
rr += [row([name]+[cell(route(ds,m)) for ds in DS]) for name,m in order]
rr += [r'\midrule',r'\multicolumn{3}{@{}l}{\textit{Qwen L21 embeddings}} \\']
rr += [row([name]+[cell(route(ds,m,'reader'),bold=m=='family_adaptation') for ds in DS]) for name,m in [('kNN-100','knn100'),('Family Adaptation','family_adaptation')]]
macros['CIOodMean']=table('tab:ood-mean','Mean held-out utility across thirteen cost penalties, with 95\\% intervals. Bold marks the highest mean among the listed methods.','lrr',row(['Method','RouterBench','SPROUT']),rr,8)
# Direct counterpart of the main NVIDIA comparison, with paired differences.
rr=[]
for name,m in [('L21 kNN-100','knn100'),('L21 Family Adaptation','family_adaptation')]:rr.append(row([name]+[cell(route(ds,m,'reader')) for ds in DS]))
rr.append(row(['NVIDIA utility adaptation']+[cell(route(ds,'nvidia','utility',scope='nvidia')) for ds in DS]))
macros['CINvidiaMean']=table('tab:nvidia-mean-ci','Mean held-out utility across thirteen penalties, with 95\\% intervals, for the comparison in the main text.','lrr',row(['Method','RouterBench','SPROUT']),rr,8)
# Lookup disagreements. Their existing 1,000-replicate intervals are kept exactly.
def dis(ds,ref,metric,lam='mean'):
 return geo(ds,'disagreement',metric,reference=ref,lambda_value=str(lam))
rr=[]
for ds,name in zip(DS,NAMES):
 for ref,label in [('family_lookup','Family'),('family_topic_lookup','Family-cell')]:
  rr.append(row([name,label,cell(dis(ds,ref,'different_choices'),2,True),cell(dis(ds,ref,'conditional_gain'),4,sign=True),cell(dis(ds,ref,'overall_gain'),4,sign=True)]))
macros['CIDisagreementSummary']=table('tab:iid-disagreement-summary','GTE kNN-100 compared with group lookup across penalties, with 95\\% paired intervals. Positive gains favour kNN.','llrrr',row(['Dataset','Lookup',r'\shortstack{Different\\choices}',r'\shortstack{Gain on\\disagreements}','Overall gain']),rr,7)
rr=[]
for ds,name,panel in zip(DS,NAMES,PANELS):
 rr.append(r'\multicolumn{5}{l}{\textbf{'+name+r'}} \\')
 lams=json.loads((ROOT/'configs/baselines.json').read_text())['lambdas'][panel]
 for lam in lams:
  k=f'{lam:g}'
  rr.append(row([k,cell(dis(ds,'family_lookup','overall_gain',k),4,sign=True),cell(dis(ds,'family_topic_lookup','different_choices',k),2,True),cell(dis(ds,'family_topic_lookup','conditional_gain',k),4,sign=True),cell(dis(ds,'family_topic_lookup','overall_gain',k),4,sign=True)]))
header=r'& \shortstack{Against family lookup\\Overall gain} & \multicolumn{3}{c}{Against family-cell lookup} \\'+'\n'+r'\cmidrule(lr){3-5}'+'\n'+row([r'$\lambda$','',r'\shortstack{Different\\choices}',r'\shortstack{Gain on\\disagreements}','Overall gain'])
macros['CIDisagreementPenalties']=table('tab:iid-disagreement-penalties','kNN-100 utility gains at every penalty, with 95\\% paired intervals.','lrrrr',header,rr,9)
# Full utility tables: split each 13-point grid into 5/4/4 penalties to retain readable intervals.
full=[];pointmap={}
for ds,name,panel in zip(DS,NAMES,PANELS):
 for reg in ['iid','ood']:
  lams=json.loads((ROOT/'configs/baselines.json').read_text())['lambdas'][panel]
  for part,selected in enumerate([lams[:5],lams[5:9],lams[9:]]):
   rr=[]
   for rep,repname in [('gte','GTE'),('reader','Qwen L21')]:
    for method,m in METHODS:
     if reg=='ood' and m in ['family_lookup','family_topic_lookup']:continue
     if rep=='reader' and m in ['family_lookup','family_topic_lookup','best_single_train','realized_oracle']:continue
     label=repname+' / '+method;vals=[]
     for lam in [*[f'{v:g}' for v in selected],'mean']:
      r=route(ds,m,rep,reg,lam);vals.append(cell(r));pointmap['|'.join([ds,reg,label,lam])]=f"{r['mean']:.4f}"
     rr.append(row([label]+vals))
   label=f'tab:ci-{ds}-{reg}-{part+1}'
   caption=name+(' in-distribution' if reg=='iid' else ' out-of-distribution')+r' utility with 95\% intervals, part '+str(part+1)+r' of 3. Mean averages all thirteen penalties.'
   full.append(table(label,caption,'l'+'r'*(len(selected)+1),row(['Method / $\\lambda$']+[f'{v:g}' for v in selected]+['Mean']),rr,3))
macros={k:v for k,v in macros.items() if k in ['CIDeployability','CIOodMean','CINvidiaMean','CIDisagreementSummary','CIDisagreementPenalties']}
dest=OUT/'overleaf';dest.mkdir(parents=True,exist_ok=True)
definitions=r'\providecommand{\resultci}[3]{\shortstack[r]{#1\\[-1pt]{\scriptsize[#2,\,#3]}}}'+'\n'
definitions += '\n\n'.join('\\newcommand{\\'+name+'}{%\n'+body+'\n}' for name,body in macros.items())+'\n'
(dest/'aistats_ci_tables.tex').write_text(definitions)
(dest/'aistats_tables.tex').write_text('\n\n'.join(full)+'\n')
write(dest/'pointmap.json',pointmap);write(dest/'macros.json',macros)
write(dest/'audit.json',dict(cells_with_intervals=len(audit),tables=len(full)+len(macros),macro_names=list(macros),full_tables=len(full),all_intervals_ordered=True))
print('GENERATED',len(full)+len(macros),'tables',len(audit),'interval cells')
