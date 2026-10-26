"""Render the approved SharedTrunkNet appendix from audited bootstrap output."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/diagnostics/appendix_confidence_intervals/sharedtrunknet'
a=json.loads((OUT/'report.json').read_text())
assert a['complete'] and a['reproduced_existing_intervals']
datasets=['routerbench-0shot','sprout-sixfam']
names={'fixed_l21':'SharedTrunkNet: fixed L21','selected_layer':'SharedTrunkNet: common selected layer','per_model':'SharedTrunkNet: per-model selected layers','knn100':'L21 kNN-100','family_adaptation':'L21 Family Adaptation'}
def ci(v,signed=False):
 f=(lambda x:f'{x:+.4f}') if signed else (lambda x:f'{x:.4f}')
 return r'\resultci{'+f(v['mean'])+'}{'+f(v['low'])+'}{'+f(v['high'])+'}'
def start(caption,label,columns,header):
 return '\n'.join([r'\begin{table}[H]',r'\centering\small',r'\setlength{\tabcolsep}{6pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{@{}'+columns+'@{}}',r'\toprule',header+r' \\',r'\midrule'])+'\n'
def end():return '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
parts=[r"""\subsection{SharedTrunkNet implementation and robustness}
\label{app:sharedtrunknet}

\paragraph{Comparison and inputs.}
We adapt the released SharedTrunkNet pipeline of \citet{varshney2026prefill} to predict utility $u_\lambda(x,m)=r(x,m)-\lambda c(x,m)$.
All compared methods use the same raw prompts, frozen Qwen2.5-7B-Instruct checkpoint, candidate models, recorded outcomes, outer splits and penalty grids.
kNN-100 and the Family Adaptation use unit-normalized L21 states as in the main text.
SharedTrunkNet retains its own feature preprocessing and network.
The comparison evaluates these routing pipelines with a common encoder and utility objective; it does not reproduce the original paper's model pools or absolute scores.

\paragraph{Features and layer selection.}
For each layer $\ell\in\{15,\ldots,28\}$, we use the last input token's hidden state.
We standardize its coordinates and fit an unwhitened PCA with 100 components on the outer training queries only.
These transforms remain fixed during feature selection.
For each candidate model and penalty, a Ridge probe with fixed $\alpha=1$ and an intercept predicts utility in five shuffled folds of the outer training queries.
We apply the released ternary layer-search rule to negative mean validation MSE.
This selects a layer separately for every model and penalty; it is a heuristic search, not an exhaustive maximization.
The selected 100-dimensional blocks are concatenated in candidate-model order, giving $100M$ inputs for $M$ candidate models.
Blocks are repeated when models select the same layer.
The five folds split training queries, not whole families; the outer test family is excluded from all preprocessing, selection and fitting.

\paragraph{Network and utility objective.}
We use the released shared network: hidden widths 256 and 128, ReLU activations, dropout 0.3 and 0.2, and $M$ linear outputs.
Training uses Adam with learning rate $10^{-3}$, weight decay $10^{-4}$, batches of 512, at most 150 epochs and early stopping with patience 15 and improvement threshold $10^{-4}$.
Each of ten independently initialized networks reserves a random 15\% of the outer training queries for early stopping; the five networks with lowest validation MSE are retained.
Their raw outputs are averaged, and the model with the largest predicted utility is selected.
We train a separate ensemble for each penalty.
The utility adaptation replaces logistic-regression/AUC feature selection with Ridge/MSE, and binary cross-entropy with utility MSE in the network.
Outputs are not passed through a sigmoid, and cost is not subtracted again.
The architecture, optimizer, stopping rule and ten-to-five ensemble procedure are retained.
All 23 outer evaluations and 299 penalty evaluations completed; saved predictions reproduce their reported utilities.

\paragraph{Alternative selection procedures.}
Table~\ref{tab:nvidia-mean-ci} also reports two completed controls.
The fixed-L21 control repeats the L21 PCA block for every candidate model.
The common-layer control selects one layer for all models and penalties by mean validation routing utility over the thirteen penalties.
It trains one screening network per layer and penalty using the two source-family holdouts defined for MLP in Appendix~\ref{app:mlp-validation}, fitting preprocessing within each training fold.
Both controls use the same final shared architecture and ten-to-five ensemble rule, with the common layer's block repeated $M$ times.
Thus, these controls compare selection procedures and feature assembly.
The per-model selection procedure used in the main comparison has the highest mean among these three SharedTrunkNet variants on both benchmarks.
"""]
parts.append(start('Out-of-distribution mean utility over thirteen penalties, with 95\\% intervals.','tab:nvidia-mean-ci','lrr','Method & RouterBench & SPROUT'))
for m in ['fixed_l21','selected_layer','per_model','knn100','family_adaptation']:
 if m=='knn100':parts.append('\\midrule\n')
 parts.append(names[m]+' & '+' & '.join(ci(a['results'][ds]['rows'][m]['mean']) for ds in datasets)+r' \\[2pt]'+'\n')
parts.append(end())
parts.append(r"""
\paragraph{Paired utility differences.}
Table~\ref{tab:sharedtrunknet-differences} subtracts the main SharedTrunkNet variant from each L21 method, using the same queries and averaging all thirteen penalties.
Intervals use the paired query bootstrap in Appendix~\ref{app:routing-ci}.
Both L21 methods have higher mean utility on RouterBench and lower mean utility on SPROUT; none of these four difference intervals includes zero.
These intervals condition on the fitted routers and evaluated families.
""")
parts.append(start('Mean utility difference from SharedTrunkNet, with paired 95\\% intervals. Positive values favor the L21 method.','tab:sharedtrunknet-differences','lrr','Comparison & RouterBench & SPROUT'))
for m in ['knn100','family_adaptation']:
 parts.append(names[m]+' minus SharedTrunkNet & '+' & '.join(ci(a['results'][ds]['differences'][m],True) for ds in datasets)+r' \\[2pt]'+'\n')
parts.append(end())
parts.append(r"""
\paragraph{Results at every penalty.}
Tables~\ref{tab:sharedtrunknet-routerbench} and~\ref{tab:sharedtrunknet-sprout} retain every prescribed penalty, including reversals.
SharedTrunkNet denotes the per-model selection procedure used in the main text.
On RouterBench, the Family Adaptation has higher utility at 13 of 13 penalties and kNN-100 at 7 of 13.
On SPROUT, each has higher utility at 3 of 13 penalties.
These counts describe the observed comparisons; penalties are not independent statistical replications.
""")
for ds,label in [('routerbench-0shot','RouterBench'),('sprout-sixfam','SPROUT')]:
 parts.append(start(label+': out-of-distribution utility at every penalty, with 95\\% intervals.','tab:sharedtrunknet-'+('routerbench' if ds==datasets[0] else 'sprout'),'rrrr',r'$\lambda$ & SharedTrunkNet & L21 kNN-100 & L21 Family Adaptation'))
 for lam in [f'{v:g}' for v in a['results'][ds]['lambdas']]+['mean']:
  if lam=='mean':parts.append('\\midrule\n')
  parts.append(('Mean' if lam=='mean' else lam)+' & '+' & '.join(ci(a['results'][ds]['rows'][m][lam]) for m in ['per_model','knn100','family_adaptation'])+r' \\[2pt]'+'\n')
 parts.append(end())
text='\n'.join(parts)
assert 'NVIDIA' not in text and 'neighbour' not in text and '—' not in text
(OUT/'aistats_sharedtrunknet.tex').write_text(text)
# Reproducible supplementary layer-choice table from the completed audited sweep.
rec=json.loads((ROOT/'results/layer_sweep/summary/source_selection_audit.json').read_text())
assert rec['status']=='passed' and rec['heldout_family_excluded']
family_names={'grade-school-math':'GSM8K','mtbench':'MT-Bench','mmlupro':'MMLU-Pro','arc-challenge':'ARC','gsm8k':'GSM8K','hellaswag':'HellaSwag','mbpp':'MBPP','mmlu':'MMLU','mt-bench':'MT-Bench','winogrande':'WinoGrande','gpqa':'GPQA','math':'MATH','mmlu-pro':'MMLU-Pro','musr':'MuSR','openhermes':'OpenHermes','ragbench':'RAGBench'}
parts=[start('Layer selected using only the remaining training families, separately for each held-out family and router.','tab:layer-selected-families','lrrr','Held-out family & kNN-100 & Validated kNN & Family Adaptation')]
for ds,label in [('routerbench-0shot','RouterBench'),('sprout-sixfam','SPROUT')]:
 parts.append(r'\multicolumn{4}{l}{\textbf{'+label+r'}} \\'+'\n')
 splits=json.loads((ROOT/('configs/baseline_splits/'+ds+'.json')).read_text())
 for sp in sorted(s for s in splits if s.startswith('ood_')):
  fam=splits[sp]['held_out_family']
  values=[next(v['layer'] for v in rec['selections'] if v['dataset']==ds and v['split']==sp and v['method']==m) for m in ['knn100','knn_validated','family_adaptation']]
  parts.append(family_names.get(fam,fam.replace('_',r'\_'))+' & '+' & '.join(map(str,values))+r' \\'+'\n')
 if ds==datasets[0]:parts.append('\\midrule\n')
parts.append(end())
(OUT/'layer_selected_families.tex').write_text('\n'.join(parts))
print('RENDERED',len(text),'characters')
