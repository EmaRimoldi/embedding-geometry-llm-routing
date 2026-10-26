from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':180})
fig,axs=plt.subplots(2,3,figsize=(12,6.5));fig.subplots_adjust(top=.84,bottom=.15,wspace=.3,hspace=.49)
lines=['# Why do L21 neighbours help routing?','', 'Final fixed-kNN-100 diagnostic on frozen held-out-family splits. All 13 predeclared lambdas retained. Query-weighted estimates. No new router was tuned.','', '## Headline interpretation','', 'The mechanism differs by dataset. At lambda 50, cost predictions account for approximately 98% of the RouterBench utility gain in the symmetric prediction-swap decomposition; reward predictions account for approximately 89% of the SPROUT gain. This decomposes decisions and is not a causal attribution of encoder training.','', 'The random-neighbour control preserves each query’s exact source-family mixture. Much of the encoder difference remains with random queries from those families. This supports a role for changed source-family composition; it does not establish a universal benefit from selecting particular nearest queries within the chosen families. The within-family selection contrast can be negative, positive, or uncertain depending on dataset and lambda.','', '## Fixed protocol and verification','', '- Full-query profile errors, preference ordering and swaps: every query, same k=100 and frozen OOD folds.','- Exclusive and random-control analyses: the original fixed random geometry sample, weighted to population family proportions.','- Sample neighbourhood means reconstructed and matched exactly (tolerance 1e-12) to definitive saved predictions in both spaces. No target-family neighbour is allowed.','- Exclusive neighbourhoods have equal counts; identical sets are reported.','- Thirty random neighbourhoods per query preserve family counts and sample without replacement. Each draw is routed separately before metrics are averaged.','- One thousand paired query bootstrap samples stratified by family; intervals are conditional on observed families and fitted predictions. These are descriptive pointwise intervals, not simultaneous guarantees or uncertainty over unseen families.','- Outcomes are single recorded benchmark outcomes; no claim about irreducible expected-reward noise.','']
for i,(panel,label) in enumerate([('routerbench-0shot','RouterBench'),('sprout-sixfam','SPROUT')]):
 s=json.loads((ROOT/'results'/panel/'summary.json').read_text());lams=np.array(s['lambdas'],float)
 def series(mapping,pattern):
  vs=[mapping[pattern.format(l=f'{l:g}')] for l in lams];return np.array([v['mean'] for v in vs]),np.array([v['ci95'] for v in vs])
 def draw(ax,mapping,pattern,color,name,scale=1):
  y,ci=series(mapping,pattern);ax.plot(lams,y*scale,color=color,marker='o',ms=2.6,lw=1.5,label=name);ax.fill_between(lams,ci[:,0]*scale,ci[:,1]*scale,color=color,alpha=.12)
 draw(axs[i,0],s['full_contrasts'],'{l}|Qwen_minus_GTE|pair_accuracy','#C5393C','Qwen − GTE',100)
 draw(axs[i,1],s['full_contrasts'],'{l}|cost_swap_contribution','#007C91','Cost predictions')
 draw(axs[i,1],s['full_contrasts'],'{l}|reward_swap_contribution','#D17B24','Reward predictions')
 draw(axs[i,2],s['sample_contrasts'],'random_family_mixture_Qwen_minus_GTE|{l}|utility','#8561A8','Family mixture')
 draw(axs[i,2],s['sample_contrasts'],'selection_gain_Qwen_minus_GTE|{l}|utility','#34846C','Within-family selection')
 for j,ax in enumerate(axs[i]):
  ax.axhline(0,color='#777',lw=.8);ax.set_xscale('symlog',linthresh=1);ax.set_xlim(-.06,550);ax.set_xticks([0,1,10,100,500],['0','1','10','100','500']);ax.grid(alpha=.14);ax.set_xlabel('Cost penalty λ');ax.legend(frameon=False,fontsize=7,loc='best');ax.set_title(label+'\n'+['Model preference accuracy','Prediction-swap attribution','Family-composition control'][j],fontsize=10)
 axs[i,0].set_ylabel('Ordering accuracy gain (pp)');axs[i,1].set_ylabel('Utility contribution');axs[i,2].set_ylabel('Utility contribution (sample)')
 lines += [f'## {label}', '', f'Full cohort: {s["n_queries"]:,} queries. Fixed sample: {s["n_sample"]:,}. Identical neighbour sets: {s["identical_neighborhood_queries"]}.', '', '### Absolute prediction errors, all queries', '', '| Error | GTE | Qwen L21 | Qwen minus GTE (95% CI) |', '|---|---:|---:|---|']
 for key in ['reward_mae','reward_mse','centered_reward_mse','cost_mae_usd','cost_mse_usd2','centered_cost_mse_usd2']:
  g=s['full_absolute']['gte|'+key];q=s['full_absolute']['reader|'+key];v=s['full_contrasts']['Qwen_minus_GTE|'+key];lines.append(f'| {key} | {g:.7g} | {q:.7g} | {v["mean"]:.7g} [{v["ci95"][0]:.7g}, {v["ci95"][1]:.7g}] |')
 lines += ['', 'Lower error is better. In particular, L21 has worse raw cost MAE on SPROUT despite better routing. Better aggregate prediction error alone need not imply better top-model decisions.', '', '### All-lambda routing and swap results (full cohort)', '', '| λ | Utility gain | Pair accuracy gain (pp) | Cost contribution | Reward contribution |', '|---:|---:|---:|---:|---:|']
 for l in lams:
  get=lambda key:s['full_contrasts'][f'{l:g}|'+key]['mean']
  lines.append(f'| {l:g} | {get("Qwen_minus_GTE|utility"):.6f} | {100*get("Qwen_minus_GTE|pair_accuracy"):.3f} | {get("cost_swap_contribution"):.6f} | {get("reward_swap_contribution"):.6f} |')
 lines += ['', '### Family-matched random control (fixed sample)', '', '| λ | GTE actual − random | L21 actual − random | Random L21 mixture − random GTE mixture | Difference in within-family selection gain |', '|---:|---:|---:|---:|---:|']
 for l in lams:
  get=lambda key:s['sample_contrasts'][key]['mean']
  lines.append(f'| {l:g} | {get(f"gte|actual_minus_family_matched_random|{l:g}|utility"):.6f} | {get(f"reader|actual_minus_family_matched_random|{l:g}|utility"):.6f} | {get(f"random_family_mixture_Qwen_minus_GTE|{l:g}|utility"):.6f} | {get(f"selection_gain_Qwen_minus_GTE|{l:g}|utility"):.6f} |')
 lines += ['', 'The last two columns add to the actual encoder utility difference on the same sampled queries. This identity is a diagnostic contrast, not a causal decomposition. Sample totals should not be substituted for full-cohort performance.', '', '### Exclusive neighbours', '', '| Metric | L21-only minus GTE-only | 95% CI |', '|---|---:|---|']
 for key in ['reward_mae','centered_reward_mse','cost_mae_usd','50|pair_accuracy','50|utility']:
  v=s['sample_contrasts']['exclusive_Qwen_minus_GTE|'+key];lines.append(f'| {key} | {v["mean"]:.7g} | [{v["ci95"][0]:.7g}, {v["ci95"][1]:.7g}] |')
 lines += ['', 'Complete confidence intervals, per-family contrasts, Monte Carlo standard errors, source hashes and definitions are retained in `results/'+panel+'/summary.json`. Query-level arrays remain beside it on the cluster.', '']
fig.suptitle('What changes when kNN uses Qwen L21?',fontsize=16,y=.97)
fig.text(.5,.045,'Left and centre: all held-out queries. Right: fixed random sample with population weights.\nShading: pointwise 95% paired bootstrap intervals; numerical λ values are dataset-specific.',ha='center',fontsize=8,color='#555')
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'neighbor_explanation_all_lambdas.{ext}',bbox_inches='tight')
plt.close(fig)
(ROOT/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
print('Wrote full report and six-panel figure')
