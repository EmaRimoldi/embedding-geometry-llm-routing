"""Write separate geometry and dataset reports from complete audit products."""
from pathlib import Path
import json,collections
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'reports';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def wavg(v,w):return float(np.average(v,weights=w))
def geometry():
 lines=['# Geometry mechanisms: GTE versus Qwen L21','',
 'Sections 3 and 4 and all shared features remain unchanged. All transformations are fitted on source families only. Source-family-balanced linear regressions of unit embeddings on text features define subspaces; SVD retains numerically nonzero directions. We then remove these directions, renormalize, and rerun the same kNN100. Independent random directions with the same rank are controls.','',
 'These interventions isolate directions associated with measured features. They do not identify an exclusively surface or exclusively semantic subspace: information is entangled. GTE-Qwen2 and Qwen2.5 are different checkpoints, so this study cannot attribute differences solely to contrastive training.','',
 'The first intervention also removes the source mean component lying in the chosen subspace. The center-preserving companion removes only variation about that mean before renormalization; renormalization can itself change the mean. An effect that disappears in the companion is not evidence for the proposed feature mechanism.','']
 result={}
 for track in ['results','center_preserved_results','within_family_results']:
  for panel,nf in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)]:
   files=sorted((ROOT/'geometry'/track/panel).glob('*/result.json'));rows=[read(p) for p in files];tag=track+'/'+panel
   lines+=['## '+tag,'',f'Completed families: {len(rows)}/{nf}.','']
   if len(rows)!=nf:continue
   assert sorted(a['family_index'] for a in rows)==list(range(nf));w=[a['rows'] for a in rows];result[tag]={}
   lines+=['Representation | Intervention | Retained neighbors | Utility λ=10 | Utility λ=50 | Utility λ=100','---|---|---:|---:|---:|---:'] if nf==7 else ['Representation | Intervention | Retained neighbors | Utility λ=0.5 | Utility λ=2 | Utility λ=10','---|---|---:|---:|---:|---:']
   ls=['10','50','100'] if nf==7 else ['0.5','2','10']
   for rep in ['gte','reader','reader_mean']:
    for method in ['original','remove_length','remove_format','remove_all_surface','random_for_length','random_for_format','random_for_all_surface']:
     values=[]
     for a in rows:
      z=a['representations'][rep];key=method
      if method.startswith('random_for_'):
       k='remove_'+method.removeprefix('random_for_');key='remove_random_rank'+str(z[k]['removed']['rank'])
      values.append(z[key])
     ov=wavg([v['neighbor_overlap'] for v in values],w);us={l:wavg([v['routing'][l]['knn']['utility'] for v in values],w) for l in ls}
     result[tag][rep+'/'+method]=dict(neighbor_overlap=ov,utility=us,
       donor_feature_mse=np.average([v['feature_mean_squared_error'] for v in values],weights=w,axis=0).tolist(),
       per_family=[dict(family=a['family'],rows=a['rows'],routing=v['routing'],neighbor_overlap=v['neighbor_overlap']) for a,v in zip(rows,values)])
     lines.append(f'{rep}|{method}|{ov:.3f}|'+'|'.join(f'{us[l]:.4f}' for l in ls))
   lines+=['','All figures are query-weighted, with family-level results retained in geometry_summary.json. Lambda units are benchmark-specific. No winning intervention is selected on target outcomes.','']
 lines+=['## Within-family identification control','', 'A third intervention learns feature-associated directions after subtracting each source family’s own feature and embedding means. This tests whether the association survives removing between-family covariation during fitting. It retains the source-mean-preserving transformation at retrieval time. It does not fit on the held-out family.', '', '## Existing complementary evidence','',
 'The preceding study separates reward/cost heads, individual donors versus donor-family means, equal-size random donors, and encoder-only prompt presentation. Those tests answer distinct questions. Source-family replacement preserving performance does not prove absence of fine information; additional averaging can reduce noise. A useful model assignment within a family does not imply an optimal overall model mix.','',
 'Reports and code for those controls remain in astra_geometry_routing_20260922_v1. The present report adds coordinate interventions; it does not merge their evaluation populations or select favorable views.']
 (OUT/'geometry_report.md').write_text('\n'.join(lines)+'\n');(OUT/'geometry_summary.json').write_text(json.dumps(result,indent=2))

def datasets():
 base=ROOT/'datasets/results';data={p.stem:read(p) for p in base.glob('*.json')};lines=['# Independent dataset audit: LLMRouterBench and SPROUT','',
 'This report concerns dataset semantics and evaluation design, independently of the GTE/Qwen comparison. Raw inputs are preserved. Numerical diagnostics are descriptive; a realized outcome oracle is a hindsight reference, not an attainable expected-reward ceiling. Failed predictive probes do not establish irreducible noise.','',
 '## Provenance and interpretation','',
 '- LLMRouterBench: pinned release NPULH/LLMRouterBench at 0e5af1b8, pinned upstream code c77cb050. Original performance-cost panel: 12,446 queries and 13 models. Current geometry cohort: 12,109 rows fitting both encoders without truncation. The 337 exclusions include 332 SWE-bench queries; this materially changes that family.','- SPROUT: CARROT-LLM-Routing/SPROUT at d54e3cda2189ca1ed13a877a44ed34736c4b6920, 44,241 rows and 13 models; 44,167 fit GTE without truncation. This is not the later SPROUT-o3mini release with an additional model.','- SPROUT scores are fractional LLM-judge correctness scores. They are not Bernoulli success probabilities or repeated generation frequencies. LLMRouterBench mixes mostly binary benchmark scores with ArenaHard ties. A common [0,1] range does not make their semantics identical.','- Single released outcomes per query/model do not identify generation variance or judge variance separately. No noise irreducibility claim is supported by these tables.','',
 '## Families and aggregation','',
 'Historical SPROUT OOD jobs use 20 raw source identifiers. The benchmark contains six broad families. Raw-source holdout leaves other RAGBench, MuSR or MATH sources in training; broad-family holdout removes all of them. These are different generalization tasks. Twelve RAGBench sources contribute 60% of a 20-source macro average, versus 1/6 under broad-family macro averaging. Both reports must be labeled explicitly.','']
 for name in ['routerbench_reference','lrb_release','lrb_kept','lrb_complete_rows','lrb_complete_models','lrb_cost_observed_rows_12models','sprout_release6','sprout_kept6','sprout_kept20']:
  if name not in data:lines+=['Pending: '+name,''];continue
  a=data[name];lines+=['### '+name,'',f"Rows: {a['rows']}; models: {len(a['models'])}; groups: {len(a['families'])}.",
   f"Equal-reward queries: {a['all_equal_reward_fraction']:.2%}; all-zero queries: {a['all_zero_reward_fraction']:.2%}; fractional score entries: {a['fractional_reward_fraction']:.2%}.",
   f"Zero-cost entries: {a['zero_cost_cells']}; positive-reward zero-cost entries: {a['positive_reward_zero_cost_cells']}; absent records: {a['invalid_missing_cells']}.",'',
   'Global static frontier (descriptive full-cohort means): '+', '.join(a['query_static_frontier']['models'])+'.',
   'Positive lambda breakpoints: '+', '.join(f'{z:.4g}' for z in sorted(a['query_static_frontier']['breakpoints']))+'.','',
   'λ | Best static utility | Family hindsight utility | Realized oracle utility | Oracle retains maximal reward | Oracle selects absent record','---:|---:|---:|---:|---:|---:']
  ls=[0,.1,.5,1,2,5,10,50,100] if name.startswith('lrb') else [0,1,5,10,20,50,100,200,500,1000]
  for z in a['curves']:
   if z['lam'] in ls:lines.append(f"{z['lam']:g}|{z['global_hindsight_utility']:.4f}|{z['family_hindsight_utility']:.4f}|{z['oracle_utility']:.4f}|{z['oracle_retains_max_reward_fraction']:.2%}|{z['oracle_missing_choice_fraction']:.2%}")
  if 'duplicates' in a:lines+=['',f"Whitespace-normalized duplicate groups: {a['duplicates']['whitespace']['duplicate_groups']}; groups crossing release splits: {a['duplicates']['whitespace']['cross_split_groups']}; crossing family groups: {a['duplicates']['whitespace']['cross_family_groups']}."]
  lines+=['']
 if 'lrb_zero_cost_investigation' in data:
  a=data['lrb_zero_cost_investigation'];lines+=['## LLMRouterBench zero-cost investigation','',
   f"On the untruncated cohort, {a['zero_cost_nonzero_tokens']} recorded calls have nonzero token counts but zero cost; {a['positive_reward_zero_cost']} have positive reward and zero cost. These originate in the release.",
   'The pinned collector returns zero when cost metadata and configured prices are unavailable, so zero cannot automatically be read as a verified free inference. Some zero-cost failures may genuinely incur no charge. Stored usage is insufficient to reconstruct every missing bill exactly.','',
   'The original released convention, globally complete model pool, complete rows, and a conservative recording-quality subset are reported separately. Removing a model changes the candidate pool; dropping rows changes the population. Neither is silently substituted for the original benchmark. Descriptive NNLS token-price fits are diagnostics and are not used to overwrite costs.','']
 if 'sprout_raw_verification' in data or 'sprout_raw_verification_stream' in data:
  a=data.get('sprout_raw_verification_stream',data.get('sprout_raw_verification'));lines+=['## SPROUT raw numeric verification','',
   'Every key, score, input-token count and output-token count is compared with the pinned Parquet release. Costs are independently reconstructed from historical per-model prices, not current API prices.','',
   'Model | Records | Score/token mismatches | Empty responses | Positive score / empty response | Extractable judge score mismatches','---|---:|---:|---:|---:|---:']
  for m,z in a['models'].items():lines.append(f"{m}|{z['records']}|{sum(z.get('mismatch_'+k,0) for k in ['score','num_input_tokens','num_output_tokens'])}|{z['empty_response']}|{z['positive_score_empty_response']}|{z['judge_score_mismatch']}/{z['judge_score_extractable']}")
  lines+=['','Judge-score extraction only checks the recorded numeric field against its accompanying explanation. It does not independently validate correctness.','']
 if 'sprout_anomalies' in data:
  a=data['sprout_anomalies'];import re
  ss=[[float(v) for v in re.findall(r'[\"\x27]?correctness_score[\"\x27]?\s*:\s*([01](?:\.\d+)?)',z['judge'])] for z in a['unparsed_judge']]
  lines+=['## SPROUT release exceptions','',
   f"There are {len(a['unparsed_judge'])} judge responses with multiple score fields. The released score matches the first extracted value in {sum(bool(v) and abs(v[0]-z['score'])<1e-9 for v,z in zip(ss,a['unparsed_judge']))} cases; {sum(len(set(v))>1 for v in ss)} contain differing values. Some are multi-question judgments and some include self-corrections. Do not silently replace first with last; retain the cases as an auditable sensitivity list.",
   'All 430 Titan zero-output-token records contain the same refusal text. One receives 0.9 because the judge interprets inability to answer as consistent with an underdetermined reference problem. This is not proof of a cost parser bug, and its reward is a subjective judgment rather than an objective success label. One Mixtral response is whitespace-only with score zero.',
   'The numeric ingestion is exact. This verifies faithful reproduction, not the correctness of every judge decision.','']
 if 'outcome_profiles' in data:
  lines+=['## Representation-independent outcome profiles','', 'Panel | Unique model with maximal reward | Opportunity to improve on best static reward | Realized reward headroom | Relative-reward variance between families','---|---:|---:|---:|---:']
  for name,z in data['outcome_profiles'].items():lines.append(f"{name}|{z['unique_best_fraction']:.2%}|{z['oracle_improvement_opportunities']:.2%}|{z['oracle_mean_reward_headroom']:.4f}|{z['model_relative_reward_variance_family_fraction']:.2%}")
  lines+=['','The relative-reward variance subtracts each query’s mean reward across models, then measures the share of remaining empirical variance explained by family means. Unexplained variance includes both predictable structure and stochastic variation; it cannot be called noise from this calculation.','']
 if 'routing_billing_sensitivity' in data:
  lines+=['## Exposure of saved routers to suspect billing','', 'Method | λ | Selected suspect zero-cost call | Selected absent call','---|---:|---:|---:']
  for key,z in sorted(data['routing_billing_sensitivity']['summary'].items()):
   name,la=key.rsplit('/',1)
   if float(la) in [.5,2,10] and (name.startswith('knn/') or name=='kernel/reader/selected_blend'):lines.append(f"{name}|{la}|{z['unknown_cost_choice_rate']:.2%}|{z['missing_call_choice_rate']:.2%}")
  lines+=['','This identifies affected decisions. A clean-subset evaluation of saved choices is not a refit under a corrected data contract. Current anomalies do not automatically explain every weak routing result.','']
 lines+=['## Lambda interpretation and proposed freeze','',
 'Our objective is reward minus lambda times USD per query. A lambda value is an exchange rate in reward units per dollar; identical numbers across benchmarks do not imply identical routing regimes.','',
 'Current LRB {0,0.5,1,2,5,10} samples the main global strong-to-cheap transition, but misses the very small penalty range. Add 0.05, 0.1 and 0.2 to the diagnostic/final curve. Retain 20 and 50 as explicitly cost-sensitive sensitivity points; do not equate them to RouterBench values. Inspect source-only per-fold frontier breakpoints and model-use shares before fixing headline points.','',
 'SPROUT should not inherit the narrow LRB grid. The initial candidate grid is {0,1,2,5,10,20,50,100,200,500,1000}, subject to the source-only scales and complete six-family results. Its broad useful range extends beyond lambda 10; the cheap-versus-midrange transition is much later than in LRB. Extremes are diagnostics, not mandatory headline columns.','',
 'Each dataset result file contains leave-one-group-out reference costs and proposed dimensionless penalty grids fitted only on source groups. Scaling by one reference cost aligns one reference penalty, not every family or model trade-off. Report raw lambda, units, and both reward/cost operating points. Final settings should be frozen before comparing final method scores; these benchmarks have already informed exploratory development.','',
 'The pinned CARROT plot4 notebook uses (1-mu)*reward - 100*mu*USD-cost. For mu<1 this corresponds to our lambda=100*mu/(1-mu). Its spider notebook uses a factor of 1000, and both notebooks load the o3mini extension. There is no single directly interchangeable published lambda range for our original 13-model release.','',
 '## Fixed text-only signal controls','',
 'TF-IDF word unigrams/bigrams, at most 6000 training-fitted features, and fixed ridge alpha=10 predict per-model reward/cost. Three grouped random splits, official SPROUT test with training duplicates purged, and held-out-family/source folds are separate. This is a positive-signal diagnostic with a fixed learner, not an exhaustive encoder-free information test.','']
 for name in ['lrb','sprout6','sprout20']:
  key='text_signal_'+name
  if key not in data:lines+=['Pending: '+key,''];continue
  rr=data[key]['results'];lines+=['### '+name,'','Protocol | Method | λ | Utility | Reward | Cost','---|---|---:|---:|---:|---:']
  group=collections.defaultdict(list)
  for z in rr:
   protocol='iid_3seeds' if z['fold'].startswith('iid_') else ('official_test' if z['fold'].startswith('official') else 'heldout_all_groups')
   if z['method'] in ['global','family_lookup','tfidf_ridge','tfidf_reward_only','tfidf_cost_only'] and z['lam'] in ([0,.5,2,5,10] if name=='lrb' else [0,5,10,50,100,500]):group[(protocol,z['method'],z['lam'])].append(z)
  for (pro,me,la),ls in sorted(group.items()):
   w=[z['rows'] for z in ls];vs=[wavg([z[k] for z in ls],w) for k in ['utility','reward','cost']];lines.append(f'{pro}|{me}|{la:g}|{vs[0]:.4f}|{vs[1]:.4f}|{vs[2]:.6f}')
  lines+=['']
 lines+=['## Labels and unresolved limitations','',
 '- Family-only and raw-source holdout results do not depend on topic labels. Topic experiments must pin a label file and join by query ID, not silently substitute a newer version.','- The SPROUT v2 artifact contains 44,241 labels with multiple sources, including Qwen majority votes and family fallback. This is not equivalent to exhaustive manual verification. Its 151 no-majority rows retain family fallback.','- LRB accepted-label audit reports 98/100 on a coverage-oriented sample; SPROUT reports 98/100 on a uniform sample of an accepted subset. Neither establishes 98% accuracy on the entire final label release. Even a uniform 98/100 sample has approximately 93.0%–99.4% Wilson uncertainty.','- Duplicate grouping, source-family definition, incomplete outcomes, unknown costs, cohort exclusions, score semantics and aggregation must be frozen in the final rerun manifest.','- The historical SPROUT 13-model release and the later o3mini extension must not share a benchmark name without the version.','- Public source files establish intended semantics; the audit cannot recover unreleased generation replicates, missing billing records, or independent human ground truth for every judge score.','',
 '## Primary sources','',
 '- https://github.com/ynulihao/LLMRouterBench/tree/c77cb050','- https://huggingface.co/datasets/NPULH/LLMRouterBench','- https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT','- https://huggingface.co/CARROT-LLM-Routing','- https://arxiv.org/html/2502.03261v2','- Public CARROT code snapshot and exact URLs/hashes: sources/carrot_manifest.json.','']
 (OUT/'dataset_report.md').write_text('\n'.join(lines)+'\n')
 plot(data)

def plot(data):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
 for col,key in enumerate(['lrb_kept','sprout_kept6']):
  if key not in data:continue
  a=data[key];ls=np.array([z['lam'] for z in a['curves']]);mask=(ls>=.01)&(ls<=1000);title='LLMRouterBench, common cohort' if col==0 else 'SPROUT, six broad families'
  ax=axs[0,col]
  for field,label in [('oracle_utility','Realized oracle'),('family_hindsight_utility','Family hindsight'),('global_hindsight_utility','Best static, hindsight')]:ax.plot(ls[mask],[z[field] for z in a['curves'] if .01<=z['lam']<=1000],label=label)
  ax.set(xscale='log',xlabel='Lambda (reward units per USD)',ylabel='Query-weighted utility',title=title,ylim=(-.1,1.05));ax.legend(fontsize=8);ax.grid(alpha=.2)
  ax=axs[1,col]
  for field,label in [('oracle_retains_max_reward_fraction','Oracle retains maximal reward'),('oracle_uses_min_cost_fraction','Oracle selects cheapest recorded outcome')]:ax.plot(ls[mask],[z[field] for z in a['curves'] if .01<=z['lam']<=1000],label=label)
  ax.set(xscale='log',xlabel='Lambda (reward units per USD)',ylabel='Fraction of queries',ylim=(0,1.02));ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.suptitle('Dataset trade-off diagnostics; hindsight references are not deployable routers')
 fig.savefig(OUT/'dataset_regimes.pdf');fig.savefig(OUT/'dataset_regimes.png',dpi=160);plt.close(fig)

if __name__=='__main__':geometry();datasets();print('Separate reports written',flush=True)
