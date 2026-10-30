"""Render the audited cost/reward diagnostic for the paper."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/diagnostics/knn_cost_reward_format'
DATASETS=['routerbench','sprout']
REPORTS={d:json.loads((OUT/d/'report.json').read_text()) for d in DATASETS}
def ci(z, signed=False, factor=1, digits=4):
    f=('+' if signed else '')+'.'+str(digits)+'f'
    return r'\resultci{'+format(z['mean']*factor,f)+'}{'+format(z['low']*factor,f)+'}{'+format(z['high']*factor,f)+'}'
def metric(d,key):
    return next(z for z in REPORTS[d]['rows'] if z['key']==key)
def delta(d,policy,reference='gte_reward_gte_cost',penalty='mean'):
    return next(z for z in REPORTS[d]['paired_utility_differences'] if z['policy']==policy and z['reference']==reference and z['penalty']==penalty)
policies=[
('gte_reward_gte_cost','GTE','GTE'),
('l21_reward_gte_cost','L21','GTE'),
('gte_reward_l21_cost','GTE','L21'),
('l21_reward_l21_cost','L21','L21'),
('gte_same_format','GTE with matching answer format',''),
('source_mean','Training-model means','')]
lines=[r"""\section{Cost, Reward and Answer-Format Controls}
\label{app:cost-reward-format}

\paragraph{Predictions and evaluation.}
We use the same out-of-distribution splits as the main comparison: each of the seven RouterBench and six SPROUT families is held out in turn.
All 34,771 RouterBench and 43,887 SPROUT queries are evaluated once.
For each test query $x$ and candidate model $m$, kNN-100 predicts reward $\widehat r(x,m)$ and cost $\widehat c(x,m)$ by averaging that model's recorded outcomes over the 100 nearest training queries.
GTE and L21 use their original cosine neighborhoods and frozen predictions.
Cost is the total dollar cost used in the routing objective, including both input and output costs where available; we do not measure answer length separately.
We retain every candidate model and prescribed penalty.

\paragraph{Prediction errors.}
Mean squared error (MSE) and mean absolute error (MAE) average equally over queries and candidate models.
To examine relative model preferences, we also center predictions and recorded outcomes separately across models within each query before computing the same errors.
Centering removes a common increase or decrease shared by all models.
The training-mean reference predicts each model's average reward and cost over all available training queries.
Table~\ref{tab:cost-reward-errors} reports the errors; Table~\ref{tab:cost-reward-relative} reports $100(1-E_{\mathrm{L21}}/E_{\mathrm{GTE}})$, where $E$ is the corresponding error.
Positive percentages mean that L21 has lower error.
Errors for reward and dollar cost have different units and should not be compared directly.

\paragraph{Answer-format control.}
For each test query, we restrict the training pool to queries with the same coarse answer-format label, then retrieve its 100 nearest GTE neighbors.
The two labels, multiple-choice and free-form, are defined in Appendix~\ref{app:answer-format}.
The query's entire family remains excluded, and every restricted training pool contains at least 100 queries.
We use the known task-format label, without generating or classifying any answers.
This control changes both format agreement and the available source-family composition; it does not isolate answer format as the sole cause.

\paragraph{Combining reward and cost estimates.}
We evaluate all four combinations of GTE and L21 reward and cost predictions.
At each penalty $\lambda$, every combination selects $\arg\max_m[\widehat r(x,m)-\lambda\widehat c(x,m)]$.
The GTE answer-format control uses both estimates from its restricted neighborhood.
We average recorded routing utility equally over test queries and then equally over the thirteen penalties.
Tables~\ref{tab:cost-reward-mean-ci} and~\ref{tab:cost-reward-differences} give means and paired differences; Tables~\ref{tab:cost-reward-routerbench} and~\ref{tab:cost-reward-sprout} retain all penalties.

\paragraph{Uncertainty and checks.}
All 95\% intervals use 2,000 paired bootstrap samples of queries within each held-out family, keeping family sizes fixed and using the same sampled queries across methods and penalties (Appendix~\ref{app:routing-ci}).
Each bootstrap mean is computed over the full penalty grid before taking interval endpoints.
These intervals condition on the evaluated families, frozen training pools and predictors.
Saved predictions reproduce the original GTE/L21 choices and utilities; restricted-neighbor predictions and every new routing choice are reconstructed from recorded outcomes.

"""]
def table_start(caption,label,columns,header):
    return '\n'.join([r'\begin{table}[H]',r'\centering\small',r'\setlength{\tabcolsep}{5pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{@{}'+columns+'@{}}',r'\toprule',header+r' \\',r'\midrule'])+'\n'
def table_end():return '\\bottomrule\n\\end{tabular}\n\\end{table}\n\n'
lines.append(table_start(r'Reward and cost prediction errors, with 95\% intervals. Cost MSE and MAE are multiplied by $10^6$ and $10^3$, respectively, for readability.','tab:cost-reward-errors','llrrrr',r'Benchmark & Predictor & Reward MSE & Reward MAE & Cost MSE & Cost MAE'))
for d,label in [('routerbench','RouterBench'),('sprout','SPROUT')]:
    for p,name in [('gte','GTE'),('l21','L21'),('gte_same_format','GTE, matching format'),('source_mean','Training-model means')]:
        values=[ci(metric(d,[p,t,'absolute',m]),factor=f) for t,m,f in [('r','mse',1),('r','mae',1),('c','mse',1e6),('c','mae',1e3)]]
        lines.append(label+' & '+name+' & '+' & '.join(values)+r' \\[2pt]'+'\n')
    if d=='routerbench':lines.append('\\midrule\n')
lines.append(table_end())
lines.append(table_start(r'L21 error reduction relative to GTE (\%), with paired 95\% intervals. Negative values indicate higher error.','tab:cost-reward-relative','lllrr',r'Target & Error & Centering & RouterBench & SPROUT'))
for target,name in [('r','Reward'),('c','Cost')]:
    for cent,clabel in [('absolute','None'),('centered','Across models')]:
        for met in ['mse','mae']:
            vals=[]
            for d in DATASETS:
                z=next(z for z in REPORTS[d]['relative_error_reductions'] if z['predictor']=='l21' and z['reference']=='gte' and z['target']==target and z['centering']==cent and z['metric']==met)
                vals.append(ci(z,factor=100,digits=2,signed=True))
            lines.append(' & '.join([name,met.upper(),clabel]+vals)+r' \\[2pt]'+'\n')
lines.append(table_end())
lines.append(table_start(r'Mean out-of-distribution utility over thirteen penalties, with 95\% intervals.','tab:cost-reward-mean-ci','llrr',r'Reward estimate & Cost estimate & RouterBench & SPROUT'))
for p,a,b in policies:
    prefix=a+' & '+b if b else r'\multicolumn{2}{l}{'+a+'}'
    lines.append(prefix+' & '+' & '.join(ci(metric(d,[p,'utility','mean'])) for d in DATASETS)+r' \\[2pt]'+'\n')
lines.append(table_end())
lines.append(table_start(r'Mean utility differences, with paired 95\% intervals. GTE and L21 alone denote ordinary kNN-100.','tab:cost-reward-differences','lrr',r'Comparison & RouterBench & SPROUT'))
comparisons=[
('l21_reward_gte_cost','gte_reward_gte_cost','Replace GTE reward estimate with L21'),
('gte_reward_l21_cost','gte_reward_gte_cost','Replace GTE cost estimate with L21'),
('l21_reward_l21_cost','gte_reward_gte_cost','Replace both GTE estimates with L21'),
('gte_same_format','gte_reward_gte_cost','GTE with matching format minus GTE'),
('gte_same_format','l21_reward_l21_cost','GTE with matching format minus L21')]
for p,ref,name in comparisons:
    lines.append(name+' & '+' & '.join(ci(delta(d,p,ref),signed=True) for d in DATASETS)+r' \\[2pt]'+'\n')
lines.append(table_end())
lines.append(r"""\paragraph{Results.}
On RouterBench, replacing only GTE's cost predictions raises mean utility from 0.6615 to 0.6965, whereas replacing only reward predictions gives 0.6754.
On SPROUT, replacing reward predictions raises utility from 0.7529 to 0.7709, whereas replacing only cost predictions gives 0.7560.
Thus, the contributions differ across benchmarks and are not additive.
SPROUT's L21 cost estimates have slightly higher MSE and MAE, yet still produce a small routing gain, illustrating that aggregate prediction error does not determine routing utility.
The answer-format restriction raises GTE utility to 0.6974 on RouterBench and 0.7584 on SPROUT, closing approximately 92\% and 27\% of the observed GTE--L21 gaps.
Both restricted-GTE means remain below L21, with paired difference intervals excluding zero.
These results support a substantial role for the format restriction on RouterBench and additional useful reward information in L21 on SPROUT.

""")
for d,label in [('routerbench','RouterBench'),('sprout','SPROUT')]:
    lines.append(table_start(label+r': utility at every penalty, with 95\% intervals. The first four columns identify the reward/cost predictors; the last uses GTE neighbors matching the query answer format.','tab:cost-reward-'+d,'rrrrrr',r'$\lambda$ & GTE/GTE & L21/GTE & GTE/L21 & L21/L21 & GTE, matching format'))
    completion=json.loads((OUT/d/'complete.json').read_text())
    for lam in [str(x) for x in completion['lambdas']]+['mean']:
        if lam=='mean':lines.append('\\midrule\n')
        lines.append(('Mean' if lam=='mean' else lam)+' & '+' & '.join(ci(metric(d,[p,'utility',lam])) for p,_,_ in policies[:5])+r' \\[2pt]'+'\n')
    lines.append(table_end())
paper=OUT/'paper';paper.mkdir(exist_ok=True)
(paper/'aistats_cost_reward.tex').write_text(''.join(lines))
print(paper/'aistats_cost_reward.tex')
