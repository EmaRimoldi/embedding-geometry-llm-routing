"""Export compact single-column tables from audited utility-rank summaries."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer'
assert json.loads((D/'audit.json').read_text())['status']=='passed'
rows=list(csv.DictReader((D/'per_lambda.csv').open()))
lookup={(r['dataset'],float(r['lambda_value'])):r for r in rows}
lam=sorted(set(float(r['lambda_value']) for r in rows))
lines=[r'\begin{table}[H]',r'\centering',r'\small',r'\setlength{\tabcolsep}{5pt}',
 r'\caption{Mean retrieval rank of the source family with the lowest relative-utility prediction error, averaging equally over held-out families. Sources are ranked by their share of GTE kNN-100 neighbours; rank 1 is highest. RouterBench has six eligible sources and SPROUT five. A dash denotes a penalty outside that benchmark''s prescribed grid.}',
 r'\label{tab:utility-family-rank}',r'\begin{tabular}{@{}rcc@{}}',r'\toprule',
 r'$\lambda$ & RouterBench & SPROUT \\',r'\midrule']
for l in lam:
 vals=[f"{float(lookup[(ds,l)]['mean_family_rank']):.2f}" if (ds,l) in lookup else r'---' for ds in ['routerbench','sprout']]
 lines.append(f'{l:g} & '+' & '.join(vals)+r' \\')
summary=json.loads((D/'summary.json').read_text())
lines += [r'\midrule','Mean & '+' & '.join(f"{s['mean_family_penalty_rank']:.2f}" for s in summary)+r' \\',r'\bottomrule',r'\end{tabular}',r'\end{table}']
(D/'per_lambda_table.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{table}[H]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
 r'\caption{GTE family retrieval versus predictive usefulness. The best source minimizes relative-utility prediction error for each held-out family and penalty. Mean rank weights families and the 13 prescribed penalties equally. Top-1 counts cases in which this source has the highest retrieval share. Candidate sources: six on RouterBench, five on SPROUT.}',
 r'\label{tab:utility-family-summary}',r'\begin{tabular}{@{}lcc@{}}',r'\toprule',
 r'Dataset & Mean rank & Top-1 \\',r'\midrule']
for s,label in zip(summary,['RouterBench','SPROUT']):
 lines.append(f"{label} & {s['mean_family_penalty_rank']:.2f} & {s['top1_count']:g}/{s['family_penalty_cases']}"+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(D/'summary_table.tex').write_text('\n'.join(lines)+'\n')
(D/'table_provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),D/'per_lambda.csv',D/'summary.json',D/'audit.json']},indent=2)+'\n')
