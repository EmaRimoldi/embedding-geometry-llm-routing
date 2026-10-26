import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
folds=[json.loads((R/'results'/f'fold{g}.json').read_text()) for g in range(6)]
assert len({d['cohort_sha'] for d in folds})==1
assert len({d['code_sha'] for d in folds})==1
assert sum(d['n'] for d in folds)==43887
scores={};md=['# Cost router on canonical SPROUT','',
 'Six complete held-out families; 43,887 matched queries; GTE reward prior held fixed.',
 'Entries are query-weighted utility [equal-family utility]. All gates use source families only.', '']
for lam in folds[0]['lambdas']:
 L=str(lam);scores[L]={};md += [f'## Lambda {lam:g}','','Method | Utility [equal-family]','---|---']
 for method in folds[0]['scores'][L]:
  vals=np.array([d['scores'][L][method]['utility'] for d in folds]);w=np.array([d['n'] for d in folds])
  scores[L][method]={'query_weighted':float(np.average(vals,weights=w)),'equal_family':float(vals.mean()),
                      'per_family':dict(zip([d['family'] for d in folds],vals.tolist()))}
 for method,v in sorted(scores[L].items(),key=lambda a:-a[1]['query_weighted']):
  md.append(f"{method} | {v['query_weighted']:.5f} [{v['equal_family']:.5f}]")
(R/'reports').mkdir(exist_ok=True)
(R/'reports/summary.json').write_text(json.dumps({'cohort_sha':folds[0]['cohort_sha'],'scores':scores},indent=2))
(R/'reports/summary.md').write_text('\n'.join(md)+'\n')
print('Complete six-family report saved',flush=True)
