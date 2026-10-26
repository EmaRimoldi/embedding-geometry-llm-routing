"""Report complete six-fold panels only, preserving both macro and pooled metrics."""
import json
from collections import defaultdict
import numpy as np
from learning_common import ROOT,REPS,LAMS

def main():
    meta=json.load(open(ROOT/'inputs/manifest.json'));groups=defaultdict(list)
    for p in (ROOT/'results').rglob('result.json'):
        d=json.load(open(p));assert d['snapshot_sha256']==meta['snapshot_sha256']
        group=(d.get('kind','baselines'),d.get('seed',0),d['representation']);groups[group].append(d)
    records={};pending=[]
    for group,ds in groups.items():
        if len(ds)!=6 or {d['family_index'] for d in ds}!=set(range(6)):
            pending.append([group,len(ds)]);continue
        assert sum(d['rows'] for d in ds)==meta['rows']
        for lam in map(lambda x:f'{x:g}',LAMS['sprout-sixfam']):
            for method in ds[0]['results'][lam]:
                name='/'.join(map(str,group))+'/'+method
                vals=np.asarray([d['results'][lam][method]['utility'] for d in ds]);ns=np.asarray([d['rows'] for d in ds])
                records.setdefault(name,{})[lam]=dict(query_weighted=float(vals@ns/ns.sum()),equal_family=float(vals.mean()),
                    per_family={d['family']:float(v) for d,v in zip(ds,vals)})
    md=['# SPROUT matched GTE versus Qwen L21 routing','',f"Cohort: {meta['rows']} identical queries; six-family holdout. Utility uses original score minus lambda times USD cost.",'',
        'Every aggregate below has all six held-out families. Seed results are displayed separately; do not pick the best seed. No topic labels enter routing.','',f'Incomplete groups: {pending}','']
    for lam in map(lambda x:f'{x:g}',LAMS['sprout-sixfam']):
        md += [f'## Lambda {lam}','','| Method / seed / encoder | Utility, pooled | Utility, equal-family |','|---|---:|---:|']
        for name,vs in sorted(records.items(),key=lambda kv:-kv[1][lam]['query_weighted']):
            v=vs[lam];md.append(f"| {name} | {v['query_weighted']:.6f} | {v['equal_family']:.6f} |")
        md.append('')
    dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    (dest/'summary.json').write_text(json.dumps({'manifest':meta,'groups_complete':len(records),'pending':pending,'results':records},indent=2))
    (dest/'summary.md').write_text('\n'.join(md));print('wrote report',len(records),pending)
if __name__=='__main__':main()
