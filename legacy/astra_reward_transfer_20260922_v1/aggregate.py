"""Aggregate complete matched folds; partial runs are status only, never rankings."""
from pathlib import Path
import json
import numpy as np
from experiment import ROOT, LAMS

def interval(a,b,w):
    d=np.asarray(a)-np.asarray(b); w=np.asarray(w,float)
    rng=np.random.default_rng(20260922); ix=rng.integers(0,len(d),(10000,len(d)))
    macro=d[ix].mean(1); pooled=(d[ix]*w[ix]).sum(1)/w[ix].sum(1)
    return dict(delta_equal_family=float(d.mean()),delta_query_weighted=float(np.dot(d,w)/w.sum()),
        ci95_equal_family=np.quantile(macro,[.025,.975]).tolist(),
        ci95_query_weighted=np.quantile(pooled,[.025,.975]).tolist(),
        family_wins=int((d>1e-9).sum()),family_losses=int((d < -1e-9).sum()),
        worst_family_delta=float(d.min()),
        interpretation='paired bootstrap of observed families; exploratory, no multiplicity correction; not seed replication')

def main():
    records=[]; errors=[]
    files=list((ROOT/'results').glob('*/*/fold_*/result.json'))
    files+=list((ROOT/'baselines').glob('*/*/*/fold_*/result.json'))
    for p in files:
        d=json.loads(p.read_text()); meta=json.loads((ROOT/'inputs'/d['panel']/'manifest.json').read_text())
        if d['snapshot_sha256'] != meta['snapshot_sha256']:
            errors.append(str(p)+': input mismatch'); continue
        records.append((str(p.relative_to(ROOT)),d))
    out=dict(status={},aggregate={},comparisons={},geometry={},errors=errors)
    md=['# Matched OOD reward transfer', '',
        'Only methods with every held-out family are aggregated. All scores use frozen common rows.',
        'Intervals resample families, not random seeds; this remains an exploratory study.', '']
    for panel in LAMS:
        meta=json.loads((ROOT/'inputs'/panel/'manifest.json').read_text()); nf=len(meta['families'])
        z=np.load(ROOT/'inputs'/panel/'panel.npz'); counts=np.bincount(z['fam'],minlength=nf)
        pool={}; geo={}; origins={}
        for path,d in records:
            if d['panel']!=panel: continue
            g=d['family_index']; rep=d['representation']
            for L, methods in d['results'].items():
                for method,metrics in methods.items():
                    k=(rep,method,float(L))
                    # Same kNN reproduced in both runners must agree, rather than silently replace.
                    if g in pool.get(k,{}):
                        if abs(pool[k][g]['utility']-metrics['utility'])>1e-6:
                            errors.append(f'duplicate method mismatch: {panel} {k} {g}')
                        continue
                    pool.setdefault(k,{})[g]=metrics; origins.setdefault(k,[]).append(path)
            if 'geometry' in d: geo[rep,g]=d['geometry']
        status={f'{rep}/{method}/lambda={L:g}':len(v) for (rep,method,L),v in pool.items()}
        out['status'][panel]=status
        complete={k:v for k,v in pool.items() if set(v)==set(range(nf))}
        out['aggregate'][panel]={}
        md+=['## '+panel,'',f'Rows: {meta["rows"]}; families: {nf}; excluded over context: {meta["excluded"]}.','',
             'Encoder | Method | Lambda | Utility (queries) | Utility (families) | Reward | Cost',
             '---|---|---:|---:|---:|---:|---:']
        for (rep,method,L),v in sorted(complete.items()):
            mean={m:float(np.average([v[g][m] for g in range(nf)],weights=counts)) for m in ('utility','reward','cost','missing_choice_rate')}
            macro=float(np.mean([v[g]['utility'] for g in range(nf)]))
            tag=f'{rep}/{method}/{L:g}'
            out['aggregate'][panel][tag]=dict(query_weighted=mean,equal_family_utility=macro,
                per_family={meta['families'][g]:v[g] for g in range(nf)},files=origins[(rep,method,L)])
            md.append(f'{rep}|{method}|{L:g}|{mean["utility"]:.6f}|{macro:.6f}|{mean["reward"]:.6f}|{mean["cost"]:.8f}')
        if not complete: md.append('No complete method yet.||||||')
        comp={}
        for (rep,method,L),v in complete.items():
            if rep=='reader' and ('gte',method,L) in complete:
                vv=complete['gte',method,L]
                comp[f'reader_minus_gte/{method}/{L:g}']=interval([v[g]['utility'] for g in range(nf)],
                    [vv[g]['utility'] for g in range(nf)],counts)
            if method.startswith('selected_'):
                for brep in ('gte','reader'):
                    for base in ('prior','knn100','avengerspro64_top1','avengerspro64_top3','proxrouter_k32','proxrouter_k64','routellm_mf'):
                        if (brep,base,L) in complete:
                            vv=complete[brep,base,L]
                            comp[f'{rep}/{method}_minus_{brep}/{base}/{L:g}']=interval(
                                [v[g]['utility'] for g in range(nf)],[vv[g]['utility'] for g in range(nf)],counts)
        out['comparisons'][panel]=comp
        if len(geo)==2*nf:
            out['geometry'][panel]={f'{rep}/{meta["families"][g]}':v for (rep,g),v in geo.items()}
        md+=['',f'Completed result files: {sum(d["panel"]==panel for _,d in records)}.', '']
    out['errors']=errors
    if errors: md+=['## Consistency errors','']+['- '+e for e in errors]
    dest=ROOT/'reports'; dest.mkdir(exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(out,indent=2))
    (dest/'summary.md').write_text('\n'.join(md)+'\n')
    print('results',len(records),'consistency_errors',len(errors),flush=True)
    if errors: raise SystemExit(1)

if __name__=='__main__':main()
