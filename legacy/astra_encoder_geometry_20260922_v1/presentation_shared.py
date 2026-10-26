"""Separate query-only mismatch from common-presentation geometry change."""
import json
import numpy as np
from geometry import unit,nearest
from presentation_controls import DEST,VARIANTS

def main():
    rows=[json.loads(s) for s in (DEST/'inputs/controlled.jsonl').read_text().splitlines()]
    result={};arrays={}
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        result[panel]={}
        baseids=[i for i,r in enumerate(rows) if r['kind']=='presentation' and r['panel']==panel and r['variant']=='original']
        fam=np.array([rows[i]['family'] for i in baseids]);ids=[rows[i]['row'] for i in baseids]
        for rep,pool in [('gte','primary'),('reader','primary'),('reader','mean'),('reader','last8')]:
            x=unit(np.load(DEST/'encoded'/rep/f'{pool}.npy'));base=x[baseids];out={}
            for v in VARIANTS:
                ix=[i for i,r in enumerate(rows) if r['kind']=='presentation' and r['panel']==panel and r['variant']==v]
                assert [rows[i]['row'] for i in ix]==ids
                changed=x[ix];rec={}
                for g in np.unique(fam):
                    train=np.flatnonzero(fam!=g);test=np.flatnonzero(fam==g);fold={}
                    for mode in ('native','source_centered'):
                        a,b=base,changed
                        if mode=='source_centered':
                            # Fit each mean on source families only, weighting families equally.
                            ma=np.mean([base[train[fam[train]==h]].mean(0) for h in np.unique(fam[train])],axis=0)
                            mb=np.mean([changed[train[fam[train]==h]].mean(0) for h in np.unique(fam[train])],axis=0)
                            a=unit(base-ma);b=unit(changed-mb)
                        orig,_=nearest(a[test],a[train],k=20)
                        shared,_=nearest(b[test],b[train],k=20)
                        query_only,_=nearest(b[test],a[train],k=20)
                        vals={}
                        for name,other in [('shared',shared),('query_only',query_only)]:
                            for k in (5,20):
                                overlaps=np.array([len(set(p[:k])&set(q[:k]))/k for p,q in zip(orig,other)])
                                vals[name+f'_overlap_k{k}']=float(overlaps.mean())
                                arrays[f'{panel}|{rep}|{pool}|{v}|{g}|{mode}|{name}|{k}']=overlaps
                        fold[mode]=vals
                    rec[str(int(g))]=fold
                out[v]=rec
            result[panel][rep+'/'+pool]=out
    dest=DEST/'results';dest.mkdir(exist_ok=True)
    (dest/'shared_presentation.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(dest/'shared_presentation_paired.npz',**arrays)
    md=['# Shared versus query-only presentation changes','',
        'Both conditions use the SAME outcome-independent sampled source cohort (at most 24 queries/family, ≤1536 original tokens). The query family is removed before retrieval. Each source and test query has its actual variant encoded. This is a smaller-candidate diagnostic, not directly comparable with full-source retrieval in summary.md. Centering is fitted on source families separately for each presentation, with equal family weights.',
        'GTE strips trailing whitespace, producing identical token sequences for both whitespace variants. Qwen keeps the raw text. Whitespace therefore tests pipeline preprocessing, not a matched token perturbation. Marker and header interventions survive preprocessing in both representations.','',
        'Benchmark | Representation | Change | Native query-only overlap@20 | Native shared overlap@20 | Centered shared overlap@20','---|---|---|---:|---:|---:']
    for panel,reps in result.items():
        for rep,variants in reps.items():
            for v,folds in variants.items():
                if v=='original':continue
                values=[np.mean([f[m][k] for f in folds.values()]) for m,k in [('native','query_only_overlap_k20'),('native','shared_overlap_k20'),('source_centered','shared_overlap_k20')]]
                md.append(f'{panel}|{rep}|{v}|'+ '|'.join(f'{a:.3f}' for a in values))
    md+=['','High shared overlap despite low query-only overlap points to presentation mismatch. Low shared overlap means neighbor relationships also change when both sides receive the same presentation. Improvement after source centering is consistent with a removable common component, not proof of a specific mechanism. k=5 and per-family values are retained.']
    (dest/'shared_presentation.md').write_text('\n'.join(md)+'\n');print('Shared presentation comparison complete',flush=True)

if __name__=='__main__':main()
