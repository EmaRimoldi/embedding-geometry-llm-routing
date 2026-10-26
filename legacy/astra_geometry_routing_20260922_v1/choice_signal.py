"""Does routing use query-specific signal inside an unseen family?"""
import json
import numpy as np
from common import *

def signal(ch,r,c,lam,groups):
    rr=r[np.arange(len(r)),ch];cc=c[np.arange(len(c)),ch]
    ref_r=np.empty(len(r));ref_c=np.empty(len(c))
    for g in np.unique(groups):
        rows=np.flatnonzero(groups==g);freq=np.bincount(ch[rows],minlength=r.shape[1])/len(rows)
        ref_r[rows]=r[rows].mean(0)@freq;ref_c[rows]=c[rows].mean(0)@freq
    dr=float((rr-ref_r).mean());dc=float((cc-ref_c).mean())
    return dict(utility=float((rr-lam*cc).mean()),expected_permuted_utility=float((ref_r-lam*ref_c).mean()),
        query_specific_utility=dr-lam*dc,reward_contribution=dr,cost_contribution=-lam*dc)

def main():
    result={};md=['# Query-specific routing signal under family holdout','',
        'The reference randomly reassigns a router’s model choices among queries of the same held-out family, preserving its exact model-use frequencies. Its expected utility is calculated analytically. A second reference also preserves input-length quintiles, whose boundaries are fitted on source families. The target-family outcomes appear only in this evaluation diagnostic, never in training or model selection. This is not a deployable baseline.','',
        'A positive difference means the assignment of choices to particular queries is useful beyond those controls. A constant choice has zero difference. This does not prove an information-theoretic limit or noise irreducibility.','']
    for panel,nf in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)]:
        d=load(panel);f=d['fam'];records=[]
        # Each record: method, family, lambda, choices, source file.
        for g in range(nf):
            te=np.flatnonzero(f==g)
            sources=[]
            p=ROOT/'results/donors'/panel/f'fold_{g:02d}'
            if (p/'result.json').exists():
                names=[f'{rep}/k100/{mode}' for rep in REPS for mode in ('knn','family_composition')]
                sources.append((p/'paired.npz','donors',names,'plain'))
            for rep in ('gte','reader'):
                p=OLD/'results'/panel/rep/f'fold_{g:02d}'
                if (p/'result.json').exists():sources.append((p/'predictions.npz','previous/'+rep,['prior','cost_only','selected_reward','selected_reward_cost'],'plain'))
                for kind,names in [('mf',['routellm_mf']),('classical',['avengerspro64_top1','avengerspro64_top3','proxrouter_k32','proxrouter_k64'])]:
                    p=OLD/'baselines'/kind/panel/rep/f'fold_{g:02d}'
                    if (p/'result.json').exists():sources.append((p/'predictions.npz','existing/'+rep,names,'plain'))
            for kind,seeds,reps in [('kernel',(42,),REPS),('mlp',(42,7),('gte','reader')),('surface',(42,),('text',))]:
                for seed in seeds:
                    for rep in reps:
                        p=ROOT/'results'/kind/f'seed_{seed}'/panel/rep/f'fold_{g:02d}'
                        if (p/'result.json').exists():
                            names=['direct','selected_blend','one_se_blend','reward_only','cost_only']
                            sources.append((p/'predictions.npz',f'{kind}/seed{seed}/{rep}',names,'choice'))
            tr=np.flatnonzero(f!=g);edges=np.unique(np.quantile(np.log1p(d['tokens'][tr]),[.2,.4,.6,.8]))
            bins=np.digitize(np.log1p(d['tokens'][te]),edges)
            for path,prefix,names,keytype in sources:
                z=np.load(path);assert np.array_equal(z['query_ids'],d['ids'][te])
                for lam in LAMS[panel]:
                    for name in names:
                        key=f'{lam:g}|'+('choice|' if keytype=='choice' else '')+name
                        if key not in z.files:continue
                        ch=z[key];records.append(dict(method=prefix+'/'+name,family=g,rows=len(te),lam=lam,
                            unconditional=signal(ch,d['r'][te],d['c'][te],lam,np.zeros(len(te),int)),
                            length_matched=signal(ch,d['r'][te],d['c'][te],lam,bins),file=str(path)))
        pool={}
        for r in records:pool.setdefault((r['method'],r['lam']),{})[r['family']]=r
        summary={};md+=['## '+panel,'','Method | Lambda | Signal beyond family | Signal beyond family and length','---|---:|---:|---:']
        for (method,lam),folds in sorted(pool.items()):
            if set(folds)!=set(range(nf)):continue
            w=np.array([folds[g]['rows'] for g in range(nf)]);entry={}
            for ref in ('unconditional','length_matched'):
                values=[folds[g][ref]['query_specific_utility'] for g in range(nf)]
                entry[ref]=dict(query_weighted=float(np.average(values,weights=w)),equal_family=float(np.mean(values)),
                    per_family=[folds[g][ref] for g in range(nf)])
            summary[method+f'/{lam:g}']=entry
            md.append(f'{method}|{lam:g}|{entry["unconditional"]["query_weighted"]:+.4f}|{entry["length_matched"]["query_weighted"]:+.4f}')
        result[panel]=dict(summary=summary,records=records)
    dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    (dest/'choice_signal.json').write_text(json.dumps(result,indent=2));(dest/'choice_signal.md').write_text('\n'.join(md)+'\n')
    print('Choice-specific signal report complete',flush=True)

if __name__=='__main__':main()
