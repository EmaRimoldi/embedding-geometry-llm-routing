"""Report all planned primary contrasts; figures use measured native geometry."""
from pathlib import Path
import itertools,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry import ROOT

def mean_ci(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)];rng=np.random.default_rng(9022)
    if not len(a):return dict(mean=None,ci95=None,n=0)
    boot=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(a))

def signflip(a):
    a=np.asarray(a);signs=np.array(list(itertools.product((-1,1),repeat=len(a))))
    return float(np.mean(np.abs((signs*a).mean(1))>=abs(a.mean())-1e-12))

def fdr(ps):
    order=np.argsort(ps);v=np.asarray(ps)[order]*len(ps)/np.arange(1,len(ps)+1)
    v=np.minimum.accumulate(v[::-1])[::-1];out=np.empty(len(ps));out[order]=np.minimum(v,1);return out

def main():
    dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    md=['# GTE versus Qwen L21: geometry results','',
        'Primary cohort: identical, untruncated queries. Atlas queries are sampled without outcomes, at most 256 per held-out family; every eligible source query is a retrieval candidate.',
        'Main neighborhood size is k=20; all k=5,20,100,500 and preprocessing controls remain in the per-fold JSON files.',
        'Gain means 1 minus neighbor outcome squared distance divided by expected distance to random donors matched on source family and input-length quintile. Each encoder has its own matched donor composition. This is not routing utility. Reported averages weight held-out families equally.',
        'Paired intervals resample held-out families. The ten planned native-geometry attribute contrasts use exact family sign-flip tests with Benjamini–Hochberg correction. These are exploratory and do not establish universal encoder superiority.','']
    report={'atlas':{},'primary_contrasts':{},'probes':{}};pkeys=[];pvals=[]
    for panel,nf in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)]:
        paths=sorted((ROOT/'results/atlas'/panel).glob('fold_*.json'));folds=[json.loads(p.read_text()) for p in paths]
        assert len(folds)==nf and {d['family_index'] for d in folds}==set(range(nf))
        md+=['## '+panel,'','Target | GTE gain | Qwen L21 gain | Paired difference [95% family CI] | Families',
             '---|---:|---:|---|---:']
        for target in ('reward_level','solve_fraction','reward_profile','logcost_level','logcost_profile'):
            vals={'gte':[],'reader':[]};keep=[]
            for d in folds:
                v=[d['encoders'][rep]['modes']['native_cosine']['behavior'][target].get('20',{}).get('relative_distance_gain') for rep in vals]
                if all(a is not None for a in v):
                    for rep,a in zip(vals,v):vals[rep].append(a)
                    keep.append(d['family'])
            if not keep:continue
            delta=np.array(vals['reader'])-np.array(vals['gte']);ci=mean_ci(delta);pk=panel+'/'+target
            report['primary_contrasts'][pk]=dict(gte=mean_ci(vals['gte']),reader=mean_ci(vals['reader']),paired=ci,families=keep,per_family_delta=delta.tolist())
            pkeys.append(pk);pvals.append(signflip(delta))
            md.append(f'{target}|{np.mean(vals["gte"]):+.3f}|{np.mean(vals["reader"]):+.3f}|{ci["mean"]:+.3f} [{ci["ci95"][0]:+.3f}, {ci["ci95"][1]:+.3f}]|{len(keep)}')
        if panel=='routerbench-0shot':
            md+=['','### Genuine held-out-family format neighborhoods','','Family | GTE native | Qwen native | GTE whitened | Qwen whitened','---|---:|---:|---:|---:']
            for d in folds:
                vals=[d['encoders'][rep]['modes'][mode]['neighbors']['20']['same_format_share'] for mode in ('native_cosine','pca128_white') for rep in ('gte','reader')]
                md.append(d['family']+'|'+'|'.join(f'{v:.3f}' for v in vals))
            labels=['ARC','GSM8K','HellaSwag','MBPP','MMLU','MT-Bench','Winogrande']
            fig,axes=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
            for ax,rep,title in zip(axes,('gte','reader'),('GTE: semantic embedding','Qwen L21: last-token state')):
                a=np.array([d['encoders'][rep]['modes']['native_cosine']['neighbors']['20']['source_shares'] for d in folds])
                im=ax.imshow(a,cmap='Blues',vmin=0,vmax=1);ax.set_title(title)
                ax.set_xticks(range(7),labels,rotation=40,ha='right');ax.set_yticks(range(7),labels)
                ax.set_xlabel('Source family of retrieved neighbors');ax.set_ylabel('Held-out family')
                for i in range(7):
                    for j in range(7):
                        if i==j:ax.text(j,i,'×',ha='center',va='center',color='gray')
                        elif a[i,j]>=.05:ax.text(j,i,f'{100*a[i,j]:.0f}%',ha='center',va='center',color='white' if a[i,j]>.55 else 'black',fontsize=9)
            fig.colorbar(im,ax=axes,shrink=.8,label='Share of top-20 neighbors')
            fig.savefig(dest/'ood_neighbor_migration.png',dpi=180);fig.savefig(dest/'ood_neighbor_migration.svg');plt.close(fig)
        report['atlas'][panel]=dict(families=nf,neighbor_overlap_k20=mean_ci([d['neighbor_overlap']['20'] for d in folds]),
            effective_rank={rep:mean_ci([d['encoders'][rep]['structure']['participation_ratio'] for d in folds]) for rep in ('gte','reader')})
        # Keep partially completed probe runs as explicit status; never silently aggregate them.
        pp=sorted((ROOT/'results/probes'/panel).glob('fold_*.json'));report['probes'][panel]={'completed_folds':len(pp),'expected_folds':nf}
        if len(pp)==nf:
            dd=[json.loads(p.read_text()) for p in pp];summary={}
            md+=['','### Information accessibility (fixed ridge alpha=100)','','Feature / target | GTE OOD R² vs source mean | Qwen OOD R² vs source mean','---|---:|---:']
            for feature in ('length','norm_length','unit','unit_controls','raw'):
                for target in ('reward_level','solve_fraction','reward_profile'):
                    vals={rep:[d['encoders'][rep]['features'][feature]['100.0']['ood'][target]['r2_vs_source_mean'] for d in dd] for rep in ('gte','reader')}
                    summary[feature+'/'+target]={rep:mean_ci(a) for rep,a in vals.items()}
                    md.append(f'{feature}/{target}|{np.mean(vals["gte"]):+.3f}|{np.mean(vals["reader"]):+.3f}')
            report['probes'][panel]['summary']=summary
            md+=['','### Probe sensitivity and cost accessibility','',
                 'Ridge uses source-only feature standardization and family-balanced fitting. These fixed penalties diagnose accessibility; they are not an optimally tuned router or an information ceiling. R² here is 1 minus test MSE divided by the MSE of the source-training mean, averaged equally over eligible held-out families.','',
                 'Target | Alpha | GTE IID | Qwen IID | GTE OOD | Qwen OOD | OOD families',
                 '---|---:|---:|---:|---:|---:|---:']
            for target in ('reward_level','reward_profile','logcost_level','logcost_profile','cross_space'):
                for alpha in ('10.0','100.0','1000.0'):
                    values=[];counts=[]
                    for split in ('iid','ood'):
                        for rep in ('gte','reader'):
                            aa=[]
                            for d in dd:
                                block=(d['cost_probes'].get(rep+'/unit',{}).get(alpha,{}) if target.startswith('logcost') else d['encoders'][rep]['features']['unit'][alpha])
                                val=block.get(split,{}).get(target,{}).get('r2_vs_source_mean')
                                if val is not None:aa.append(val)
                            values.append(np.mean(aa) if aa else float('nan'));counts.append(len(aa))
                    md.append(f'{target}|{alpha}|'+ '|'.join(f'{v:+.3f}' for v in values)+f'|{min(counts[2:])}')
            md+=['','Cross-space targets differ by direction: the GTE column predicts source-fitted Qwen PCs, and the Qwen column predicts source-fitted GTE PCs. Their reconstruction scores are not a symmetric test of which representation contains more information.']
    for k,p,q in zip(pkeys,pvals,fdr(pvals)):
        report['primary_contrasts'][k].update(exact_signflip_p=p,bh_q=float(q))
    md+=['','## Family-level uncertainty for the ten primary neighborhood contrasts','',
         'Contrast | Exact sign-flip p | BH-adjusted q','---|---:|---:']
    for k in pkeys:
        d=report['primary_contrasts'][k]
        md.append(f'{k}|{d["exact_signflip_p"]:.4f}|{d["bh_q"]:.4f}')
    md+=['','None passes q < 0.05. Bootstrap intervals are descriptive and do not override this result. These tests assume exchangeable family-level signs under the null; overlapping source folds and only 7 or 10 benchmark families limit generalization.']
    confirm=json.loads((ROOT/'results/confirmation/result.json').read_text());report['confirmation']=confirm
    md+=['','## Controlled prompts: same problem versus same response format','',
        'Fraction of comparisons where the same problem under another response instruction is nearer than another domain’s problem under the same response instruction. All comparisons use actual encoded prompts, not averaged prototypes. Constructional demand and wording version are matched; superficial difficulty label is neutral.',
        'Response modes are answer-only, explained solution, Python code, and JSON. These mix output formatting with requested work, so the joint effect is instruction sensitivity, not a pure surface-format effect. The eight synthetic domains are mathematical, logical, and programming tasks. In causal mean pooling, states before a suffix instruction cannot attend to it; content retention is not evidence of superior comprehension.','',
        'Representation | Prefix/no marker | Prefix/shared marker | Suffix/no marker | Suffix/shared marker','---|---:|---:|---:|---:']
    names=['gte/primary','reader/primary','reader/mean','reader/last8'];labels=['GTE','Qwen L21 last','Qwen L21 mean','Qwen L21 last 8']
    matrix=[]
    for name in names:
        vals=[confirm['literal_prompts'][name][f'position={p},terminal={t}']['same_problem_cross_format_win']['mean'] for p,t in itertools.product((0,1),repeat=2)]
        matrix.append(vals);md.append(name+'|'+'|'.join(f'{v:.1%}' for v in vals))
    fig,ax=plt.subplots(figsize=(12,4.6),layout='constrained');im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(range(4),labels);ax.set_xticks(range(4),['Instruction first\nNo marker','Instruction first\n“Response:” marker','Instruction last\nNo marker','Instruction last\n“Response:” marker'])
    ax.set_title('Same problem with another instruction, or another problem with the same instruction? ',fontsize=12)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{100*matrix[i][j]:.1f}%',ha='center',va='center',color='black' if matrix[i][j]>.65 else 'white',fontsize=12)
    fig.colorbar(im,ax=ax,label='Same-problem proximity rate',shrink=.9)
    fig.savefig(dest/'controlled_content_vs_response.png',dpi=180);fig.savefig(dest/'controlled_content_vs_response.svg');plt.close(fig)
    md+=['','## Interpretation boundaries','',
        '- A native-neighborhood difference can come from metric emphasis, anisotropy, norm, or pooling. Information accessibility is a separate measurement.',
        '- The marker and instruction-position sensitivity is itself a result. It prevents describing Qwen last-token geometry as an invariant representation of task meaning.',
        '- GTE norm differences are numerical fluctuations after an explicit normalization module.',
        '- Synthetic high/low levels are computational constructions, not measured model difficulty.',
        '- Cost comparisons exclude any query with a nonpositive model cost. This can remove whole problematic families in LRB; eligible-family counts are shown.',
        '- GTE-Qwen2 and Qwen2.5-Instruct are different checkpoints. This study does not isolate the contrastive objective.',
        '- The prior reward-router jobs continue separately. No routing success is required for reporting these geometry results.','',
        'Full evidence: results/atlas, results/probes, results/controlled, results/confirmation, results/global. Methodological corrections: ISSUES.md.']
    global_path=ROOT/'results/global/result.json'
    if global_path.exists():report['global_geometry']=json.loads(global_path.read_text())
    (dest/'summary.json').write_text(json.dumps(report,indent=2));(dest/'summary.md').write_text('\n'.join(md)+'\n')
    print('reports written',dest,flush=True)

if __name__=='__main__':main()
