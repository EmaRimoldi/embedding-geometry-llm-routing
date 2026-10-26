"""Complete-cohort routing comparisons with explicit partial-job status."""
import json
import numpy as np
from common import *

def interval(v,w):
    a=np.asarray(v,float);w=np.asarray(w,float);rng=np.random.default_rng(9022)
    ix=rng.integers(0,len(a),(10000,len(a)))
    return dict(query_weighted=float(a@w/w.sum()),equal_family=float(a.mean()),
        equal_family_ci95=np.quantile(a[ix].mean(1),[.025,.975]).tolist(),
        query_weighted_ci95=np.quantile((a[ix]*w[ix]).sum(1)/w[ix].sum(1),[.025,.975]).tolist(),
        family_wins=int((a>1e-9).sum()),family_losses=int((a < -1e-9).sum()),per_family=a.tolist())

def main():
    output={'methods':{},'status':{},'same_classifier':{},'donor_decomposition':{},'errors':[]};lines=['# Geometry to routing: current results','',
        'All aggregate routing comparisons below require every outer family on the same frozen no-truncation cohort. Intervals resample the observed families and are exploratory; overlapping source fits and repeated method development limit confirmatory inference. Separate seeds are not independent families. Sections 3 and 4 are unchanged.','']
    for panel,nf in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)]:
        d=load(panel);counts=np.bincount(d['fam'],minlength=nf);pool={}
        def add(tag,g,lam,metric):
            pool.setdefault((tag,float(lam)),{})[g]=metric
        # Prior experiment: strict common-row baseline reproductions and research methods.
        paths=list((OLD/'results'/panel).glob('*/fold_*/result.json'))+list((OLD/'baselines').glob(f'*/{panel}/*/fold_*/result.json'))
        for p in paths:
            a=json.loads(p.read_text());assert a['snapshot_sha256']==d['meta']['snapshot_sha256']
            rep=a['representation'];g=a['family_index']
            isbaseline='baselines' in p.parts
            for lam,methods in a['results'].items():
                for name,metric in methods.items():
                    if name in ('oracle','family_hindsight_reference'):continue
                    prefix='existing' if isbaseline else 'previous'
                    add(prefix+'/'+rep+'/'+name,g,lam,metric)
        donorfiles=sorted((ROOT/'results/donors'/panel).glob('fold_*/result.json'));donors=[json.loads(p.read_text()) for p in donorfiles]
        for a in donors:
            assert a['snapshot_sha256']==d['meta']['snapshot_sha256']
            for lam,z in a['results'].items():
                for name,metric in z['methods'].items():add('donors/'+name,a['family_index'],lam,metric)
        for p in (ROOT/'results/random_donors'/panel).glob('fold_*/result.json'):
            a=json.loads(p.read_text());assert a['snapshot_sha256']==d['meta']['snapshot_sha256']
            for representation_reference,ls in a['results'].items():
                for lam,z in ls.items():add('smoothed_donors/'+representation_reference,a['family_index'],lam,z['smoothed'])
        for kind in ('kernel','mlp','surface'):
            for p in (ROOT/'results'/kind).glob(f'seed_*/{panel}/*/fold_*/result.json'):
                a=json.loads(p.read_text());assert a['snapshot_sha256']==d['meta']['snapshot_sha256']
                for lam,methods in a['results'].items():
                    for name,metric in methods.items():add(f'{kind}/seed{a["seed"]}/{a["representation"]}/{name}',a['family_index'],lam,metric)
        for p in (ROOT/'results/mf_control').glob(f'*/{panel}/*/fold_*/result.json'):
            a=json.loads(p.read_text());assert a['snapshot_sha256']==d['meta']['snapshot_sha256']
            for lam,methods in a['results'].items():
                for name,metric in methods.items():add(f'mf_{a["settings"]["input_mode"]}/{a["representation"]}/{name}',a['family_index'],lam,metric)
        for p in (ROOT/'results/cross_space').glob(f'*/seed_*/{panel}/fold_*/result.json'):
            a=json.loads(p.read_text());assert a['snapshot_sha256']==d['meta']['snapshot_sha256']
            for lam,methods in a['results'].items():
                for name,metric in methods.items():add(f'cross/{a["kind"]}/seed{a["seed"]}/{name}',a['family_index'],lam,metric)
        complete={key:v for key,v in pool.items() if set(v)==set(range(nf))}
        output['status'][panel]={tag+f'/lambda={lam:g}':len(v) for (tag,lam),v in pool.items()}
        output['methods'][panel]={}
        for (tag,lam),folds in complete.items():
            metrics={k:float(np.average([folds[g][k] for g in range(nf)],weights=counts)) for k in ('utility','reward','cost','missing_choice_rate')}
            output['methods'][panel][tag+f'/lambda={lam:g}']=dict(query_weighted=metrics,equal_family_utility=float(np.mean([folds[g]['utility'] for g in range(nf)])),per_family=folds)
        lines+=['## '+panel,'',f'Common rows: {len(d["r"])}; outer families: {nf}.','',
                'The original RouteLLM-MF reproduction consumes raw representation outputs: GTE is already normalized and Qwen is not. Unit-vector and source-RMS controls are separate pending/completed methods; raw-input differences do not isolate angular geometry.','',
                '### Same classifier, different representation','',
                'Classifier | Lambda | GTE utility | Qwen L21 utility | Difference | Equal-family difference','---|---:|---:|---:|---:|---:']
        pairs=[('kNN100','donors/gte/k100/knn','donors/reader/k100/knn')]
        for name in ('routellm_mf','avengerspro64_top1','avengerspro64_top3','proxrouter_k32','proxrouter_k64'):
            pairs.append((name,'existing/gte/'+name,'existing/reader/'+name))
        for mode in ('unit','source_rms'):
            pairs.append(('routellm_mf_'+mode,f'mf_{mode}/gte/routellm_mf',f'mf_{mode}/reader/routellm_mf'))
        for kind,seeds in [('kernel',(42,)),('mlp',(42,7))]:
            for seed in seeds:
                for name in ('direct','selected_blend','one_se_blend','direct_geometry'):
                    pairs.append((f'{kind}, seed {seed}, {name}',f'{kind}/seed{seed}/gte/{name}',f'{kind}/seed{seed}/reader/{name}'))
        comps={}
        for name,a,b in pairs:
            for lam in LAMS[panel]:
                if (a,lam) not in complete or (b,lam) not in complete:continue
                av=np.array([complete[a,lam][g]['utility'] for g in range(nf)]);bv=np.array([complete[b,lam][g]['utility'] for g in range(nf)])
                ci=interval(bv-av,counts);comps[name+f'/{lam:g}']=ci
                lines.append(f'{name}|{lam:g}|{np.average(av,weights=counts):.4f}|{np.average(bv,weights=counts):.4f}|{ci["query_weighted"]:+.4f}|{ci["equal_family"]:+.4f}')
        output['same_classifier'][panel]=comps
        if len(donors)==nf:
            lines+=['','### Reward/cost head swaps for kNN100','',
                'Lambda | Total Qwen minus GTE | Reward attribution | Cost attribution','---|---:|---:|---:'];decomp={}
            for lam in LAMS[panel]:
                v={key:interval([a['results'][f'{lam:g}']['reward_cost_swap_decomposition'][key] for a in donors],counts) for key in ('total','reward','cost')}
                decomp[f'{lam:g}']=v;lines.append(f'{lam:g}|{v["total"]["query_weighted"]:+.4f}|{v["reward"]["query_weighted"]:+.4f}|{v["cost"]["query_weighted"]:+.4f}')
            output['donor_decomposition'][panel]=decomp
            lines+=['','Attribution averages the two orders of swapping reward and cost estimates. It exactly sums to the utility difference, but is not a causal attribution to training or an encoder mechanism.',
                    '','### Individual neighbors versus their family composition','',
                    'Representation | Lambda | kNN100 | Family-mean replacement | Difference','---|---:|---:|---:|---:']
            for rep in REPS:
                for lam in LAMS[panel]:
                    a=complete['donors/'+rep+'/k100/knn',lam];b=complete['donors/'+rep+'/k100/family_composition',lam]
                    av=np.average([a[g]['utility'] for g in range(nf)],weights=counts);bv=np.average([b[g]['utility'] for g in range(nf)],weights=counts)
                    lines.append(f'{rep}|{lam:g}|{av:.4f}|{bv:.4f}|{av-bv:+.4f}')
            lines+=['','Family means use more outcome observations than k neighbors. Equal-size random-donor controls test whether this benefit is variance reduction; no absence-of-information conclusion follows from this table alone.']
        rr=[json.loads(p.read_text()) for p in sorted((ROOT/'results/random_donors'/panel).glob('fold_*/result.json'))]
        lines+=['',f'Equal-size random-donor folds complete: {len(rr)}/{nf}.']
        if len(rr)==nf:
            summary={}
            lines+=['','Representation | Reference composition | Lambda | kNN minus equal-size random donors','---|---|---:|---:']
            for rep in REPS:
                for ref in ('family','family_length'):
                    for lam in LAMS[panel]:
                        vals=[a['results'][rep+'/'+ref][f'{lam:g}']['knn_minus_random'] for a in rr]
                        summary[rep+'/'+ref+f'/{lam:g}']=interval(vals,counts)
                        lines.append(f'{rep}|{ref}|{lam:g}|{summary[rep+"/"+ref+f"/{lam:g}"]["query_weighted"]:+.4f}')
            output.setdefault('equal_size_donors',{})[panel]=summary
        planned={}
        for kind,seeds,reps in [('kernel',(42,),REPS),('mlp',(42,7),('gte','reader')),('surface',(42,),('text',))]:
            for seed in seeds:
                for rep in reps:
                    planned[f'{kind}/seed{seed}/{rep}']=len(list((ROOT/'results'/kind/f'seed_{seed}'/panel/rep).glob('fold_*/result.json')))
        output.setdefault('planned_status',{})[panel]=planned
        lines+=['','### Planned nonlinear runs','', 'Run | Completed outer families','---|---:']+[f'{name}|{n}/{nf}' for name,n in planned.items()]
        incomplete={k:len(v) for k,v in pool.items() if len(v)<nf}
        lines+=['',f'Incomplete method/lambda entries: {len(incomplete)}. They are retained in summary.json status and not presented as complete comparisons.','']
    output['view_stability_complete']=(ROOT/'results/view_stability/result.json').exists()
    dest=ROOT/'reports';dest.mkdir(exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(output,indent=2));(dest/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Report written; complete comparisons only',flush=True)
    from choice_signal import main as choice_report
    choice_report()

if __name__=='__main__':main()
