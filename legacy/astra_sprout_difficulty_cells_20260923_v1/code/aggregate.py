"""Aggregate all five prespecified splits; never select a favourable resolution."""
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
N=43887

def main():
    summaries=[]; sums={}; counts={}; family=np.full(N,-1,int); query_id=np.full(N,'',dtype='<U128')
    for seed in range(5):
        p=ROOT/'results'/f'seed{seed}'; meta=json.loads((p/'summary.json').read_text())
        summaries.append(meta); z=np.load(p/'query_metrics.npz'); rr=z['test_rows']
        known=family[rr]>=0
        assert np.array_equal(family[rr][known],z['family'][known])
        family[rr]=z['family']; query_id[rr]=z['query_ids']
        for k in z.files:
            if k.endswith(('_abs','_mse')) or k.startswith(('global10_matched','global100_matched')):
                a=z[k];assert a.shape==(len(rr),5)
                if k not in sums:sums[k]=np.zeros((N,5));counts[k]=np.zeros((N,5),int)
                valid=np.isfinite(a);sums[k][rr]+=np.where(valid,a,0);counts[k][rr]+=valid
    assert len({s['canonical_protocol_sha256'] for s in summaries})==1
    assert len({s['code_sha256'] for s in summaries})==1
    assert all(s['input_hashes']==summaries[0]['input_hashes'] for s in summaries)
    observed=np.flatnonzero(family>=0); f=family[observed]
    arrays={k:np.divide(v,counts[k],out=np.full_like(v,np.nan),where=counts[k]>0)[observed] for k,v in sums.items()}
    keys=sorted(arrays); cols={k:slice(5*i,5*i+5) for i,k in enumerate(keys)}
    values=np.concatenate([arrays[k] for k in keys],1);valid=np.isfinite(values)
    clean=np.where(valid,values,0); rng=np.random.default_rng(20260923)
    boot=[]; boot_family=[]
    # Query-ID clustered, family-stratified bootstrap. Repeated split records
    # were first averaged per ID, so no seed replication is treated as new data.
    for _ in range(20):
        num=np.zeros((50,values.shape[1]));den=np.zeros_like(num); family_chunk=[]
        for g in np.unique(f):
            ix=np.flatnonzero(f==g)
            weights=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=50).astype(float)
            gn=weights@clean[ix];gd=weights@valid[ix]
            num+=gn;den+=gd
            family_chunk.append(np.divide(gn,gd,out=np.full_like(gn,np.nan),where=gd>0))
        boot.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
        boot_family.append(np.stack(family_chunk,axis=1))
    bs=np.concatenate(boot);bf=np.concatenate(boot_family);point=np.nanmean(values,0)
    def pack(estimate,replicas):
        return dict(estimate=np.asarray(estimate).tolist(),ci95=np.nanpercentile(replicas,[2.5,97.5],axis=0).T.tolist())
    def compare(base,new):
        a,b=cols[base],cols[new]
        p=100*(1-point[b]/point[a]);bb=100*(1-bs[:,b]/bs[:,a])
        famres={}
        for g,name in enumerate(summaries[0]['families']):
            va,vb=arrays[base][f==g],arrays[new][f==g]
            shared=np.isfinite(va)&np.isfinite(vb)
            ma=np.sum(np.where(shared,va,0),axis=0)/shared.sum(0)
            mb=np.sum(np.where(shared,vb,0),axis=0)/shared.sum(0)
            famres[name]=(100*(1-mb/ma)).tolist()
        fb=100*(1-bf[:,:,b]/bf[:,:,a])
        return dict(**pack(p,bb),baseline_mean=point[a].tolist(),new_mean=point[b].tolist(),
                    per_family=famres,per_family_ci95={name:np.nanpercentile(fb[:,g],[2.5,97.5],axis=0).T.tolist() for g,name in enumerate(summaries[0]['families'])},
                    equal_family=np.mean(list(famres.values()),0).tolist(),
                    equal_family_ci95=np.nanpercentile(np.nanmean(fb,axis=1),[2.5,97.5],axis=0).T.tolist())
    variants=sorted({k.split('__')[0] for k in keys if '__' in k})
    local={};prediction={}
    for name in variants:
        for k in (10,100):
            if f'{name}__knn{k}_abs' not in arrays:continue
            local[f'{name}_k{k}']=compare(f'{name}__base_abs',f'{name}__knn{k}_abs')
            prediction[f'{name}_k{k}_vs_own_mean']=compare(f'{name}__base_mse',f'{name}__knn{k}_mse')
    prediction['family_vs_global']=compare('global__base_mse','family__base_mse')
    prediction['cell_vs_family']=compare('family__base_mse','approved_cell__base_mse')
    prediction['cell_vs_global']=compare('global__base_mse','approved_cell__base_mse')
    for k in (2,4,8):prediction[f'clusters{k}_vs_cell']=compare('approved_cell__base_mse',f'cell_clusters{k}__base_mse')
    decomposition={}
    for k in (10,100):
        chain=['global__base_abs',f'global{k}_matched_family',f'global{k}_matched_cell',f'global{k}_matched_cluster8',f'global__knn{k}_abs']
        terms=['family_composition','cell_beyond_family','clusters_beyond_cell','remaining_selection']
        total=point[cols[chain[0]]]-point[cols[chain[-1]]]
        tb=bs[:,cols[chain[0]]]-bs[:,cols[chain[-1]]]
        record={'total_absolute_advantage':pack(total,tb),'terms':{}}
        check=np.zeros(5)
        for label,left,right in zip(terms,chain[:-1],chain[1:]):
            diff=point[cols[left]]-point[cols[right]]; bd=bs[:,cols[left]]-bs[:,cols[right]]
            check+=diff
            record['terms'][label]=dict(absolute=pack(diff,bd),share_percent=pack(100*diff/total,100*bd/tb))
        assert np.allclose(check,total,atol=1e-12)
        for label,key in [('family',chain[1]),('cell',chain[2]),('cluster8',chain[3])]:
            diff=point[cols[chain[0]]]-point[cols[key]];bd=bs[:,cols[chain[0]]]-bs[:,cols[key]]
            record[f'cumulative_share_{label}']=pack(100*diff/total,100*bd/tb)
        decomposition[f'k{k}']=record
    summary=dict(canonical_protocol_sha256=summaries[0]['canonical_protocol_sha256'],
                 aggregation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 code_sha256=summaries[0]['code_sha256'],proxies=summaries[0]['proxies'],
                 n_unique_test_queries=len(observed),splits=5,
                 aggregation='Mean per unique query across its held-out appearances, then equal query weights',
                 uncertainty='1000 query-ID clustered bootstraps stratified by family; conditional on fits and labels',
                 locality_percent_reduction=local,prediction_mse_percent_reduction=prediction,
                 donor_composition=decomposition,
                 per_split=[dict(seed=s['seed'],metrics=s['metrics'],permutation=s['permutation']) for s in summaries])
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    np.savez_compressed(out/'unique_query_metrics.npz',rows=observed,family=f,query_ids=query_id[observed],**arrays)
    def fmt(v,i=0):
        a=v['estimate'][i];lo,hi=v['ci95'][i];return f'{a:.2f} [{lo:.2f}, {hi:.2f}]'
    md=['# Canonical SPROUT: difficulty and grouping','',f'All five fixed splits complete; {len(observed):,} unique test queries.','',
        'Primary columns below: solve fraction (score >= .5), mean continuous reward, full reward profile.','',
        '## Locality: reduction in absolute outcome difference relative to random donors from the same pool','',
        '| Pool and k | Solve fraction | Mean reward | Reward profile |','|---|---:|---:|---:|']
    for name,v in local.items():md.append('| '+name+' | '+' | '.join(fmt(v,i) for i in range(3))+' |')
    md+=['','## Held-out prediction: MSE reduction relative to the named baseline','','| Comparison | Solve fraction | Mean reward | Reward profile |','|---|---:|---:|---:|']
    for name,v in prediction.items():md.append('| '+name+' | '+' | '.join(fmt(v,i) for i in range(3))+' |')
    md+=['','## Fraction of unrestricted neighbour advantage accounted for by donor composition','',
         '| k | Conditioning | Solve fraction | Mean reward | Reward profile |','|---|---|---:|---:|---:|']
    for k,rec in decomposition.items():
        for cond in ['family','cell','cluster8']:
            md.append('| '+k+' | '+cond+' | '+' | '.join(fmt(rec[f'cumulative_share_{cond}'],i) for i in range(3))+' |')
    md+=['','## Approved-cell k=10 by family','','| Family | Solve fraction | Mean reward | Reward profile |','|---|---:|---:|---:|']
    for name,v in local['approved_cell_k10']['per_family'].items():md.append('| '+name+' | '+' | '.join(f'{x:.2f}' for x in v[:3])+' |')
    md+=['','All fixed controls and split-wise results are retained in summary.json. Cluster labels are outcome-free geometry groups, not certified topics. A composition decomposition is descriptive, not causal. Older scripts pooled only rare cells in their nominal family fallback; this implementation uses the full family.']
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md),flush=True)

if __name__=='__main__':main()
