"""All-split paired uncertainty, retaining the complete fixed comparison grid."""
from pathlib import Path
import json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PAIRS=[('global_knn100','cell_mean'),('cell_knn100','cell_mean'),
       ('global_knn100','cell_knn100'),('cell_knn100','random_cell100'),
       ('global_knn100','random_cell100'),('group2_mean','cell_mean'),
       ('group4_mean','cell_mean'),('group8_mean','cell_mean'),
       ('cell_knn100','group8_mean'),('group8_knn100','group8_mean'),
       ('global_knn10','cell_knn10'),('cell_knn10','group8_mean')]

def summarize(panel,nrows):
    sums={};counts={};fam=np.full(nrows,-1,int);split_values={};metas=[]
    for seed in range(5):
        folder=ROOT/'results'/panel/f'iid_{seed}';m=json.loads((folder/'result.json').read_text());metas.append(m)
        z=np.load(folder/'predictions.npz');te=z['test_rows'];fam[te]=z['family'];a={}
        for key in z.files:
            if key.endswith(('|utility','|gain','|agreement','|reward_part','|cost_part','|pair_accuracy','|mean_reward_mse','|centered_reward_mse')):
                a[key]=z[key];assert a[key].shape==(len(te),)
        for lam in m['lambdas']:
            for left,right in PAIRS:
                a[f'contrast:{left}-minus-{right}|{lam:g}']=z[f'{left}|{lam:g}|utility']-z[f'{right}|{lam:g}|utility']
        for key,v in a.items():
            if key not in sums:sums[key]=np.zeros(nrows);counts[key]=np.zeros(nrows,int);split_values[key]=[]
            ok=np.isfinite(v);sums[key][te]+=np.where(ok,v,0);counts[key][te]+=ok;split_values[key].append(float(np.nanmean(v)))
    assert len({m['code_sha256'] for m in metas})==1
    assert len({m['protocol_sha256'] for m in metas})==1
    if panel=='sprout-sixfam':assert len({m['canonical_sprout_lock_sha256'] for m in metas})==1
    keys=sorted(sums);observed=np.flatnonzero(fam>=0);f=fam[observed]
    s=np.stack([sums[k][observed] for k in keys],1);c=np.stack([counts[k][observed] for k in keys],1)
    pt=s.sum(0)/c.sum(0);bootstrap=[];rng=np.random.default_rng(7823)
    for block in range(20):
        num=np.zeros((50,len(keys)));den=np.zeros_like(num)
        for g in np.unique(f):
            ix=np.flatnonzero(f==g);w=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=50).astype(float)
            num+=w@s[ix];den+=w@c[ix]
        bootstrap.append(np.divide(num,den,out=np.full_like(num,np.nan),where=den>0))
    bs=np.concatenate(bootstrap);ci=np.nanpercentile(bs,[2.5,97.5],axis=0)
    stats={}
    for j,key in enumerate(keys):
        # Equal outer split sizes make pooled appearances equal the five-split
        # mean except metrics whose coverage varies because of observed ties.
        if '|pair_accuracy' not in key:assert abs(pt[j]-np.mean(split_values[key]))<1e-10
        stats[key]=dict(estimate=float(pt[j]),ci95=ci[:,j].tolist(),split_values=split_values[key],
                       positive_splits=int(np.sum(np.array(split_values[key])>0)),
                       per_family={metas[0]['families'][g]:float(s[f==g,j].sum()/c[f==g,j].sum()) for g in np.unique(f)})
    # Preserve conditional disagreement utility as its own denominator.
    conditional={};ix={k:j for j,k in enumerate(keys)}
    for key in keys:
        if not key.endswith('|gain'):continue
        agreement=key[:-5]+'|agreement'
        if agreement not in ix:continue
        a,b=ix[key],ix[agreement]
        if 1-pt[b]<1e-12:continue
        vv=bs[:,a]/np.maximum(1-bs[:,b],1e-12)
        conditional[key[:-5]]=dict(estimate=float(pt[a]/(1-pt[b])),ci95=np.percentile(vv,[2.5,97.5]).tolist())
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    report=dict(panel=panel,splits=5,unique_test_queries=len(observed),families=metas[0]['families'],lambdas=metas[0]['lambdas'],
                primary='Mean of five equally sized test splits; each test appearance has equal weight',
                uncertainty='1000 family-stratified query-ID bootstraps preserve all repeated appearances; conditional pointwise intervals',
                methods=sorted({v['method'] for v in metas[0]['results']}),statistics=stats,conditional_disagreement=conditional,
                source_code_sha256=metas[0]['code_sha256'],aggregation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                canonical_sprout_lock_sha256=metas[0]['canonical_sprout_lock_sha256'],checks='All five splits loaded; source hashes uniform; primary estimates match split means; run-level baseline/attribution checks passed')
    (out/f'{panel}.json').write_text(json.dumps(report,indent=2))
    md=[f'# IID geometry and routing: {panel}','','All five fixed splits complete. Positive contrasts favour the first method. Intervals are paired and pointwise.','',
        '| Comparison | Lambda | Utility difference | 95% CI | Positive splits |','|---|---:|---:|---|---:|']
    for left,right in PAIRS:
        for lam in report['lambdas']:
            v=stats[f'contrast:{left}-minus-{right}|{lam:g}'];lo,hi=v['ci95']
            md.append(f"| {left} vs {right} | {lam:g} | {v['estimate']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {v['positive_splits']}/5 |")
    md+=['','All per-family outcomes, reward/cost attribution, model-order accuracy, centered reward errors and disagreement gains are in the JSON report. No method, resolution or lambda was selected on test utility.']
    (out/f'{panel}.md').write_text('\n'.join(md)+'\n');print('DONE SUMMARY',panel,len(keys),flush=True)

if __name__=='__main__':
    summarize('sprout-sixfam',43887)
    summarize('routerbench-0shot',34661)
