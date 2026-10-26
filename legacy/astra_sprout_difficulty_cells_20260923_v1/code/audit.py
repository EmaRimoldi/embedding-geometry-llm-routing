"""Independent direct checks of saved neighbours and outcome denominators."""
import json
import numpy as np
from run import ROOT, DEFINITIVE, load, compress, sha

def main():
    d=load('sprout-sixfam','gte');x=d['x'];r=d['r'];f=d['fam']
    y=np.column_stack([(r>=.5).mean(1),r.mean(1),r,(r>=.25).mean(1),(r>=.75).mean(1)])
    cells=np.load(DEFINITIVE/'inputs/sprout_cells_native_openhermes_v1/cells.npz')['cells']
    splits=json.loads((DEFINITIVE/'inputs/sprout-sixfam/splits.json').read_text())
    rng=np.random.default_rng(933);records=[]
    for seed in range(5):
        z=np.load(ROOT/'results'/f'seed{seed}'/'query_metrics.npz')
        te=z['test_rows'];tr=np.asarray(splits[f'iid_{seed}']['train'])
        assert np.array_equal(te,splits[f'iid_{seed}']['test'])
        assert np.array_equal(z['query_ids'],d['query_ids'][te])
        samples=np.concatenate([rng.choice(np.flatnonzero(f[te]==g),2,replace=False) for g in np.unique(f)])
        for name in ['global','family','approved_cell','approved_min50','approved_cos090']:
            for pos in samples:
                q=te[pos]
                if name=='global':pool=tr
                elif name=='family':pool=tr[f[tr]==f[q]]
                else:
                    pool=tr[cells[tr]==cells[q]]
                    if len(pool)<(50 if name=='approved_min50' else 1):pool=tr[f[tr]==f[q]]
                sim=x[q].astype(float)@x[pool].astype(float).T
                if name=='approved_cos090':
                    # If a score is within numerical epsilon of the threshold,
                    # the fixture is ambiguous; disclose rather than hiding it.
                    assert np.min(np.abs(sim-.9))>1e-7
                    pool=pool[sim<=.9];sim=sim[sim<=.9]
                    if len(pool)<10:continue
                assert q not in pool
                assert len(pool)==z[name+'__pool_n'][pos]
                exact=compress(np.abs(y[q]-y[pool]).mean(0))
                assert np.allclose(exact,z[name+'__base_abs'][pos],atol=1e-10)
                saved=z[name+'__nn10'][pos];saved=saved[saved>=0]
                assert len(np.unique(saved))==len(saved) and np.isin(saved,pool).all()
                selected=x[q].astype(float)@x[saved].astype(float).T
                threshold=np.sort(sim)[-len(saved)]
                assert selected.min()>=threshold-1e-6
                assert np.allclose(compress(np.abs(y[q]-y[saved]).mean(0)),z[name+'__knn10_abs'][pos],atol=1e-10)
                assert np.allclose(compress((y[q]-y[saved].mean(0))**2),z[name+'__knn10_mse'][pos],atol=1e-10)
                records.append(dict(seed=seed,pool=name,query=int(q),n_donors=len(pool),passed=True))
    lock=json.loads((DEFINITIVE/'config/SPROUT_CANONICAL_PROTOCOL.json').read_text())
    for path,expected in lock['file_sha256'].items():assert sha(path)==expected,path
    report=dict(passed=True,checked_query_pool_cases=len(records),records=records,
                checks=['frozen query identity','train/test separation','correct full-family fallback',
                        'exact absolute-difference baseline','same cosine-filtered comparison pool',
                        'float64 top-k boundary within 1e-6','saved neighbour errors','canonical input hashes'])
    out=ROOT/'reports';out.mkdir(exist_ok=True);(out/'AUDIT.json').write_text(json.dumps(report,indent=2))
    print('PASS independent saved-output audit:',len(records),'query/pool cases',flush=True)

if __name__=='__main__':main()
