"""Independent saved-output arithmetic and donor checks. No experiment edits."""
import json
import numpy as np
from run import ROOT, DEF, load, sha

def main():
    records = []
    rng = np.random.default_rng(20260924)
    for panel in ['sprout-sixfam', 'routerbench-0shot']:
        d = load(panel, 'gte')
        x, r, c, fam = (d[k] for k in ['x', 'r', 'c', 'fam'])
        cellpath = DEF / ('inputs/sprout_cells_native_openhermes_v1/cells.npz' if panel == 'sprout-sixfam' else 'cell_lookup_v2/routerbench-0shot/cells.npz')
        cells = np.load(cellpath)['cells']
        splits = json.loads((DEF/'inputs'/panel/'splits.json').read_text())
        report = json.loads((ROOT/'reports'/f'{panel}.json').read_text())
        totals = {}
        for seed in range(5):
            folder = ROOT/'results'/panel/f'iid_{seed}'
            meta = json.loads((folder/'result.json').read_text())
            z = np.load(folder/'predictions.npz')
            tr = np.asarray(splits[f'iid_{seed}']['train'])
            te = np.asarray(splits[f'iid_{seed}']['test'])
            assert np.array_equal(te, z['test_rows'])
            assert np.array_equal(d['query_ids'][te], z['query_ids'])
            assert not np.intersect1d(tr, te).size
            assert meta['code_sha256'] == sha(ROOT/'code/run.py')
            for path, expected in meta['input_hashes'].items():
                assert sha(path) == expected, path
            samples = np.concatenate([rng.choice(np.flatnonzero(fam[te] == f), min(2, np.sum(fam[te] == f)), replace=False) for f in np.unique(fam[te])])
            for name, labels, nnkey in [('cell', cells, 'cell_neighbors100'), ('group8', z['groups8'], 'group8_neighbors100')]:
                nn = z[nnkey]
                assert np.isin(nn[nn >= 0], tr).all()
                for pos in samples:
                    q = te[pos]
                    pool = tr[labels[tr] == labels[q]]
                    if not len(pool):
                        pool = tr[fam[tr] == fam[q]]
                    saved = nn[pos][nn[pos] >= 0]
                    assert len(saved) == min(100, len(pool))
                    assert len(np.unique(saved)) == len(saved)
                    assert np.isin(saved, pool).all()
                    sims = x[pool].astype(float) @ x[q].astype(float)
                    selected = x[saved].astype(float) @ x[q].astype(float)
                    assert selected.min() >= np.partition(sims, -len(saved))[-len(saved)] - 1e-6
                    for k in [10, 100]:
                        assert np.allclose(z[f'{name}_knn{k}|r'][pos], r[saved[:k]].mean(0), atol=1e-10)
                        assert np.allclose(z[f'{name}_knn{k}|c'][pos], c[saved[:k]].mean(0), atol=1e-10)
            ii = np.arange(len(te))
            for rec in meta['results']:
                method, lam = rec['method'], rec['lambda_value']
                key = f'{method}|{lam:g}'
                actual = r[te] - lam*c[te]
                if method == 'random_cell100':
                    u = np.mean([actual[ii, z[f'random_cell100_draw{draw}|{lam:g}|choice']] for draw in range(20)], axis=0)
                else:
                    rr, cc = z[method+'|r'], z[method+'|c']
                    choice = (rr-lam*cc).argmax(1)
                    assert np.array_equal(choice, z[key+'|choice'])
                    u = actual[ii, choice]
                    assert np.allclose(z[key+'|reward_part'] + z[key+'|cost_part'], z[key+'|gain'], atol=1e-12)
                assert np.allclose(u, z[key+'|utility'], atol=1e-12)
                assert abs(u.mean()-rec['utility']) < 1e-10
                assert np.allclose(z[key+'|gain'], u-z[f'cell_mean|{lam:g}|utility'], atol=1e-12)
                totals.setdefault(key, []).append(float(u.mean()))
            # Reconstruct every reported contrast from the saved query outcomes.
            for key, stat in report['statistics'].items():
                if not key.startswith('contrast:'):
                    continue
                pair, lam = key[len('contrast:'):].split('|')
                left, right = pair.split('-minus-')
                val = (z[f'{left}|{lam}|utility'] - z[f'{right}|{lam}|utility']).mean()
                assert abs(val - stat['split_values'][seed]) < 1e-10
            records.append(dict(panel=panel, seed=seed, query_pool_cases=int(len(samples)*2), passed=True))
            print('PASS', panel, seed, flush=True)
        for key, values in totals.items():
            assert abs(np.mean(values) - report['statistics'][key+'|utility']['estimate']) < 1e-10
    lock = DEF/'config/SPROUT_CANONICAL_PROTOCOL.json'
    for path, expected in json.loads(lock.read_text())['file_sha256'].items():
        assert sha(path) == expected
    out = dict(passed=True, splits=len(records), query_pool_cases=sum(v['query_pool_cases'] for v in records), records=records,
               audit_code_sha256=sha(__file__), canonical_sprout_lock_sha256=sha(lock),
               checks=['input hashes and fixed row identity', 'all neighbours are training rows', 'sampled pool eligibility and float64 top-100 boundaries', 'sampled neighbour reward/cost averages', 'all saved policy utilities and random-draw averaging', 'all contrast point estimates', 'reward/cost attribution arithmetic'],
               limitations=['Intervals are conditional pointwise query-ID bootstraps, not simultaneous or retraining uncertainty.', 'Random donor eligibility is guaranteed by inspected generation code; donor IDs were not stored.', 'RouterBench cell minimum is one, unlike historical minimum-50 comparisons.'])
    (ROOT/'reports/AUDIT.json').write_text(json.dumps(out, indent=2))
    print('ALL PASS', len(records), flush=True)

if __name__ == '__main__':
    main()
