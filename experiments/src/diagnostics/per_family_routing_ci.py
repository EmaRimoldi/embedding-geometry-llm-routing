"""Family-level and equal-family paired intervals for frozen OOD kNN-100."""
from pathlib import Path
import csv
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/baselines'))
import common as base
sys.path.insert(0, str(ROOT / 'src/diagnostics/appendix_ci'))
from ci_common import CFG, draws, record, sha, track, write

OUT = ROOT / 'results/diagnostics/per_family_knn_ci/v1'
PANELS = ('routerbench-0shot', 'sprout-sixfam')


def run(panel):
    dataset = base.DATASETS[panel]
    dest = OUT / dataset
    assert not dest.exists(), 'Preserve completed or failed artifacts: ' + str(dest)
    d = base.load(panel)
    n = len(d['r'])
    families = np.asarray(d['meta']['families'])
    lams = json.loads((ROOT / 'configs/baselines.json').read_text())['lambdas'][panel]
    assert len(lams) == 13
    splits = json.loads(base.split_path(panel).read_text())
    names = sorted((s for s in splits if s.startswith('ood_')), key=lambda s: int(s.split('_')[1]))
    assert len(names) == len(families)
    assert n == {'routerbench': 34771, 'sprout': 43887}[dataset]
    previous = ROOT / 'results/diagnostics/appendix_confidence_intervals/routing' / panel / 'ood'
    prior_report = json.loads((previous / 'report.json').read_text())
    assert prior_report['complete'] and prior_report['replicates'] == CFG['replicates'] == 2000
    prov = {}
    for p in [base.split_path(panel), ROOT / 'datasets' / dataset / 'prepared/panel.npz',
              ROOT / 'datasets' / dataset / 'prepared/manifest.json',
              ROOT / 'configs/baselines.json', ROOT / 'configs/appendix_ci.json',
              ROOT / 'src/diagnostics/appendix_ci/ci_common.py',
              ROOT / 'src/baselines/common.py', Path(__file__),
              previous / 'report.json', previous / 'bootstrap.npz']:
        track(p, prov)
    utilities = np.full((n, 2, len(lams)), np.nan, dtype=np.float64)
    seen = np.zeros((n, 2), dtype=np.int8)
    checks = []
    for rep_index, rep in enumerate(('gte', 'reader')):
        for sp in names:
            te = np.asarray(splits[sp]['test'], dtype=np.int64)
            tr = np.asarray(splits[sp]['train'], dtype=np.int64)
            family = splits[sp]['held_out_family']
            assert not np.intersect1d(te, tr).size
            assert np.all(families[d['fam'][te]] == family)
            assert len(np.unique(te)) == len(te)
            p = base.destination('knn', panel, sp, rep)
            report = json.loads((p / 'result.json').read_text())
            assert report['n_test'] == len(te)
            assert report['split_sha256'] == prov[str(base.split_path(panel).relative_to(ROOT))]
            for path in (p / 'result.json', p / 'predictions.npz'):
                track(path, prov)
                rel = str(path.relative_to(ROOT))
                assert prov[rel] == prior_report['source_hashes'][rel], ('Changed original input', rel)
            with np.load(p / 'predictions.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['test_rows'], te)
                np.testing.assert_array_equal(z['query_ids'], d['query_ids'][te])
                for j, lam in enumerate(lams):
                    ch = z[f'{lam:g}|knn100']
                    assert np.issubdtype(ch.dtype, np.integer)
                    assert ch.shape == (len(te),) and ((ch >= 0) & (ch < d['r'].shape[1])).all()
                    u = (d['r'][te] - lam * d['c'][te])[np.arange(len(te)), ch]
                    assert np.isfinite(u).all()
                    old = report['results'][str(lam)]['knn100']
                    assert len(old['per_family']) == 1
                    assert next(iter(old['per_family'].values()))['n'] == len(te)
                    err = abs(float(u.mean()) - old['utility'])
                    assert err < 1e-10, (family, rep, lam, err)
                    utilities[te, rep_index, j] = u
                    checks.append(err)
            seen[te, rep_index] += 1
    assert (seen == 1).all() and np.isfinite(utilities).all()
    # Query identity is the resampling unit. Every penalty stays with its query.
    query_means = utilities.mean(axis=2)
    indices = [np.flatnonzero(d['fam'] == j) for j in range(len(families))]
    sizes = np.array([len(ix) for ix in indices])
    point = np.stack([query_means[ix].mean(axis=0) for ix in indices])
    boots = np.empty((CFG['replicates'], len(families), 2), dtype=np.float64)
    for start, counts in draws(n, CFG['replicates'], CFG['seed'], families=d['fam']):
        for j, ix in enumerate(indices):
            boots[start:start + len(counts), j] = counts[:, ix] @ query_means[ix] / len(ix)
    assert np.isfinite(boots).all()
    macro_point = point.mean(axis=0)
    macro_boot = boots.mean(axis=1)
    pooled_point = (point * sizes[:, None]).sum(axis=0) / n
    pooled_boot = (boots * sizes[None, :, None]).sum(axis=1) / n
    # Exact draw-by-draw audit against the original published query-weighted CI.
    with np.load(previous / 'bootstrap.npz', allow_pickle=False) as old:
        np.testing.assert_array_equal(old['query_ids'], d['query_ids'])
        columns = old['columns'].tolist()
        old_boots = old['draws']
        old_points = old['point']
        original_draw_error = 0.0
        for k, rep in enumerate(('gte', 'reader')):
            ix = [columns.index(f'base|{rep}|knn100|{lam:g}') for lam in lams]
            np.testing.assert_allclose(utilities[:, k].mean(0), old_points[ix], rtol=0, atol=1e-10)
            previous_mean_draws = old_boots[:, ix].mean(1)
            error = np.max(np.abs(previous_mean_draws - pooled_boot[:, k]))
            assert error < 1e-10, ('Original draw mismatch', rep, error)
            original_draw_error = max(original_draw_error, float(error))
    rows = []
    for j, family in enumerate(families):
        rows.append(dict(family=str(family), n_queries=int(sizes[j]),
                         gte=record(point[j, 0], boots[:, j, 0]),
                         l21=record(point[j, 1], boots[:, j, 1]),
                         difference=record(point[j, 1] - point[j, 0], boots[:, j, 1] - boots[:, j, 0])))
    def aggregate_record(p, b):
        return dict(gte=record(p[0], b[:, 0]), l21=record(p[1], b[:, 1]),
                    difference=record(p[1] - p[0], b[:, 1] - b[:, 0]))
    result = dict(dataset=dataset, panel=panel, complete=True, method='kNN-100',
                  representations={'gte': base.REPS['gte'], 'l21': base.REPS['reader']},
                  n_queries=n, penalties=lams, replicates=CFG['replicates'], seed=CFG['seed'],
                  interval='Pointwise 95% percentile, paired query bootstrap within fixed held-out families',
                  scope=CFG['scope'], family_rows=rows,
                  equal_family=aggregate_record(macro_point, macro_boot),
                  query_weighted=aggregate_record(pooled_point, pooled_boot),
                  audited_fold_penalty_scores=len(checks), max_score_error=max(checks),
                  max_original_bootstrap_draw_error=original_draw_error,
                  source_hashes=prov, slurm_job=os.environ.get('SLURM_JOB_ID'))
    dest.mkdir(parents=True)
    np.savez_compressed(dest / 'bootstrap.npz', families=families, n_queries=sizes,
                        means=point, draws=boots, equal_family=macro_boot, query_weighted=pooled_boot,
                        query_ids=d['query_ids'], family_index=d['fam'], query_penalty_means=query_means)
    write(dest / 'report.json', result)
    print('COMPLETE', dataset, json.dumps(result['equal_family']), flush=True)
    return result


def main():
    results = [run(panel) for panel in PANELS]
    fields = ['dataset', 'family', 'n_queries'] + [f'{m}_{s}' for m in ('gte', 'l21', 'difference') for s in ('mean', 'low', 'high')]
    with (OUT / 'per_family_knn100_ci.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for row in result['family_rows'] + [dict(family='Equal-family average', n_queries=result['n_queries'], **result['equal_family'])]:
                writer.writerow(dict(dataset=result['dataset'], family=row['family'], n_queries=row['n_queries'],
                                     **{f'{m}_{s}': row[m][s] for m in ('gte', 'l21', 'difference') for s in ('mean', 'low', 'high')}))
    write(OUT / 'report.json', dict(complete=True, datasets=results))


if __name__ == '__main__':
    main()
