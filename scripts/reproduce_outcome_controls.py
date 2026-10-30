#!/usr/bin/env python3
"""Recompute all single-model/oracle controls from freshly prepared public outcomes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/fresh-outcome-controls.json')
    args = parser.parse_args()
    with (ROOT / 'reproducibility/evidence/baselines/_summary_reduced/per_split.csv').open() as stream:
        reference = {(r['dataset'], r['split'], r['encoder'], r['method'], float(r['lambda_value'])): r
                     for r in csv.DictReader(stream) if r['method'] in ['Best single model', 'Realized oracle']}
    config = json.loads((ROOT / 'configs/baselines.json').read_text())
    original = json.loads((ROOT / 'reproducibility/provenance/reviewer_smoke/public_data_preparation.json').read_text())
    results, verified_inputs, comparisons = [], [], 0
    max_error = 0.
    for panel, dataset in [('routerbench-0shot', 'routerbench'), ('sprout-sixfam', 'sprout')]:
        folder = ROOT / 'datasets' / dataset / 'prepared'
        for filename, expected in original['datasets'][dataset]['file_hashes'].items():
            actual = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
            if actual != expected['canonical']:
                raise ValueError('Prepared input differs from frozen canonical bytes: ' + dataset + '/' + filename)
            verified_inputs.append({'dataset': dataset, 'file': filename, 'sha256': actual})
        with np.load(folder / 'panel.npz', allow_pickle=False) as data:
            reward, cost = data['reward'], data['cost']
        splits = json.loads((ROOT / 'configs/baseline_splits' / (panel + '.json')).read_text())
        for split, rows in splits.items():
            train, test = np.asarray(rows['train']), np.asarray(rows['test'])
            if np.intersect1d(train, test).size or len(train) + len(test) != len(reward):
                raise ValueError('Invalid outer partition: ' + panel + '/' + split)
            for penalty in config['lambdas'][panel]:
                best = (reward[train] - penalty * cost[train]).mean(0).argmax()
                choices = {'Best single model': np.full(len(test), best),
                           'Realized oracle': (reward[test] - penalty * cost[test]).argmax(1)}
                for method, choice in choices.items():
                    r, c = reward[test, choice], cost[test, choice]
                    scores = {'utility': float((r - penalty * c).mean()), 'reward': float(r.mean()), 'cost': float(c.mean())}
                    for encoder in ['gte', 'reader']:
                        saved = reference[panel, split, encoder, method, float(penalty)]
                        for metric, value in scores.items():
                            error = abs(value - float(saved[metric]))
                            max_error = max(max_error, error)
                            if not math.isclose(value, float(saved[metric]), rel_tol=1e-10, abs_tol=1e-12):
                                raise ValueError(f'Fresh {panel}/{split}/{method}/{penalty}/{metric} differs: {error}')
                            comparisons += 1
                    results.append(dict(panel=panel, split=split, penalty=penalty, method=method, **scores))
    report = {'passed': True, 'verified_canonical_inputs': verified_inputs, 'fresh_method_scores': len(results),
              'saved_metric_comparisons': comparisons, 'maximum_absolute_error': max_error, 'results': results,
              'scope': 'Actual recomputation from downloaded public outcomes and frozen splits. No embedding or fitted router required. Oracle uses test outcomes only as an upper bound.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['verified_canonical_inputs', 'results']}))


if __name__ == '__main__':
    main()
