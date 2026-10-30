import ast
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    path = ROOT / 'src/diagnostics/appendix_ci/family_rank.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'summarize')
    namespace = {'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace['summarize']


def test_bootstrap_top1_uses_the_point_estimators_average_rank():
    rank, top = summarize()(np.array([[[0.], [1.]]]), np.array([[.5, .5]]))
    np.testing.assert_array_equal(rank, [[1.5]])
    np.testing.assert_array_equal(top, [[0.]])


def test_all_169_observed_family_penalty_points_are_unchanged():
    base = ROOT / 'results/evidence/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer'
    grouped = defaultdict(list)
    with (base / 'source_errors.csv').open() as stream:
        for row in csv.DictReader(stream):
            grouped[row['dataset'], row['heldout_family'], float(row['lambda_value'])].append(row)
    with (base / 'per_family_ranks.csv').open() as stream:
        expected = {(r['dataset'], r['heldout_family'], float(r['lambda_value'])): r for r in csv.DictReader(stream)}
    assert len(grouped) == len(expected) == 169
    function = summarize()
    for key, rows in grouped.items():
        mse = np.array([float(r['relative_utility_mse']) for r in rows])[None, :, None]
        shares = np.array([float(r['retrieval_share']) for r in rows])[None, :]
        rank, top = function(mse, shares)
        np.testing.assert_allclose(rank[0, 0], float(expected[key]['retrieval_rank']), atol=1e-12)
        np.testing.assert_allclose(top[0, 0], float(expected[key]['top1']), atol=1e-12)
