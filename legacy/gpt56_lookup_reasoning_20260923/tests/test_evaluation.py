from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import paired_stats, existing_frontier_functions, campaign_baselines, read_jsonl


def test_exact_paired_randomization_and_zero_difference():
    r = paired_stats(np.array([1., 1.]), [True, True], bootstrap=100)
    assert r['delta_utility'] == 1
    assert r['randomization_p'] == .5
    assert r['ci95_low'] == r['ci95_high'] == 1
    assert r['improved_given_change'] == 1
    z = paired_stats(np.zeros(7), np.zeros(7, bool), bootstrap=100)
    assert z['randomization_p'] == 1
    assert z['changed_fraction'] == 0
    assert z['improved_given_change'] is None


def test_repository_hull_aiq_on_known_geometry():
    source = Path(__file__).resolve().parents[1] / 'data/campaign_code/frontier_metrics.py'
    if not source.exists():
        import pytest
        pytest.skip('Requires exported frozen repository metric implementation')
    hull, aiq = existing_frontier_functions(source)
    x,y = hull([0.,1.,2.,3.],[0.,.1,.8,1.])
    assert np.array_equal(x,[0,2,3])
    assert np.allclose(y,[0,.8,1.])
    scores=aiq({'diagonal':(np.array([0.,1.]),np.array([0.,1.])),
                'perfect':(np.array([0.,1.]),np.array([1.,1.]))})
    assert np.isclose(scores['diagonal'],.5)
    assert np.isclose(scores['perfect'],1)


def test_archived_baselines_share_approved_alias_mask(tmp_path):
    import json
    import pytest
    ids = np.array(['train_alias', 'keep1', 'keep2'])
    meta = {'split_sha256': 'frozen'}
    for kind, fields in [('knn', {'0|knn100': [2, 1, 0], '0|knn_validated': [0, 1, 2]}),
                         ('prior', {'0|soft_prior': [1, 2, 0]}),
                         ('mlp_gpu/lambda_0', {'choice': [2, 0, 1]})]:
        folder = tmp_path / 'baselines' / kind
        folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps(meta))
        np.savez(folder/'predictions.npz', query_ids=ids, **fields)
    manifest = {'stage': 'test', 'source_split_sha256': 'frozen',
                'outer_test_ids': ids.tolist(), 'excluded_test_alias_ids': ['train_alias']}
    results, _ = campaign_baselines(tmp_path, manifest, ['keep1', 'keep2'], [0])
    assert results['campaign_knn100'][0].tolist() == [1, 0]
    assert results['campaign_knn_validated'][0].tolist() == [1, 2]
    assert results['campaign_soft_prior'][0].tolist() == [2, 0]
    assert results['campaign_mlp_utility_validated'][0].tolist() == [0, 1]
    with pytest.raises(ValueError, match='exclusions'):
        campaign_baselines(tmp_path, manifest, ['keep1'], [0])


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    import json
    path = tmp_path/'queries.jsonl'
    row = {'query': 'line1\u2028line2'}
    path.write_text(json.dumps(row, ensure_ascii=False)+'\n')
    assert read_jsonl(path) == [row]
