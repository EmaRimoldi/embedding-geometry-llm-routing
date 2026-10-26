import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (InferenceQuery, TrainingLookup, canonical, choose, digest,
                  load_config, make_evidence, output_schema, parse_prediction, validate_split)
from codex_backend import thread_params, turn_params


@pytest.fixture
def fixture():
    q = InferenceQuery('test-query', 'What is 2 + 3?', 'math', 'arithmetic')
    tr = TrainingLookup([InferenceQuery(f'train-{i}', f'Calculate {i}+2.', 'math', '') for i in range(3)],
                        np.array([[1., 0.], [.5, 1.], [0., 1.]]), np.array([[.1, .02]] * 3), ['A', 'B'])
    cfg = load_config()
    return q, tr, cfg


def test_no_query_overlap_and_grouped_pairs():
    ids = np.array(['a', 'b', 'c', 'd'])
    s = dict(train=[0, 1, 2], test=[3], inner=[dict(fit=[0, 1], val=[2])])
    validate_split(ids, s)
    # Every candidate for a query receives its query-level partition.
    train_pairs = {(ids[i], m) for i in s['train'] for m in ['A', 'B']}
    test_pairs = {(ids[i], m) for i in s['test'] for m in ['A', 'B']}
    assert {q for q, _ in train_pairs}.isdisjoint(q for q, _ in test_pairs)
    with pytest.raises(ValueError):
        validate_split(ids, dict(train=[0, 1, 3], test=[3], inner=[]))
    with pytest.raises(ValueError):
        validate_split(['a', 'a', 'b', 'c'], s)
    with pytest.raises(ValueError):
        validate_split(ids, dict(train=[0, 1, 2], test=[3], inner=[dict(fit=[0, 1], val=[1, 2])]))


def test_inference_query_cannot_contain_test_outcomes(fixture):
    q, tr, cfg = fixture
    with pytest.raises(TypeError):
        InferenceQuery(**vars(q), true_label='SECRET', cost=.01)
    e = make_evidence(q, tr, [0, 1, 2], [.9, .8, .7], cfg)
    forbidden = ['model_response', 'true_label', 'test_reward', 'realized_cost', 'oracle']
    assert all(key not in e['prompt'] for key in forbidden)
    # Unused/evaluator-only outcomes cannot influence a builder with no access to them.
    external_test_outcomes = {'r': [1, 0], 'c': [3, 4], 'gold': 'SECRET'}
    before = e['prompt_hash']
    external_test_outcomes.update(r=[0, 1], c=[90, 0], gold='POISON')
    assert make_evidence(q, tr, [0, 1, 2], [.9, .8, .7], cfg)['prompt_hash'] == before


def test_self_and_duplicate_text_excluded(fixture):
    q, tr, cfg = fixture
    tr.queries[0] = InferenceQuery(q.sample_id, 'different text', '', '')
    with pytest.raises(ValueError):
        make_evidence(q, tr, [0], [.9], cfg)
    tr.queries[0] = InferenceQuery('other-id', '  What is 2 + 3?\n', '', '')
    with pytest.raises(ValueError):
        make_evidence(q, tr, [0], [.9], cfg)


def test_same_inputs_only_effort_differs(fixture, tmp_path):
    q, tr, cfg = fixture
    e = make_evidence(q, tr, [0, 1, 2], [.9, .8, .7], cfg)
    a = thread_params(cfg['model'], 'low', 'default', tmp_path, 'shared instructions')
    b = thread_params(cfg['model'], 'xhigh', 'default', tmp_path, 'shared instructions')
    assert a['config'].pop('model_reasoning_effort') == 'low'
    assert b['config'].pop('model_reasoning_effort') == 'xhigh'
    assert a == b
    a = turn_params('test-session', e['prompt'], output_schema(tr.models), cfg['model'], 'low', 'default')
    b = turn_params('test-session', e['prompt'], output_schema(tr.models), cfg['model'], 'xhigh', 'default')
    a.pop('effort'); b.pop('effort')
    assert canonical(a) == canonical(b)
    assert a['environments'] == [] and a['runtimeWorkspaceRoots'] == []
    assert 'previous_response_id' not in a


def test_schema_and_unknown_duplicate_models(fixture):
    q, tr, cfg = fixture
    valid = dict(sample_id=q.sample_id, model_estimates=[dict(model_name=m, expected_performance=.5) for m in tr.models])
    assert parse_prediction(json.dumps(valid), q.sample_id, tr.models) == valid
    for names in [['A', 'C'], ['A', 'A'], ['B', 'A']]:
        d = copy.deepcopy(valid)
        for row, name in zip(d['model_estimates'], names):
            row['model_name'] = name
        with pytest.raises(ValueError):
            parse_prediction(json.dumps(d), q.sample_id, tr.models)
    for value in [True, float('nan'), 1.1, -.1, '0.5']:
        d = copy.deepcopy(valid); d['model_estimates'][0]['expected_performance'] = value
        with pytest.raises(ValueError):
            parse_prediction(json.dumps(d), q.sample_id, tr.models)


def test_campaign_utility_and_ties():
    assert choose([.9, .8], [.1, .001], 0) == 0
    assert choose([.9, .8], [.1, .001], 10) == 1
    # Exactly tied utility: current campaign takes first column, not cheapest.
    assert choose([.9, .8], [.2, .1], 1) == np.argmax(np.array([.9, .8]) - np.array([.2, .1]))
    assert choose([1., 0.], [1., 0.], 1) == 0


def test_deterministic_serialization_and_truncation(fixture):
    q, tr, cfg = fixture
    a = make_evidence(q, tr, [0, 1, 2], [.9, .8, .7], cfg)
    b = make_evidence(q, tr, [0, 1, 2], [.9, .8, .7], cfg)
    assert a == b
    assert digest(canonical(a['packet'])) == a['evidence_hash']
    cfg['max_prompt_tokens'] = a['prompt_tokens_estimate'] - 1
    c = make_evidence(q, tr, [0, 1, 2], [.9, .8, .7], cfg)
    assert c['final_k'] < a['final_k']
    assert all(row['support'] == c['final_k'] for row in c['packet']['lookup_aggregates'])
    assert c['packet']['query'] == q.query
    assert digest(canonical(c['packet'])) == c['evidence_hash']


def test_aggregates_use_exact_visible_donors(fixture):
    q, tr, cfg = fixture
    e = make_evidence(q, tr, [0, 2], [.8, .8], cfg)
    a = e['packet']['lookup_aggregates']
    assert a[0]['weighted_mean_performance'] == .5
    assert a[1]['unweighted_mean_performance'] == .5
    assert a[0]['performance_variance'] == .25
    assert a[0]['local_win_rate'] == .5
