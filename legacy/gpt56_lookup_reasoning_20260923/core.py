"""Leakage-separated evidence construction and shared routing contracts."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_config(path=None):
    return json.loads(Path(path or ROOT / 'config.json').read_text())


def normalize_text(text):
    return ' '.join(text.split())


def validate_split(ids, split):
    ids = np.asarray(ids)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate unique query IDs in panel; query-model rows must be grouped first')
    train, test = np.asarray(split['train']), np.asarray(split['test'])
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Duplicate split indices')
    if set(train) & set(test) or set(train) | set(test) != set(range(len(ids))):
        raise ValueError('Train/test overlap or incomplete coverage')
    for inner in split['inner']:
        a, b = set(inner['fit']), set(inner['val'])
        if a & b or a | b != set(train):
            raise ValueError('Inner split leakage')


def choose(performance, costs, lam):
    """Campaign convention and exact np.argmax first-column tie rule."""
    return np.argmax(np.asarray(performance) - lam * np.asarray(costs), axis=-1)


@dataclass(frozen=True)
class InferenceQuery:
    sample_id: str
    query: str
    dataset: str
    task: str


@dataclass
class TrainingLookup:
    """Contains training outcomes only. InferenceQuery has no outcome fields."""
    queries: list[InferenceQuery]
    rewards: np.ndarray
    costs: np.ndarray
    models: list[str]


def output_schema(models):
    return {'type': 'object', 'additionalProperties': False,
            'required': ['sample_id', 'model_estimates'],
            'properties': {'sample_id': {'type': 'string'},
                           'model_estimates': {'type': 'array', 'minItems': len(models), 'maxItems': len(models),
                            'items': {'type': 'object', 'additionalProperties': False,
                             'required': ['model_name', 'expected_performance'], 'properties': {
                              'model_name': {'type': 'string', 'enum': models},
                              'expected_performance': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}}}


def parse_prediction(raw, sample_id, models):
    def unique_pairs(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('Duplicate JSON key')
            out[k] = v
        return out
    d = json.loads(raw, object_pairs_hook=unique_pairs)
    if set(d) != {'sample_id', 'model_estimates'} or d['sample_id'] != sample_id:
        raise ValueError('Wrong sample_id or top-level fields')
    rows = d['model_estimates']
    if not isinstance(rows, list) or len(rows) != len(models):
        raise ValueError('Wrong number of candidate predictions')
    if [r.get('model_name') for r in rows] != models:
        raise ValueError('Missing, duplicate, unknown or reordered candidate')
    for row in rows:
        p = row.get('expected_performance')
        if set(row) != {'model_name', 'expected_performance'} or type(p) not in (int, float) or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid expected performance')
    return d


def make_evidence(query, lookup, neighbors, similarities, cfg, template=None):
    """Outcome access is structurally restricted to the supplied training lookup."""
    enc = tiktoken.get_encoding(cfg['tokenizer'])
    count = lambda s: len(enc.encode(s, disallowed_special=()))
    template = template or (ROOT / 'prompts/router_prompt.txt').read_text()
    nb = list(map(int, neighbors))
    sim = np.asarray(similarities, float)
    if not nb or len(nb) != len(sim) or len(set(nb)) != len(nb) or not np.isfinite(sim).all():
        raise ValueError('Invalid neighbors')
    for j in nb:
        donor = lookup.queries[j]
        if donor.sample_id == query.sample_id or normalize_text(donor.query) == normalize_text(query.query):
            raise ValueError('Self or duplicate-query evidence leakage')
    clipped = []
    for j in nb:
        text = lookup.queries[j].query
        tokens = enc.encode(text, disallowed_special=())
        cap = cfg['neighbor_text_tokens']
        clipped.append((enc.decode(tokens[:cap]), len(tokens), len(tokens) > cap))
    requested_k = len(nb)
    while nb:
        rr, cc = lookup.rewards[nb], lookup.costs[nb]
        w = np.exp((sim[:len(nb)] - np.max(sim[:len(nb)])) / cfg['weight_temperature'])
        w /= w.sum()
        means = w @ rr
        costs = w @ cc
        ties = rr == rr.max(axis=1, keepdims=True)
        wins = (ties / ties.sum(axis=1, keepdims=True)).mean(axis=0)
        packet = {'sample_id': query.sample_id, 'query': query.query,
                  'metadata': {'dataset': query.dataset, 'task': query.task or None},
                  'candidate_models': [{'model_name': m, 'training_cost_estimate': float(costs[i])} for i, m in enumerate(lookup.models)],
                  'retrieved_training_examples': [
                      {'sample_id': lookup.queries[j].sample_id, 'similarity': float(sim[n]),
                       'query': clipped[n][0], 'query_truncated': clipped[n][2],
                       'performance_by_model': {m: float(rr[n, i]) for i, m in enumerate(lookup.models)}}
                      for n, j in enumerate(nb)],
                  'lookup_aggregates': [
                      {'model_name': m, 'weighted_mean_performance': float(means[i]),
                       'unweighted_mean_performance': float(rr[:, i].mean()),
                       'performance_variance': float(rr[:, i].var()),
                       'local_win_rate': float(wins[i]), 'support': len(nb),
                       'training_cost_estimate': float(costs[i])} for i, m in enumerate(lookup.models)]}
        body = canonical(packet)
        prompt = template.replace('{{EVIDENCE_PACKET}}', body)
        nt = count(prompt)
        if nt <= cfg['max_prompt_tokens']:
            return {'packet': packet, 'prompt': prompt, 'evidence_hash': digest(body), 'prompt_hash': digest(prompt),
                    'prompt_tokens_estimate': nt, 'tokenizer': cfg['tokenizer'], 'requested_k': requested_k,
                    'final_k': len(nb), 'neighbor_texts_truncated': sum(x[2] for x in clipped[:len(nb)]),
                    'mean_similarity': float(sim[:len(nb)].mean()),
                    'lookup_uncertainty': float(rr.var(axis=0).mean()),
                    'neighbor_disagreement': float(1 - np.max(np.bincount(rr.argmax(axis=1), minlength=rr.shape[1])) / len(nb))}
        nb.pop()
    raise ValueError('Query plus minimum evidence exceeds prompt budget; do not truncate test query')
