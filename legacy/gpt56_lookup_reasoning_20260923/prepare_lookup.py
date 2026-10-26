"""Freeze exact source split, build leakage-separated pilot/test packets, estimate usage."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from core import (ROOT, InferenceQuery, TrainingLookup, canonical, digest, load_config,
                  make_evidence, normalize_text, output_schema, sha_file, validate_split, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stage', choices=['pilot', 'test'], required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--exclude-test-text-duplicates', action='store_true',
                    help='User-approved audit mask; preserves saved training and excludes test aliases')
    args = ap.parse_args()
    cfg = load_config(args.config)
    data, out = args.data.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    # Hash-check every input, including the saved split and source-derived baselines.
    export = json.loads((data / 'export_manifest.json').read_text())
    for name, expected in export['files'].items():
        if sha_file(data / name) != expected:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    z = np.load(data / 'panel.npz', allow_pickle=False)
    ids, fam, models = z['query_ids'], z['fam'], z['models'].tolist()
    with open(data / 'queries.jsonl') as f:
        queries = [InferenceQuery(**q) for q in map(json.loads, f)]
    assert np.array_equal(ids, [q.sample_id for q in queries])
    split = json.loads((data / 'splits.json').read_text())[cfg['split']]
    validate_split(ids, split)
    if args.stage == 'pilot':
        source = np.asarray(split['inner'][0]['fit'])
        pool = np.asarray(split['inner'][0]['val'])
        selected = []
        for g in np.unique(fam):
            rows = pool[fam[pool] == g]
            # Stable hash sample depends only on query ID, seed and family, never outcomes.
            rows = sorted(rows, key=lambda i: digest(f"{cfg['pilot_seed']}|{ids[i]}"))
            selected.extend(rows[:cfg['pilot_per_family']])
        target = np.asarray(selected)
    else:
        source, target = np.asarray(split['train']), np.asarray(split['test'])
    normalized = [normalize_text(q.query) for q in queries]
    overlap = set(normalized[i] for i in source) & set(normalized[i] for i in target)
    excluded = []
    if overlap and args.stage == 'test' and args.exclude_test_text_duplicates:
        excluded = [int(i) for i in target if normalized[i] in overlap]
        target = np.asarray([i for i in target if normalized[i] not in overlap])
        overlap = set(normalized[i] for i in source) & set(normalized[i] for i in target)
    if overlap:
        raise ValueError(f'{len(overlap)} duplicate texts cross the frozen source/evaluation split; protocol review required')
    spec = importlib.util.spec_from_file_location('campaign_common', data / 'campaign_code/common.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    x = campaign.unit(np.load(data / 'gte.npy', mmap_mode='r'))
    if len(x) != len(ids):
        raise ValueError('Embedding alignment mismatch')
    lookup = TrainingLookup([queries[i] for i in source], z['r'][source], z['c'][source], models)
    out.mkdir(parents=True)
    write_json(out / 'config.json', cfg)
    write_json(out / 'output_schema.json', output_schema(models))
    shutil.copy2(ROOT / 'prompts/system.txt', out / 'system.txt')
    shutil.copy2(ROOT / 'prompts/router_prompt.txt', out / 'router_prompt.txt')
    manifest = {'stage': args.stage, 'split_name': cfg['split'],
                'source_rows': source.tolist(), 'evaluation_rows': target.tolist(),
                'source_ids': ids[source].tolist(), 'evaluation_ids': ids[target].tolist(),
                'outer_train_ids': ids[split['train']].tolist(), 'outer_test_ids': ids[split['test']].tolist(),
                'source_split_sha256': sha_file(data / 'splits.json'),
                'source_export_sha256': sha_file(data / 'export_manifest.json'),
                'config_hash': digest(canonical(cfg)), 'models': models, 'families': z['families'].tolist(),
                'data_root': str(data), 'duplicate_text_overlap': 0,
                'excluded_test_alias_rows': excluded, 'excluded_test_alias_ids': ids[excluded].tolist(),
                'test_alias_policy': 'User approved 2026-09-23: exclude test text aliases, preserve frozen training/split',
                'code_hashes': {p.name: sha_file(p) for p in sorted(ROOT.glob('*.py'))}}
    write_json(out / 'split_manifest.json', manifest)
    # Separate evaluator-only labels from the inference payload directory.
    np.savez_compressed(out / 'evaluation_only.npz', query_ids=ids[target], r=z['r'][target], c=z['c'][target],
                        fam=fam[target], topic=z['topic'][target],
                        source_mean_r=lookup.rewards.mean(axis=0), source_mean_c=lookup.costs.mean(axis=0))
    summaries = []
    with open(out / 'evidence.jsonl', 'w') as f:
        for start in range(0, len(target), 128):
            batch = target[start:start + 128]
            ix, dist = campaign.nearest(x[batch], x[source], cfg['k'])
            for i, nn, dd in zip(batch, ix, dist):
                evidence = make_evidence(queries[i], lookup, nn, 1 - dd, cfg)
                f.write(canonical(evidence) + '\n')
                summaries.append({k: v for k, v in evidence.items() if k not in ['packet', 'prompt']})
            f.flush()
            print(f'evidence {min(start + 128, len(target))}/{len(target)}', flush=True)
    counts = np.asarray([s['prompt_tokens_estimate'] for s in summaries])
    estimate = {'stage': args.stage, 'outer_train_queries': len(split['train']),
                'outer_test_queries': len(split['test']), 'retrieval_training_queries': len(source),
                'excluded_test_alias_queries': len(excluded),
                'evaluation_queries': len(target), 'candidates': len(models), 'k': cfg['k'],
                'actual_k_min': min(s['final_k'] for s in summaries), 'actual_k_max': max(s['final_k'] for s in summaries),
                'calls_before_retries': 2 * len(target), 'max_calls_with_retries': 2 * len(target) * cfg['max_attempts'],
                'prompt_tokens_mean_estimate': float(counts.mean()), 'prompt_tokens_max_estimate': int(counts.max()),
                'paired_prompt_tokens_total_estimate': int(2 * counts.sum()),
                'token_count_note': 'o200k_base text estimate; actual Codex context overhead and billing measured in pilot',
                'evidence_bytes': (out / 'evidence.jsonl').stat().st_size,
                'full_test_calls_before_retries': 2 * (len(target) if args.stage == 'test' else len(split['test'])),
                'full_test_evidence_bytes_extrapolated': int((out / 'evidence.jsonl').stat().st_size * len(split['test']) / len(target)),
                'api_cost_usd': None, 'billing_note': 'Codex subscription/quota; API tariff not assumed',
                'evidence_sha256': sha_file(out / 'evidence.jsonl'), 'manifest_sha256': sha_file(out / 'split_manifest.json')}
    write_json(out / 'dry_run_report.json', estimate)
    print(json.dumps(estimate, indent=2))


if __name__ == '__main__':
    main()
