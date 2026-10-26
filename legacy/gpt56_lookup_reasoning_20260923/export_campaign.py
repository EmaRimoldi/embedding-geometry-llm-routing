"""Read canonical campaign arrays; export only explicitly allowed fields for this study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e4-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    e4, out = args.e4_root.resolve(), args.output.resolve()
    if not out.is_relative_to(e4):
        raise ValueError('Remote exports must remain inside experiments4')
    out.mkdir(parents=True, exist_ok=False)
    campaign = e4 / 'astra_definitive_paper_20260923_v1'
    sys.path.insert(0, str(e4 / 'src'))
    os.environ['E4_ROOT'] = str(e4)
    os.environ['RB_RUNTIME_ROOT'] = str(e4.parent)
    from rb.data import load_panel
    raw = load_panel('routerbench-0shot')
    frozen = campaign / 'inputs/routerbench-0shot'
    z = np.load(frozen / 'panel.npz', allow_pickle=False)
    meta = json.loads((frozen / 'source_manifest.json').read_text())
    assert np.array_equal(raw.query_ids, z['query_ids'])
    assert np.array_equal(raw.models, meta['models'])
    assert np.array_equal(raw.rewards, z['r']) and np.array_equal(raw.costs, z['c'])
    assert raw.available.all()
    np.savez_compressed(out / 'panel.npz', **{k: z[k] for k in ['r', 'c', 'fam', 'query_ids', 'topic']},
                        models=raw.models, families=np.asarray(meta['families']))
    topics = z['topic']
    with open(out / 'queries.jsonl', 'w') as f:
        for i, qid in enumerate(raw.query_ids):
            f.write(json.dumps(dict(sample_id=str(qid), query=str(raw.prompts[i]),
                                   dataset=str(raw.dataset_ids[i]), task=str(topics[i])), ensure_ascii=False) + '\n')
    src = e4 / 'astra_reward_transfer_20260922_v1/inputs/routerbench-0shot/gte.npy'
    assert sha(src) == meta['snapshot_sha256']['gte.npy']
    shutil.copy2(src, out / 'gte.npy')
    shutil.copy2(frozen / 'splits.json', out / 'splits.json')
    shutil.copy2(frozen / 'source_manifest.json', out / 'source_manifest.json')
    # Reuse the exact campaign scoring, normalization and retrieval implementation.
    vendor = out / 'campaign_code'
    vendor.mkdir()
    for name in ['common.py', 'run_fixed.py', 'freeze.py', 'run_mlp_gpu.py']:
        shutil.copy2(campaign / 'code' / name, vendor / name)
    shutil.copy2(e4 / 'src/rb/data.py', vendor / 'rb_data.py')
    metric = Path(__file__).resolve().parent / 'provenance/frontier_metrics.py'
    if not metric.is_file():
        raise FileNotFoundError(metric)
    shutil.copy2(metric, vendor / 'frontier_metrics.py')
    for kind in ['knn', 'prior', 'mlp_gpu']:
        source = campaign / 'results' / kind / 'routerbench-0shot/iid_0/gte'
        if source.is_dir():
            shutil.copytree(source, out / 'baselines' / kind)
    manifest = {'source_root': str(e4), 'campaign': str(campaign), 'rows': len(raw.query_ids),
                'models': raw.models.tolist(), 'source_panel_sha256': sha(frozen / 'panel.npz'),
                'files': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    (out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'rows': len(raw.query_ids), 'files': len(manifest['files']), 'output': str(out)}))


if __name__ == '__main__':
    main()
