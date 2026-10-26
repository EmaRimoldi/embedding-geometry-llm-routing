import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def baseline_common():
    path = Path(__file__).resolve().parents[1] / 'src/baselines/common.py'
    spec = importlib.util.spec_from_file_location('integrity_baseline_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_embedding_bytes_are_rejected(tmp_path, monkeypatch):
    common = baseline_common()
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    data = tmp_path / 'datasets/routerbench/prepared'
    data.mkdir(parents=True)
    np.savez(data / 'panel.npz', reward=np.ones((4, 2)), cost=np.zeros((4, 2)),
             family=np.array(['a', 'a', 'b', 'b']), topic=np.repeat('', 4),
             subgroup=np.repeat('group', 4), keys=np.array(['a0', 'a1', 'b0', 'b1']))
    (data / 'queries.jsonl').write_text(''.join(json.dumps({'key': i}) + '\n' for i in range(4)))
    (data / 'manifest.json').write_text('{}')
    vectors = tmp_path / 'artifacts/embeddings/gte-qwen2-7b/routerbench'
    vectors.mkdir(parents=True)
    np.save(vectors / 'vectors.npy', np.ones((4, 3), dtype=np.float32))
    metadata = {'rows': 4, 'query_sha256': common.sha(data / 'queries.jsonl'),
                'vector_sha256': common.sha(vectors / 'vectors.npy')}
    (vectors / 'manifest.json').write_text(json.dumps(metadata))
    assert common.load('routerbench-0shot', 'gte')['x'].shape == (4, 3)
    # Preserve dimensions and finite values, so shape/norm checks alone cannot
    # detect a changed feature cache that would silently alter routing results.
    x = np.ones((4, 3), dtype=np.float32)
    x[0, 0] = 2
    np.save(vectors / 'vectors.npy', x)
    with pytest.raises(ValueError, match='Embedding checksum'):
        common.load('routerbench-0shot', 'gte')
