#!/usr/bin/env python3
"""Verify model weight bytes against the pinned public LFS SHA-256 values."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'embedders')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/model-weights.json')
    args = parser.parse_args()
    records = json.loads((ROOT / 'manifests/model_weight_checksums.json').read_text())['files']
    verified = []
    for record in records:
        path = args.model_root / record['asset'] / record['file']
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        if path.stat().st_size != record['bytes'] or h.hexdigest() != record['sha256']:
            raise ValueError('Model weight mismatch: ' + record['asset'] + '/' + record['file'])
        verified.append(record)
    result = {'passed': True, 'scope': 'Full model-weight bytes match pinned public LFS checksums', 'files': verified}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'files': len(verified), 'bytes': sum(r['bytes'] for r in verified)}))


if __name__ == '__main__':
    main()
