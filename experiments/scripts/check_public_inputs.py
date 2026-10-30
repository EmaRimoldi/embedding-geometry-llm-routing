#!/usr/bin/env python3
"""Verify pinned public data/tokenizer files, optionally downloading and preparing them.

Model weights are deliberately excluded from this preparation check. Encoder
extraction still requires the complete assets downloaded by src/download_assets.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_asset(asset, download):
    info = HfApi().repo_info(asset['repo'], repo_type=asset['type'],
                             revision=asset['revision'], files_metadata=True)
    if info.sha != asset['revision']:
        raise ValueError('Resolved revision differs: ' + asset['name'])
    files = []
    for entry in info.siblings:
        if not any(fnmatch.fnmatch(entry.rfilename, pattern) for pattern in asset['include']):
            continue
        if entry.rfilename.endswith(('.safetensors', '.bin', '.pt')):
            continue
        path = ROOT / asset['destination'] / entry.rfilename
        if download:
            hf_hub_download(asset['repo'], entry.rfilename, repo_type=asset['type'],
                            revision=asset['revision'], local_dir=ROOT / asset['destination'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path.relative_to(ROOT)}; rerun with --download')
        expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
        actual = digest(path, git_blob=not entry.lfs)
        if actual != expected or path.stat().st_size != entry.size:
            raise ValueError('Remote checksum/size mismatch: ' + str(path.relative_to(ROOT)))
        files.append({'path': str(path.relative_to(ROOT)), 'bytes': entry.size,
                      'checksum_kind': 'sha256' if entry.lfs else 'git-blob-sha1', 'checksum': actual})
    if not files:
        raise ValueError('Empty public input selection: ' + asset['name'])
    return {'asset': asset['name'], 'revision': info.sha, 'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='Run both production preparation commands after verification')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/reproduced/public-inputs.json')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'manifests/assets.json').read_text())
    with ThreadPoolExecutor(max_workers=7) as pool:
        verified = list(pool.map(lambda asset: verify_asset(asset, args.download), assets))
    prepared = []
    if args.prepare:
        for dataset in ['routerbench', 'sprout']:
            manifest = ROOT / 'datasets' / dataset / 'prepared/manifest.json'
            if manifest.exists():
                raise FileExistsError(f'{dataset} already prepared; use a fresh reproduction copy')
            subprocess.run([sys.executable, '-B', str(ROOT / 'src/prepare_data.py'), dataset],
                           cwd=ROOT, check=True)
            prepared.append({'dataset': dataset, 'manifest': json.loads(manifest.read_text())})
    report = {'passed': True, 'scope': 'Remote-verified pinned public data/tokenizers; no model weights or encoder execution',
              'assets': verified, 'prepared': prepared}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'assets': len(verified),
                      'files': sum(len(a['files']) for a in verified),
                      'prepared': [v['dataset'] for v in prepared]}))


if __name__ == '__main__':
    main()
