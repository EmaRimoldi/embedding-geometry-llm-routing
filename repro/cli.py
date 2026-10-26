"""Run recovered experiments without changing the original source or inputs."""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'legacy'
STUDY = 'astra_definitive_paper_20260923_v1'
PANELS = {'routerbench-0shot': (34661, 11), 'sprout-sixfam': (43887, 13)}
TARGETS = {'routerbench-0shot': (34771, 11), 'sprout-sixfam': (43887, 13)}
ARCHIVE_SHA256 = '24db2a3b666e75bbf75dd72af5ab5cdd150da19b72d4a463b4ad263dc43608ab'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment():
    packages = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def panel_directory(data_root, panel):
    return Path(data_root).resolve() / STUDY / 'inputs' / panel


def read_panel(data_root, panel, profile='legacy'):
    directory = panel_directory(data_root, panel)
    with np.load(directory / 'panel.npz', allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    require({'r', 'c', 'fam', 'query_ids'} <= set(data), 'Missing panel fields')
    r, c = data['r'], data['c']
    require(r.ndim == 2 and r.shape == c.shape and all(r.shape), 'Invalid reward/cost dimensions')
    require(np.isfinite(r).all() and np.isfinite(c).all(), 'Nonfinite outcomes')
    require((c >= 0).all() and ((r >= 0) & (r <= 1)).all(), 'Invalid reward/cost range')
    n = len(r)
    require(data['fam'].shape == (n,) and data['query_ids'].shape == (n,), 'Invalid label/ID dimensions')
    require(len(np.unique(data['query_ids'])) == n, 'Duplicate query IDs')
    if profile != 'any':
        expected = (PANELS if profile == 'legacy' else TARGETS)[panel]
        require(r.shape == expected, f'{profile} cohort requires {expected}, received {r.shape}')
    splits = json.loads((directory / 'splits.json').read_text())
    require(isinstance(splits, dict) and bool(splits), 'Empty split registry')

    def indices(value, label):
        a = np.asarray(value)
        require(a.ndim == 1 and np.issubdtype(a.dtype, np.integer) and len(a) > 0, f'Invalid {label} indices')
        require(a.min() >= 0 and a.max() < n and len(np.unique(a)) == len(a), f'Invalid/duplicate {label} rows')
        return a

    ood_coverage = np.zeros(n, dtype=np.int16)
    ood_count = 0
    for name, split in splits.items():
        train, test = indices(split['train'], name + ' train'), indices(split['test'], name + ' test')
        require(not np.intersect1d(train, test).size, f'{name}: train/test overlap')
        require(len(train) + len(test) == n, f'{name}: incomplete outer partition')
        for inner in split.get('inner', []):
            fit, val = indices(inner['fit'], name + ' fit'), indices(inner['val'], name + ' val')
            require(not np.intersect1d(fit, val).size, f'{name}: inner overlap')
            require(np.isin(fit, train).all() and np.isin(val, train).all(), f'{name}: inner rows outside training')
            require(len(fit) + len(val) == len(train), f'{name}: incomplete inner partition')
        if name.startswith('ood_'):
            require(len(np.unique(data['fam'][test])) == 1, f'{name}: multiple target families')
            require(not np.intersect1d(data['fam'][train], data['fam'][test]).size, f'{name}: held-out family leakage')
            ood_coverage[test] += 1
            ood_count += 1
    if profile != 'any':
        require(sum(name.startswith('iid_') for name in splits) == 5, 'Expected five ID splits')
        require(ood_count == len(np.unique(data['fam'])) and np.all(ood_coverage == 1), 'Incomplete OOD coverage')
    summary = {'panel': panel, 'profile': profile, 'shape': list(r.shape),
               'families': len(np.unique(data['fam'])), 'splits': list(splits),
               'panel_sha256': sha256(directory / 'panel.npz'),
               'splits_sha256': sha256(directory / 'splits.json')}
    return data, splits, summary


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def prepare_legacy(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    require(sha256(archive) == ARCHIVE_SHA256, 'Archive hash differs from the audited September release')
    require(not destination.exists(), 'Destination exists; select a new directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'data'
        with zipfile.ZipFile(archive) as z:
            for panel in PANELS:
                for name in ['panel.npz', 'splits.json', 'source_manifest.json']:
                    relative = f'{STUDY}/inputs/{panel}/{name}'
                    matches = [n for n in z.namelist() if n == relative or n.endswith('/' + relative)]
                    require(len(matches) == 1, f'Expected one archive member for {relative}')
                    member = z.getinfo(matches[0])
                    require(member.file_size < 512 * 1024 * 1024, 'Unexpectedly large input member')
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
        summaries = [read_panel(staging, panel)[2] for panel in PANELS]
        staging.rename(destination)
    return {'data_root': str(destination), 'archive_sha256': ARCHIVE_SHA256, 'panels': summaries}


def common_module():
    path = LEGACY / STUDY / 'code/common.py'
    spec = importlib.util.spec_from_file_location('recovered_legacy_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_baselines(data_root, panel, split_name, output, profile='legacy'):
    data, splits, summary = read_panel(data_root, panel, profile)
    selected = list(splits) if split_name == 'all' else [split_name]
    require(all(s in splits for s in selected), 'Unknown split')
    cfg = json.loads((LEGACY / STUDY / 'config/protocol.json').read_text())
    common = common_module()
    results = {}
    for name in selected:
        train, test = np.array(splits[name]['train']), np.array(splits[name]['test'])
        r, c, fam = data['r'], data['c'], data['fam']
        curves = {}
        for lam in cfg['lambdas'][panel]:
            best = np.argmax((r[train] - lam * c[train]).mean(axis=0))
            choices = {'best_single_train': np.full(len(test), best),
                       'realized_oracle': np.argmax(r[test] - lam * c[test], axis=1)}
            curves[str(lam)] = {method: common.score(ch, r[test], c[test], fam[test], lam)
                                for method, ch in choices.items()}
        results[name] = curves
    value = {'experiment': 'training-selected single-model baseline and realized oracle upper bound',
             'source_status': 'historical September cohort; not final October reproduction',
             'input': summary, 'environment': environment(), 'results': results,
             'score_code_sha256': sha256(LEGACY / STUDY / 'code/common.py')}
    write_json(output, value)
    return {'output': str(output), 'panel': panel, 'splits_completed': len(results), 'penalties': len(cfg['lambdas'][panel])}


def legacy_run(data_root, panel, split_name, rep, kind, workdir, profile='legacy'):
    _, splits, summary = read_panel(data_root, panel, profile)
    require(split_name in splits, 'Unknown split')
    data_root, workdir = Path(data_root).resolve(), Path(workdir).resolve()
    feature = data_root / 'embeddings' / panel / f'{rep}.npy'
    x = np.load(feature, mmap_mode='r', allow_pickle=False)
    require(x.ndim == 2 and len(x) == summary['shape'][0], 'Embedding/panel dimensions differ')
    require(not workdir.exists(), 'Work directory exists; select a fresh directory')
    tasks = []
    for task_panel, nf in [('routerbench-0shot', 7), ('sprout-sixfam', 6)]:
        for split in [f'iid_{i}' for i in range(5)] + [f'ood_{i}' for i in range(nf)]:
            for representation in ['gte', 'reader']:
                tasks.append((task_panel, split, representation))
    task = tasks.index((panel, split_name, rep))
    workspace = workdir / 'legacy'
    shutil.copytree(LEGACY, workspace, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    inputs = workspace / STUDY / 'inputs'
    require(not inputs.exists(), 'Source export unexpectedly contains input data')
    inputs.symlink_to(data_root / STUDY / 'inputs', target_is_directory=True)
    # E4_ROOT selects external embeddings. ROOT in the unchanged common.py selects
    # this private workspace, keeping source files and published results untouched.
    env = dict(os.environ, E4_ROOT=str(data_root), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    command = [sys.executable, str(workspace / STUDY / 'code/run_fixed.py'),
               '--task', str(task), '--kind', kind]
    write_json(workdir / 'launch.json', {'input': summary, 'kind': kind, 'representation': rep,
                                       'split': split_name, 'environment': environment(),
                                       'feature_sha256': sha256(feature),
                                       'code_sha256': sha256(LEGACY / STUDY / 'code/run_fixed.py')})
    completed = subprocess.run(command, cwd=workspace, env=env, check=False)
    require(completed.returncode == 0, f'Original runner failed with exit code {completed.returncode}; inspect {workdir}')
    result = workspace / STUDY / 'results' / kind / panel / split_name / rep / 'result.json'
    require(result.is_file(), 'Runner exited without result.json')
    return {'result': str(result), 'task': task, 'historical_protocol': True}


def inspect_submission(archive):
    """Inspect metadata only. Never extract or execute a supplied package."""
    archive = Path(archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            entries = [(i.filename, i.file_size) for i in z.infolist() if not i.is_dir()]
    else:
        with tarfile.open(archive, 'r:*') as t:
            entries = [(i.name, i.size) for i in t.getmembers() if i.isfile() or i.issym() or i.islnk()]
    unsafe = [name for name, _ in entries if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name]
    return {'sha256': sha256(archive), 'files': len(entries), 'uncompressed_bytes': sum(size for _, size in entries),
            'unsafe_member_paths': unsafe, 'python_sources': sum(name.endswith('.py') for name, _ in entries),
            'entrypoint_candidates': [name for name, _ in entries if name.endswith(('.sbatch', '.sh'))],
            'dependency_candidates': [name for name, _ in entries if any(s in name.rsplit('/', 1)[-1].lower() for s in ['requirements', 'pyproject', 'lock', 'environment'])],
            'status': 'Metadata inspected only; no source imported or executed'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('catalog', help='List recovered experiment entry points')
    prep = sub.add_parser('prepare-legacy-data', help='Stage audited historical inputs locally; no data is published')
    prep.add_argument('--archive', type=Path, required=True)
    prep.add_argument('--destination', type=Path, default=Path('data/legacy'))
    inspect = sub.add_parser('inspect-submission', help='Inspect a recovered conference archive without executing it')
    inspect.add_argument('--archive', type=Path, required=True)
    for name in ['validate', 'reference-baselines', 'run-legacy']:
        p = sub.add_parser(name)
        p.add_argument('--data-root', type=Path, required=True)
        p.add_argument('--panel', choices=list(PANELS), required=True)
        p.add_argument('--profile', choices=['legacy', 'target', 'any'], default='legacy',
                       help='Shape/split checks; target only checks structure, never establishes full scientific equivalence')
        if name != 'validate':
            p.add_argument('--split', default='all' if name == 'reference-baselines' else 'iid_0')
        if name == 'reference-baselines':
            p.add_argument('--output', type=Path, required=True)
        if name == 'run-legacy':
            p.add_argument('--representation', choices=['gte', 'reader'], required=True)
            p.add_argument('--kind', choices=['knn', 'prior'], required=True)
            p.add_argument('--workdir', type=Path, required=True)
    a = parser.parse_args(argv)
    try:
        if a.command == 'catalog':
            with (ROOT / 'provenance/experiment-code-inventory.csv').open() as f:
                result = list(csv.DictReader(f))
        elif a.command == 'prepare-legacy-data':
            result = prepare_legacy(a.archive, a.destination)
        elif a.command == 'inspect-submission':
            result = inspect_submission(a.archive)
        elif a.command == 'validate':
            result = read_panel(a.data_root, a.panel, a.profile)[2]
        elif a.command == 'reference-baselines':
            result = reference_baselines(a.data_root, a.panel, a.split, a.output, a.profile)
        else:
            require(a.profile != 'target', 'The historical runner cannot certify the final protocol; import and verify the conference implementation first')
            result = legacy_run(a.data_root, a.panel, a.split, a.representation, a.kind, a.workdir, a.profile)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, tarfile.TarError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
