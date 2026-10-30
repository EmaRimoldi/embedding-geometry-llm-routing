import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('documentation_checks', ROOT / 'scripts/check_docs.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def put(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def test_nested_links_duplicate_headings_and_code_fences(tmp_path):
    put(tmp_path, 'README.md', '[Guide](docs/guides/RUN.md#same-1)\n[Ref][ref]\n[ref]: docs/guides/RUN.md\n')
    put(tmp_path, 'docs/guides/RUN.md', '# Same\n## Same\n[Home](../../README.md)\n'
        '```markdown\n[Not a link](missing.md)\n```\nRun `python scripts/run.py`.\n')
    put(tmp_path, 'scripts/run.py', 'pass\n')
    report = checks.check(tmp_path)
    assert report['passed'], report['errors']
    assert report['local_links'] == 3


@pytest.mark.parametrize('target,reason', [('docs/missing.md', 'Missing exported'),
                                         ('docs/guide.md#absent', 'Missing heading'),
                                         ('../outside.md', 'escapes the repository')])
def test_broken_links_and_anchors_are_rejected(tmp_path, target, reason):
    put(tmp_path, 'README.md', f'[Guide]({target})\n')
    put(tmp_path, 'docs/guide.md', '# Guide\n')
    assert any(reason in item['problem'] for item in checks.check(tmp_path)['errors'])


def test_ignored_local_files_cannot_mask_a_broken_clone_link(tmp_path):
    put(tmp_path, 'README.md', '[Private cache](artifacts/local.json)\n')
    put(tmp_path, 'artifacts/local.json', '{}')
    assert not checks.check(tmp_path)['passed']


def test_missing_command_path_and_unreachable_manual(tmp_path):
    put(tmp_path, 'README.md', 'Run `python scripts/missing.py`.\n')
    put(tmp_path, 'docs/orphan.md', '# Not linked\n')
    report = checks.check(tmp_path)
    assert report['unreachable_documents'] == ['docs/orphan.md']
    assert any(item['target'] == 'scripts/missing.py' for item in report['errors'])


def test_html_assets_and_case_sensitive_links(tmp_path):
    put(tmp_path, 'README.md', '[Demo](docs/demo.html#panel)\n')
    put(tmp_path, 'docs/demo.html', '<div id="panel"><img src="plot.svg"></div>')
    put(tmp_path, 'docs/plot.svg', '<svg/>')
    assert checks.check(tmp_path)['passed']
    put(tmp_path, 'docs/demo.html', '<div id="panel"><img src="PLOT.svg"></div>')
    assert not checks.check(tmp_path)['passed']


def test_manifest_only_preserves_published_archive(tmp_path):
    for name in ['build_submission.py', 'verify_release.py']:
        target = tmp_path / 'scripts' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'scripts' / name, target)
    put(tmp_path, 'README.md', '# Initial\n')
    put(tmp_path, 'dist/routing_reproduction.zip', 'published immutable bytes')
    put(tmp_path, 'dist/routing_reproduction.zip.sha256', 'published checksum')
    archive = (tmp_path / 'dist/routing_reproduction.zip').read_bytes()
    sidecar = (tmp_path / 'dist/routing_reproduction.zip.sha256').read_bytes()
    for title in ['# Initial\n', '# Edited\n']:
        put(tmp_path, 'README.md', title)
        result = subprocess.run([sys.executable, '-B', 'scripts/build_submission.py', '--manifest-only'],
                                cwd=tmp_path, check=True, text=True, capture_output=True)
        assert json.loads(result.stdout)['archive_updated'] is False
        manifest = json.loads((tmp_path / 'reproducibility/provenance/release_manifest.json').read_text())
        assert manifest['files']['README.md']['bytes'] == len(title)
        assert (tmp_path / 'dist/routing_reproduction.zip').read_bytes() == archive
        assert (tmp_path / 'dist/routing_reproduction.zip.sha256').read_bytes() == sidecar
