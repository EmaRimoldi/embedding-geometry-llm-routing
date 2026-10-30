import importlib.util
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_release_exports_root_license_metadata_and_entrypoints(tmp_path):
    spec = importlib.util.spec_from_file_location('release_root_checks', ROOT / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {'LICENSE', 'CITATION.cff', '.zenodo.json', 'Makefile', 'docs/CONTRIBUTING.md', 'docs/THIRD_PARTY.md', 'experiments/README.md'}
    for name in required:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text('required release material\n')
    (tmp_path / 'private-local-file.txt').write_text('must stay local\n')
    exported = {str(p.relative_to(tmp_path)) for p in module.public_files(tmp_path)}
    assert exported == required, 'The built ZIP must contain the license and release entrypoints'


def test_finder_metadata_is_not_exported(tmp_path):
    spec = importlib.util.spec_from_file_location('release_checks', ROOT / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'experiments/src').mkdir(parents=True)
    (tmp_path / 'experiments/src/.DS_Store').write_bytes(b'irrelevant machine metadata')
    (tmp_path / 'experiments/src/actual.py').write_text('value = 1\n')
    files = {str(p.relative_to(tmp_path)) for p in module.public_files(tmp_path)}
    assert files == {'experiments/src/actual.py'}


def test_nested_numeric_evidence_is_not_ignored_by_git(tmp_path):
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    shutil.copyfile(ROOT.parent / '.gitignore', tmp_path / '.gitignore')
    relative = 'experiments/results/evidence/diagnostics/knn_cost_reward_format/paper/aistats_cost_reward.tex'
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text('numerical evidence\n')
    result = subprocess.run(['git', 'check-ignore', '--quiet', relative], cwd=tmp_path)
    assert result.returncode == 1, 'The clone would silently omit a required frozen evidence file'


def test_generated_workspace_data_is_excluded_but_root_is_compact(tmp_path):
    spec = importlib.util.spec_from_file_location('workspace_export_checks', ROOT / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ['experiments/src/actual.py', 'experiments/results/evidence/score.json',
                 'experiments/datasets/proof_twins_v1/prepared/queries.jsonl',
                 'experiments/results/new-run/checkpoint.pt', 'experiments/embedders/weights.bin',
                 'experiments/datasets/routerbench/raw/private.bin']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    exported = {str(p.relative_to(tmp_path)) for p in module.public_files(tmp_path)}
    assert exported == {'experiments/src/actual.py', 'experiments/results/evidence/score.json',
                        'experiments/datasets/proof_twins_v1/prepared/queries.jsonl'}
    actual = {str(p.relative_to(ROOT.parent)) for p in module.public_files(ROOT.parent)}
    assert {p.split('/')[0] for p in actual if '/' in p and not p.startswith('.')} == {'docs', 'experiments'}
    assert {p for p in actual if '/' not in p and not p.startswith('.')} == {'README.md', 'Makefile', 'LICENSE', 'CITATION.cff'}


@pytest.mark.parametrize('generated,expected_status', [('correct', 'passed'), ('changed', 'failed')])
def test_saved_table_gate_checks_values_not_only_nonempty_output(tmp_path, monkeypatch, generated, expected_status):
    spec = importlib.util.spec_from_file_location('saved_table_checks', ROOT / 'scripts/render_saved_tables.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    evidence = project / 'results/evidence'
    evidence.mkdir(parents=True)
    (evidence / 'input.json').write_text('{}')
    (evidence / 'table.tex').write_text('correct')
    (project / 'src/render.py').write_text(
        "from pathlib import Path\np=Path('results/table.tex')\n"
        "p.parent.mkdir(parents=True,exist_ok=True)\np.write_text(" + repr(generated) + ")\n")
    monkeypatch.setattr(renderer, 'ROOT', project)
    monkeypatch.setattr(renderer, 'EVIDENCE', evidence)
    run = project / 'scratch'
    run.mkdir()
    report = renderer.render(dict(name='fixture', script='src/render.py', sources=[], configs=[],
                                  evidence=['input.json'], outputs=['table.tex']), run)
    assert report['status'] == expected_status
