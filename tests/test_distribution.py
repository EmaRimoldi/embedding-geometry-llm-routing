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
    required = {'LICENSE', 'Makefile', 'requirements.lock', 'docs/THIRD_PARTY.md', 'README.md'}
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
    (tmp_path / 'src').mkdir(parents=True)
    (tmp_path / 'src/.DS_Store').write_bytes(b'irrelevant machine metadata')
    (tmp_path / 'src/actual.py').write_text('value = 1\n')
    files = {str(p.relative_to(tmp_path)) for p in module.public_files(tmp_path)}
    assert files == {'src/actual.py'}


@pytest.mark.parametrize('relative', [
    'reproducibility/evidence/diagnostics/knn_cost_reward_format/paper/aistats_cost_reward.tex',
    'reproducibility/evidence/nvidia/results/routerbench/iid_0/selection.json',
])
def test_nested_numeric_evidence_is_not_ignored_by_git(tmp_path, relative):
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    shutil.copyfile(ROOT / '.gitignore', tmp_path / '.gitignore')
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text('numerical evidence\n')
    result = subprocess.run(['git', 'check-ignore', '--quiet', relative], cwd=tmp_path)
    assert result.returncode == 1, 'The clone would silently omit a required frozen evidence file'


def test_generated_workspace_data_is_excluded_but_root_is_compact(tmp_path):
    spec = importlib.util.spec_from_file_location('workspace_export_checks', ROOT / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ['src/actual.py', 'reproducibility/evidence/score.json',
                 'reproducibility/examples/proof_twins.jsonl',
                 'results/new-run/checkpoint.pt', 'embedders/weights.bin',
                 'datasets/routerbench/raw/private.bin']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    exported = {str(p.relative_to(tmp_path)) for p in module.public_files(tmp_path)}
    assert exported == {'src/actual.py', 'reproducibility/evidence/score.json',
                        'reproducibility/examples/proof_twins.jsonl'}
    actual = {str(p.relative_to(ROOT)) for p in module.public_files(ROOT)}
    assert {p.split('/')[0] for p in actual if '/' in p and not p.startswith('.')} == {'src', 'configs', 'tests', 'scripts', 'jobs', 'docs', 'reproducibility'}
    assert {p for p in actual if '/' not in p and not p.startswith('.')} == {'README.md', 'Makefile', 'LICENSE', 'requirements.lock'}


def test_layout_materializes_example_without_replacing_changed_runtime_data(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('layout_checks', ROOT / 'scripts/initialize_release.py')
    initializer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(initializer)
    monkeypatch.setattr(initializer, 'ROOT', tmp_path)
    shutil.copytree(ROOT / 'src/nvidia_utility/vendor', tmp_path / 'src/nvidia_utility/vendor')
    example = tmp_path / 'reproducibility/examples/proof_twins.jsonl'
    example.parent.mkdir(parents=True)
    example.write_bytes((ROOT / 'reproducibility/examples/proof_twins.jsonl').read_bytes())
    prompts = tmp_path / 'datasets/proof_twins_v1/prepared/queries.jsonl'
    initializer.layout()
    assert prompts.read_bytes() == example.read_bytes()
    initializer.layout()
    assert not (tmp_path / 'datasets/routerbench').exists()
    prompts.write_bytes(b'changed runtime data\n')
    with pytest.raises(RuntimeError, match='Illustrative prompt fixture differs'):
        initializer.layout()
    assert prompts.read_bytes() == b'changed runtime data\n'


@pytest.mark.parametrize('generated,expected_status', [('correct', 'passed'), ('changed', 'failed')])
def test_saved_table_gate_checks_values_not_only_nonempty_output(tmp_path, monkeypatch, generated, expected_status):
    spec = importlib.util.spec_from_file_location('saved_table_checks', ROOT / 'scripts/render_saved_tables.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    evidence = project / 'reproducibility/evidence'
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
