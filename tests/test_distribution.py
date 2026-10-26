import importlib.util
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_finder_metadata_is_not_exported(tmp_path):
    spec = importlib.util.spec_from_file_location('release_checks', ROOT / 'scripts/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/.DS_Store').write_bytes(b'irrelevant machine metadata')
    (tmp_path / 'src/actual.py').write_text('value = 1\n')
    files = {str(p.relative_to(tmp_path)) for p in module.public_files(tmp_path)}
    assert files == {'src/actual.py'}


def test_nested_numeric_evidence_is_not_ignored_by_git(tmp_path):
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    shutil.copyfile(ROOT / '.gitignore', tmp_path / '.gitignore')
    relative = 'results/evidence/diagnostics/knn_cost_reward_format/paper/aistats_cost_reward.tex'
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text('numerical evidence\n')
    result = subprocess.run(['git', 'check-ignore', '--quiet', relative], cwd=tmp_path)
    assert result.returncode == 1, 'The clone would silently omit a required frozen evidence file'
