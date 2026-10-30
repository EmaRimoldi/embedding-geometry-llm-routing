import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_checks', ROOT / 'scripts/check_paper_results.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_printed_precision_and_nonfinite_values():
    checks.check_rounded(.66154589119, .6615, 4)
    with pytest.raises(ValueError):
        checks.check_rounded(.662, .6615, 4)
    with pytest.raises(ValueError, match='Nonfinite'):
        checks.check_rounded(float('nan'), .6615, 4)


def test_missing_confidence_cell_is_rejected():
    with pytest.raises(ValueError, match='cell counts'):
        checks.check_cells([], [['0.1', '0.0', '0.2']])


def test_modified_aggregate_cannot_pass_paper_gate(tmp_path):
    subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce_results.py'),
                    '--output', str(tmp_path)], check=True, stdout=subprocess.DEVNULL)
    reference = json.loads((ROOT / 'configs/paper_reference.json').read_text())
    assert checks.verify(reference, tmp_path)['passed']
    target = tmp_path / 'main_comparison_mean_utility.csv'
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    changed = next(row for row in rows if row['dataset'] == 'routerbench-0shot'
                   and row['protocol'] == 'ood' and row['encoder'] == 'gte'
                   and row['method'] == 'kNN-100')
    changed['mean_utility'] = str(float(changed['mean_utility']) + .01)
    with target.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = checks.verify(reference, tmp_path)
    assert not result['passed']
    assert any(row.get('table') == 'tab:ood-encoder-means' for row in result['errors'])
