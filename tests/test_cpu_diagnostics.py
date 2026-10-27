import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'fresh_cpu_checks', Path(__file__).resolve().parents[1] / 'scripts/reproduce_cpu_diagnostics.py')
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


@pytest.mark.parametrize('actual,expected', [
    ({'score': 0.2}, {'score': 0.1}),
    ({'score': float('nan')}, {'score': 0.1}),
    ({'count': True}, {'count': 1}),
    ({'values': [1, 2]}, {'values': [1]}),
    ({'definition': 'changed'}, {'definition': 'reference'}),
    ({}, {'score': 0.1}),
])
def test_fresh_diagnostic_gate_rejects_changed_or_invalid_report(actual, expected):
    with pytest.raises(ValueError):
        CHECKS.compare(actual, expected, 'fixture', {'numeric_values': 0, 'maximum_absolute_error': 0.0})


def test_fresh_diagnostic_gate_accepts_only_documented_float_tolerance():
    counts = {'numeric_values': 0, 'maximum_absolute_error': 0.0}
    CHECKS.compare({'scores': [0.1 + 5e-13, 12], 'definition': 'same'},
                   {'scores': [0.1, 12], 'definition': 'same'}, 'fixture', counts)
    assert counts['numeric_values'] == 2
    assert 0 < counts['maximum_absolute_error'] < 1e-12
