"""Exercise the real audit launcher with an inert scheduler and no model fits."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import importlib.util
from concurrent.futures import ThreadPoolExecutor

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_audit_launcher_defers_missing_baselines(tmp_path):
    for name in ['jobs', 'scripts', 'configs']:
        (tmp_path / name).mkdir()
    for name in ['env.sh', 'nvidia_utility_audit.sbatch']:
        shutil.copyfile(ROOT / 'jobs' / name, tmp_path / 'jobs' / name)
    helper = ROOT / 'scripts/submit_utility_report.py'
    shutil.copyfile(helper, tmp_path / 'scripts' / helper.name)
    (tmp_path / 'configs/geometry.json').write_text('{}')
    finished = tmp_path / 'results/nvidia_layer_validation/utility_v1/finished'
    finished.mkdir(parents=True)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    scheduler = bindir / 'sbatch'
    scheduler.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\nprintf "12345;fixture\\n"\n')
    scheduler.chmod(0o755)
    log = tmp_path / 'calls.log'
    env = dict(os.environ, PAPER_PYTHON=sys.executable, CALL_LOG=str(log),
               PATH=str(bindir) + os.pathsep + os.environ['PATH'])
    subprocess.run(['bash', 'jobs/nvidia_utility_audit.sbatch'], cwd=tmp_path,
                   env=env, check=True, capture_output=True, text=True)
    calls = log.read_text().splitlines()
    assert len(calls) == 1, 'Report was submitted before its baseline prerequisite'
    assert '--array=0-22%4' in calls[0]
    record = json.loads((finished / 'audit_submission.json').read_text())
    assert record['audit_array'] == '12345'
    assert record['report'] is None
    assert record['status'] == 'waiting_for_baselines'


def coordinator():
    spec = importlib.util.spec_from_file_location('utility_report_submission',
                                                ROOT / 'scripts/submit_utility_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.audit_accounting = lambda array: ''  # Pending array; never query real Slurm.
    return module


def baselines(root, complete=True):
    directory = root / 'results/baselines/_summary_reduced'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'scores.csv').write_text('complete\nTrue\n')
    (directory / 'status.json').write_text(json.dumps(
        {'all_complete': complete, 'missing': [] if complete else [{'split': 'ood_0'}]}))


def test_join_both_completion_orders_and_partial_baselines(tmp_path):
    helper = coordinator()
    calls = []

    def scheduler(command):
        calls.append(command)
        return '45678;fixture\n'

    # Utility audits finish first. A present but partial CSV must also defer.
    assert helper.submit_report(tmp_path, '12345', scheduler)['report'] is None
    baselines(tmp_path, complete=False)
    assert helper.submit_report(tmp_path, scheduler=scheduler)['report'] is None
    assert not calls
    baselines(tmp_path)
    record = helper.submit_report(tmp_path, scheduler=scheduler)
    assert record['report'] == '45678' and record['status'] == 'submitted'
    assert calls[0][2] == '--dependency=afterok:12345'
    assert calls[0][-1] == str(tmp_path / 'jobs/nvidia_utility_report.sbatch')
    assert len(record['baseline_scores_sha256']) == 64
    helper.submit_report(tmp_path, scheduler=scheduler)
    assert len(calls) == 1

    # Baselines finish first. No audit dependency means no report submission.
    other = tmp_path / 'other-order'
    baselines(other)
    assert helper.submit_report(other, scheduler=scheduler)['status'] == 'waiting_for_audit'
    assert len(calls) == 1
    assert helper.submit_report(other, '23456', scheduler)['report'] == '45678'
    assert calls[1][2] == '--dependency=afterok:23456'


def test_concurrent_completion_submits_once(tmp_path):
    helper = coordinator()
    helper.submit_report(tmp_path, '12345')
    baselines(tmp_path)
    calls = []

    def scheduler(command):
        calls.append(command)
        return '45678'

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda _: helper.submit_report(tmp_path, scheduler=scheduler), range(4)))
    assert len(calls) == 1
    assert all(record['report'] == '45678' for record in records)


def test_scheduler_failure_retains_dependency_without_success(tmp_path):
    helper = coordinator()
    baselines(tmp_path)

    def failed(command):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        helper.submit_report(tmp_path, '12345', failed)
    path = tmp_path / 'results/nvidia_layer_validation/utility_v1/finished/audit_submission.json'
    record = json.loads(path.read_text())
    assert record['audit_array'] == '12345' and record['report'] is None
    assert record['status'] == 'ready_for_submission'


def test_completed_audit_array_needs_no_expired_dependency(tmp_path):
    helper = coordinator()
    baselines(tmp_path)
    helper.audit_accounting = lambda array: '\n'.join(
        f'{array}_{i}|COMPLETED|0:0' for i in range(23))
    calls = []
    record = helper.submit_report(tmp_path, '12345', lambda command: calls.append(command) or '45678')
    assert record['audits_verified_completed'] is True
    assert record['audit_dependency'] is None
    assert not any(arg.startswith('--dependency') for arg in calls[0])


def test_incomplete_accounting_retains_afterok_dependency():
    helper = coordinator()
    helper.audit_accounting = lambda array: '\n'.join(
        f'{array}_{i}|COMPLETED|0:0' for i in range(22))
    assert helper.audit_dependency('12345') == 'afterok:12345'


@pytest.mark.parametrize('last_state', ['FAILED|1:0', 'CANCELLED by operator|0:15'])
def test_failed_audit_array_cannot_submit_report(tmp_path, last_state):
    helper = coordinator()
    baselines(tmp_path)
    helper.audit_accounting = lambda array: '\n'.join(
        [f'{array}_{i}|COMPLETED|0:0' for i in range(22)] + [f'{array}_22|{last_state}'])
    calls = []
    with pytest.raises(RuntimeError, match='audit tasks failed'):
        helper.submit_report(tmp_path, '12345', lambda command: calls.append(command) or '45678')
    assert not calls


@pytest.mark.parametrize('value', ['', '123_bad', '../report', '１２３'])
def test_invalid_scheduler_id_is_rejected(value):
    with pytest.raises(ValueError):
        coordinator().job_id(value)
