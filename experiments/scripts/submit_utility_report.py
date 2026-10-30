#!/usr/bin/env python3
"""Join the utility audit and complete baseline report before submitting Slurm work.

Both completion paths call this helper. Missing prerequisites are recorded as
pending; they are never reported as successful numerical comparisons.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def job_id(value):
    value = str(value).strip().split(';')[0]
    if not value.isascii() or not value.isdecimal():
        raise ValueError('Invalid Slurm job identifier')
    return value


def audit_accounting(array):
    return subprocess.check_output(
        ['sacct', '--array', '-X', '-n', '-P', '-j', array,
         '--format=JobID,State,ExitCode'], text=True)


def audit_dependency(array):
    # Completed IDs can leave slurmctld after MinJobAge. Recreating an afterok
    # dependency hours later is then rejected; accept only complete, successful
    # accounting for every prescribed audit task before omitting that dependency.
    expected = {f'{array}_{i}' for i in range(23)}
    records = {}
    for line in audit_accounting(array).splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 3 and fields[0] in expected:
            records[fields[0]] = tuple(fields[1:3])
    if set(records) == expected and all(value == ('COMPLETED', '0:0') for value in records.values()):
        return None
    failed = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL'}
    if any(state.split()[0] in failed or (state == 'COMPLETED' and code != '0:0')
           for state, code in records.values()):
        raise RuntimeError('Utility audit tasks failed; aggregate report not submitted')
    return 'afterok:' + array


def submit_report(root=ROOT, audit_array=None, scheduler=None):
    root = Path(root)
    finished = root / 'results/nvidia_layer_validation/utility_v1/finished'
    finished.mkdir(parents=True, exist_ok=True)
    record_path = finished / 'audit_submission.json'
    baseline = root / 'results/baselines/_summary_reduced'

    def save(record):
        temporary = record_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(record_path)

    with (finished / 'report_submission.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        if audit_array is not None:
            audit_array = job_id(audit_array)
            if record.get('report') and record.get('audit_array') != audit_array:
                raise RuntimeError('An aggregate report is already registered; '
                                   'rerun its report launcher explicitly if required.')
            record.update(audit_array=audit_array)
        if not record.get('audit_array'):
            return {'status': 'waiting_for_audit', 'report': None}
        array = job_id(record['audit_array'])
        if record.get('report'):
            job_id(record['report'])
            return record
        status_path = baseline / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if (status.get('all_complete') is not True or status.get('missing')
                or not (baseline / 'scores.csv').is_file()):
            record.update(report=None, status='waiting_for_baselines')
            save(record)
            return record
        # Persist the audit dependency even when scheduler submission fails.
        record.update(report=None, status='ready_for_submission')
        save(record)
        dependency = audit_dependency(array)
        command = ['sbatch', '--parsable']
        if dependency is not None:
            command.append('--dependency=' + dependency)
        command.append(str(root / 'jobs/nvidia_utility_report.sbatch'))
        if scheduler is None:
            scheduler = lambda args: subprocess.check_output(args, cwd=root, text=True)
        report = job_id(scheduler(command))
        record.update(report=report, status='submitted', audit_dependency=dependency,
                      audits_verified_completed=dependency is None,
                      baseline_scores_sha256=hashlib.sha256((baseline / 'scores.csv').read_bytes()).hexdigest(),
                      baseline_status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest())
        save(record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-array', help='Register an already submitted utility audit array')
    args = parser.parse_args()
    print(json.dumps(submit_report(audit_array=args.audit_array), indent=2))


if __name__ == '__main__':
    main()
