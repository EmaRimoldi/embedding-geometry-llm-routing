"""Run the user-reviewed frozen experiment, evaluate, then archive on CLUSTER.

This wrapper changes no inference parameters. It never retries or resumes a
failed inference run automatically. Launch it with detached stdio for durability.
"""
import datetime
import json
from pathlib import Path
import subprocess
import sys

from core import write_json

ROOT = Path(__file__).resolve().parent
REMOTE = '__REMOTE_HOST__:__E4_ROOT__/gpt56_lookup_reasoning_20260923/'


def main():
    control = ROOT / 'runs/full_codex_v1_control'
    state_path = control / 'pipeline.json'
    state = json.loads(state_path.read_text())
    if state['status'] != 'launching':
        raise RuntimeError('This pipeline has already started; refusing duplicate inference')
    prepared = ROOT / 'runs/test_prepared_v1'
    inference = ROOT / 'runs/full_codex_v1'
    evaluation = ROOT / 'runs/full_evaluation_v1'
    if inference.exists() or evaluation.exists():
        raise FileExistsError('Unique inference and evaluation output paths required')
    wake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(state['pipeline_pid'])],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    state.update(status='inference', idle_sleep_inhibitor_pid=wake.pid)
    write_json(state_path, state)
    try:
        subprocess.run([sys.executable, '-u', str(ROOT/'run_gpt_router.py'),
                        '--prepared', str(prepared), '--output', str(inference),
                        '--reviewed-full-run', '--max-calls', '41580'], check=True)
        state['status'] = 'evaluation'
        write_json(state_path, state)
        subprocess.run([sys.executable, '-u', str(ROOT/'evaluate.py'),
                        '--prepared', str(prepared), '--inference', str(inference),
                        '--data', str(ROOT/'data'), '--output', str(evaluation)], check=True)
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(state_path, state)
        # Existing frozen data/packets are already on CLUSTER. Preserve local
        # artifacts even if the SSH connection has expired by completion.
        try:
            copied = subprocess.run(['rsync', '-az', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                                     '--exclude', 'data', '--exclude', 'codex-schema',
                                     '--exclude', '__pycache__', '--exclude', '.pytest_cache',
                                     str(ROOT)+'/', REMOTE], timeout=300)
            state['final_archive_exit_code'] = copied.returncode
        except Exception as exc:
            state['final_archive_error'] = f'{type(exc).__name__}: {exc}'
        write_json(state_path, state)
        wake.terminate()


if __name__ == '__main__':
    main()
