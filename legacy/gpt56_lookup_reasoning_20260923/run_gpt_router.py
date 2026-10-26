"""Bounded pilot or explicitly released full run through isolated Codex inference."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from codex_backend import CodexClient
from core import canonical, digest, parse_prediction, sha_file, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--reviewed-full-run', action='store_true')
    ap.add_argument('--max-calls', type=int, default=28)
    args = ap.parse_args()
    prepared, out = args.prepared.resolve(), args.output.resolve()
    cfg = json.loads((prepared / 'config.json').read_text())
    manifest = json.loads((prepared / 'split_manifest.json').read_text())
    dry = json.loads((prepared / 'dry_run_report.json').read_text())
    schema = json.loads((prepared / 'output_schema.json').read_text())
    system = (prepared / 'system.txt').read_text()
    if sha_file(prepared / 'evidence.jsonl') != dry['evidence_sha256']:
        raise ValueError('Evidence file changed after freezing')
    if digest(canonical(cfg)) != manifest['config_hash']:
        raise ValueError('Configuration changed after preprocessing')
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return
    if manifest['stage'] == 'test' and not args.reviewed_full_run:
        raise ValueError('Full test requires review of pilot and usage estimate first')
    if out.exists():
        raise FileExistsError('Unique output directory required; never overwrite previous inference')
    out.mkdir(parents=True)
    snapshot = out / 'code_snapshot'
    snapshot.mkdir()
    for path in Path(__file__).resolve().parent.glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    run = {'run_id': uuid.uuid4().hex, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'config': cfg, 'prepared': str(prepared), 'prepared_manifest_sha256': sha_file(prepared / 'split_manifest.json'),
           'codex_version': subprocess.check_output(['codex', '--version'], text=True).strip(),
           'max_calls': args.max_calls, 'status': 'running'}
    run['code_hashes'] = {p.name: sha_file(p) for p in sorted(snapshot.glob('*.py'))}
    write_json(out / 'run.json', run)
    calls = 0
    # Same empty directory for both conditions: no per-effort path in model context.
    client = None
    try:
        client = CodexClient(out / 'empty_inference_workspace', cfg['request_timeout_seconds'])
        with open(out / 'predictions.jsonl', 'x') as predictions, open(out / 'request_metadata.jsonl', 'x') as log:
            for qi, line in enumerate((prepared / 'evidence.jsonl').open()):
                e = json.loads(line)
                if digest(canonical(e['packet'])) != e['evidence_hash'] or digest(e['prompt']) != e['prompt_hash']:
                    raise ValueError('Packet or prompt hash mismatch')
                sid = e['packet']['sample_id']
                # Counterbalance temporal order using a predeclared outcome-free hash.
                order = cfg['efforts'] if int(digest(sid)[0], 16) % 2 == 0 else list(reversed(cfg['efforts']))
                paired = {}
                for effort in order:
                    for attempt in range(cfg['max_attempts']):
                        if calls >= args.max_calls:
                            raise RuntimeError('Declared call budget reached')
                        calls += 1
                        record = {'sample_id': sid, 'effort': effort, 'attempt': attempt,
                                  'evidence_hash': e['evidence_hash'], 'prompt_hash': e['prompt_hash'],
                                  'final_k': e['final_k'], 'prompt_tokens_estimate': e['prompt_tokens_estimate']}
                        try:
                            proof = client.infer(e['prompt'], schema, system, cfg, effort)
                            record.update(proof)
                            prediction = parse_prediction(proof['raw_structured_response'], sid, manifest['models'])
                            record['validation_error'] = None
                            paired[effort] = dict(sample_id=sid, effort=effort,
                                                  evidence_hash=e['evidence_hash'], prediction=prediction)
                            log.write(canonical(record) + '\n'); log.flush()
                            break
                        except Exception as exc:
                            record['validation_error'] = f'{type(exc).__name__}: {exc}'
                            log.write(canonical(record) + '\n'); log.flush()
                            # Protocol/tool/model violations fail closed; only output parse validation can retry.
                            if not isinstance(exc, (ValueError, json.JSONDecodeError)) or attempt + 1 == cfg['max_attempts']:
                                raise
                            time.sleep(1)
                    print(f'query {qi+1}: {effort}, tokens={record.get("total_tokens")}, reasoning={record.get("reasoning_tokens")}', flush=True)
                assert len(set(v['evidence_hash'] for v in paired.values())) == 1
                for effort in cfg['efforts']:
                    predictions.write(canonical(paired[effort]) + '\n')
                predictions.flush()
        run['status'] = 'complete'
    except Exception as exc:
        run['status'] = 'failed'
        run['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if client is not None:
            client.close()
        run['calls_attempted'] = calls
        run['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out / 'run.json', run)


if __name__ == '__main__':
    main()
