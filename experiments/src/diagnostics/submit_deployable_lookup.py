"""Submit the fixed IID lookup diagnostic; run only on a configured Slurm cluster."""
from pathlib import Path
import datetime, hashlib, json, os, shlex, subprocess
root = Path(__file__).resolve().parents[2]
out = root / "results/diagnostics/deployable_lookup"
out.mkdir(parents=True, exist_ok=True)
(root / "logs").mkdir(exist_ok=True)
jid = subprocess.check_output(
    ["sbatch", "--parsable", "--array=0-9%2", "jobs/deployable_lookup.sbatch"],
    cwd=root, text=True).strip().split(";")[0]
inner = "source " + shlex.quote(str(root / "jobs/env.sh")) + ' && "$PAPER_PYTHON" -u src/diagnostics/deployable_lookup.py --report'
wrap = "bash -c " + shlex.quote(inner)
wrap = "bash -c " + shlex.quote(wrap)
report = subprocess.check_output([
    "sbatch", "--parsable", "--cpus-per-task=2", "--mem=4G", "--time=00:10:00",
    "--dependency=afterok:" + jid, "--job-name=label-report",
    "--output=logs/label-report-%j.log", "--wrap", wrap], cwd=root, text=True).strip().split(";")[0]
record = dict(array=jid, report_job=report, expected=10,
    source_sha256=hashlib.sha256((root / "src/diagnostics/deployable_lookup.py").read_bytes()).hexdigest(),
    submitted=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out / "submission.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
