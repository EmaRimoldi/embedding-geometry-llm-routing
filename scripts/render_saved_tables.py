#!/usr/bin/env python3
"""Render selected tables from bundled reports in isolated scratch directories."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/evidence"
CI = "diagnostics/appendix_confidence_intervals"
FAMILY = "geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer"
PANELS = ("routerbench-0shot", "sprout-sixfam")
DATASETS = ("routerbench", "sprout")
SPECS = [
    dict(name="appendix_ci", script="src/diagnostics/appendix_ci/render_tables.py",
         sources=["src/diagnostics/appendix_ci/ci_common.py"],
         configs=["configs/appendix_ci.json", "configs/baselines.json"],
         evidence=[f"{CI}/routing/{p}/{s}/report.json" for p in PANELS for s in ("iid", "ood")]
                  + [f"{CI}/geometry/{d}/report.json" for d in DATASETS],
         outputs=[f"{CI}/overleaf/{n}" for n in
                  ("aistats_ci_tables.tex", "aistats_tables.tex", "pointmap.json", "macros.json", "audit.json")]),
    dict(name="sharedtrunknet", script="src/diagnostics/appendix_ci/render_sharedtrunknet.py",
         sources=[], configs=[f"configs/baseline_splits/{p}.json" for p in PANELS],
         evidence=[f"{CI}/sharedtrunknet/report.json", "layer_sweep/summary/source_selection_audit.json"],
         outputs=[f"{CI}/sharedtrunknet/{n}" for n in ("aistats_sharedtrunknet.tex", "layer_selected_families.tex")]),
    dict(name="cost_reward", script="src/diagnostics/render_knn_cost_reward_format.py",
         sources=[], configs=[],
         evidence=[f"diagnostics/knn_cost_reward_format/{d}/{n}" for d in DATASETS
                   for n in ("report.json", "complete.json")],
         outputs=["diagnostics/knn_cost_reward_format/paper/aistats_cost_reward.tex"]),
    dict(name="family_utility", script="src/diagnostics/export_family_utility_tables.py",
         sources=[], configs=[],
         evidence=[f"{FAMILY}/{n}" for n in ("audit.json", "summary.json", "per_lambda.csv")],
         outputs=[f"{FAMILY}/{n}" for n in ("per_lambda_table.tex", "summary_table.tex", "table_provenance.json")]),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(spec, run):
    name = spec["name"]
    record = dict(family=name, status="pending", inputs={}, outputs={})
    present = [p for p in spec["evidence"] if (EVIDENCE / p).is_file()]
    if not present:
        record.update(status="skipped", reason="No evidence for this family is bundled.",
                      missing=["results/evidence/" + p for p in spec["evidence"]])
        return record
    files = [(ROOT / p, p) for p in [spec["script"], *spec["sources"], *spec["configs"]]]
    files += [(EVIDENCE / p, "results/" + p) for p in spec["evidence"]]
    missing = [str(p.relative_to(ROOT)) for p, _ in files if not p.is_file()]
    if missing:
        record.update(status="missing", reason="Required source, configuration, or evidence is absent.", missing=missing)
        return record
    work = run / name
    work.mkdir()
    for source, target in files:
        target = work / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        record["inputs"][str(source.relative_to(ROOT))] = digest(target)
    cache = work / ".cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", XDG_CACHE_HOME=str(cache),
               MPLCONFIGDIR=str(cache / "matplotlib"), TMPDIR=str(cache))
    # Clear external import paths: the staged renderer uses its colocated helpers.
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-B", str(work / spec["script"])]
    try:
        completed = subprocess.run(command, cwd=work, env=env, capture_output=True,
                                   text=True, timeout=60, check=False)
        (work / "stdout.log").write_text(completed.stdout)
        (work / "stderr.log").write_text(completed.stderr)
        record["returncode"] = completed.returncode
        record["logs"] = [str((work / f"{s}.log").relative_to(ROOT)) for s in ("stdout", "stderr")]
        if completed.returncode:
            record.update(status="failed", reason="Renderer returned a nonzero exit code; see logs.")
            return record
    except subprocess.TimeoutExpired:
        record.update(status="failed", reason="Renderer exceeded the 60-second limit.")
        return record
    absent = []
    for relative in spec["outputs"]:
        output = work / "results" / relative
        if output.is_file() and output.stat().st_size:
            record["outputs"][str(output.relative_to(ROOT))] = digest(output)
        else:
            absent.append("results/" + relative)
    record.update(status="failed" if absent else "passed")
    if absent:
        record.update(reason="Renderer did not produce every expected nonempty output.", missing_outputs=absent)
    return record


def main():
    base = ROOT / "results/reproduced/saved_tables"
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    records = []
    for spec in SPECS:
        record = render(spec, run)
        records.append(record)
        print(f"{record['status'].upper()}: {record['family']}")
        if record.get("reason"):
            print("  " + record["reason"])
        for name in record.get("missing", []) + record.get("missing_outputs", []):
            print("  missing: " + name)
    report = dict(scope="Render four selected table families from saved reports; no experiments or intervals are recomputed.",
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python_version=sys.version.split()[0], families=records,
                  counts={s: sum(r["status"] == s for r in records)
                          for s in ("passed", "skipped", "missing", "failed")})
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Report: " + str((run / "report.json").relative_to(ROOT)))
    return int(any(r["status"] in ("missing", "failed") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
