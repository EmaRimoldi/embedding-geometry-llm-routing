#!/usr/bin/env python3
"""Rerun complete metadata lookups and the MMLU topic diagnostic on canonical inputs.

Production sources run unchanged in a new isolated workspace. No embeddings or
GPU are needed; the CPU experiment environment includes their import dependencies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, context, counts):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Report keys differ: " + context)
        for key in expected:
            compare(actual[key], expected[key], context + "/" + key, counts)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("Report lengths differ: " + context)
        for index, (value, reference) in enumerate(zip(actual, expected)):
            compare(value, reference, context + "/" + str(index), counts)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(actual, (float, int)) or isinstance(actual, bool):
            raise ValueError("Report numeric type differs: " + context)
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError("Non-finite report value: " + context)
        error = abs(actual - expected)
        if isinstance(expected, int):
            equal = actual == expected
        else:
            equal = math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
        if not equal:
            raise ValueError(f"Report value differs at {context}: {actual} != {expected}")
        counts["numeric_values"] += 1
        counts["maximum_absolute_error"] = max(counts["maximum_absolute_error"], error)
    elif actual != expected:
        raise ValueError("Report metadata differs: " + context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced/cpu-diagnostics.json")
    args = parser.parse_args()
    reference_path = ROOT / "reproducibility/provenance/reviewer_smoke/public_data_preparation.json"
    reference = json.loads(reference_path.read_text())
    verified = {}
    for dataset, record in reference["datasets"].items():
        for filename, hashes in record["file_hashes"].items():
            relative = f"datasets/{dataset}/prepared/{filename}"
            path = ROOT / relative
            if not path.is_file() or sha(path) != hashes["canonical"]:
                raise ValueError("Missing or changed canonical input: " + relative)
            verified[relative] = sha(path)
    cache = ROOT / ".cache/fresh-cpu-diagnostics"
    cache.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
    sources = ["src/baselines/common.py", "src/baselines/run_lookups.py",
               "src/baselines/run_mlp.py", "src/baselines/batched_mlp.py",
               "src/diagnostics/mmlu_topic_preference.py", "configs/baselines.json",
               "configs/baseline_splits/routerbench-0shot.json",
               "configs/baseline_splits/sprout-sixfam.json"]
    for relative in sources:
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    for dataset in reference["datasets"]:
        target = workspace / "datasets" / dataset / "prepared"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(ROOT / "datasets" / dataset / "prepared", target.parent),
                          target_is_directory=True)
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    environment.pop("PYTHONPATH", None)
    commands = [
        [sys.executable, "-B", "-c", "import sys; sys.path.insert(0, 'src/baselines'); "
         "import run_lookups; from run_mlp import tasks; "
         "[run_lookups.main(i) for i in range(len(tasks()))]"],
        [sys.executable, "-B", "src/diagnostics/mmlu_topic_preference.py"],
    ]
    timings = []
    for index, command in enumerate(commands):
        print("Running", ["all IID metadata lookups", "full MMLU topic diagnostic"][index], flush=True)
        start = time.perf_counter()
        with (workspace / f"execution-{index}.log").open("w") as log:
            subprocess.run(command, cwd=workspace, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        timings.append(time.perf_counter() - start)
    files = sorted((workspace / "results/baselines").rglob("lookups/result.json"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 complete IID lookup reports, found {len(files)}")
    files.append(workspace / "results/geometry/_diagnostics/mmlu_topic_preference/report.json")
    counts = {"numeric_values": 0, "maximum_absolute_error": 0.0}
    reports = []
    for path in files:
        relative = path.relative_to(workspace / "results")
        frozen = ROOT / "reproducibility/evidence" / relative
        actual, expected = json.loads(path.read_text()), json.loads(frozen.read_text())
        # Original-run source fingerprints remain historical. New source hashes
        # are recorded below; every scientific value and other field is compared.
        actual.pop("provenance", None)
        expected.pop("provenance", None)
        compare(actual, expected, str(relative), counts)
        reports.append({"path": str(relative), "fresh_sha256": sha(path),
                        "reference_sha256": sha(frozen)})
    for relative, expected in verified.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Canonical input changed during execution: " + relative)
    report = {"passed": True, "scope": "Complete production IID metadata lookups and MMLU topic diagnostic; "
              "all prescribed penalties, full canonical cohorts and 1,000 MMLU paired bootstrap draws per penalty. "
              "No embedding-based router training or full-study rerun.",
              "input_origin": "Previously downloaded public assets; prepared bytes verified against canonical hashes.",
              "lookup_reports": 20, "lookup_policy_scores": 520, "mmlu_bootstrap_replicates_per_penalty": 1000,
              "reports_compared": len(reports), **counts, "relative_tolerance": 1e-10, "absolute_tolerance": 1e-12,
              "comparison_excludes": ["Historical MMLU provenance field; new source/input fingerprints recorded separately"],
              "python": sys.version.split()[0],
              "verification_code_sha256": sha(Path(__file__)),
              "elapsed_seconds": sum(timings),
              "stage_seconds": dict(zip(["iid_metadata_lookups", "mmlu_topic_bootstrap"], timings)),
              "packages": {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "torch"]},
              "verified_inputs": verified, "source_hashes": {rel: sha(workspace / rel) for rel in sources},
              "reports": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ["passed", "reports_compared", "numeric_values", "maximum_absolute_error"]}))


if __name__ == "__main__":
    main()
