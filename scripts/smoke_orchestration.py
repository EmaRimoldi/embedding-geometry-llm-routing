#!/usr/bin/env python3
"""Exercise release orchestration with tiny fixtures and mocked compute/Slurm.

No real dataset, model, fit, scheduler submission or original research directory
is used. All generated fixtures and logs stay under this release's .cache.
"""
import argparse
import ast
import copy
import collections
import contextlib
import io
import time
import types
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "results/baselines/_run/source_hashes.json"
NV = "results/nvidia/provenance/source_hashes.json"
UTILITY = "results/nvidia_layer_validation/utility_v1/provenance/source_hashes.json"
REDUCED_CONFIG = "configs/mlp_reduced.json"
NV_CONFIG = "configs/nvidia.json"
UTILITY_CONFIG = "configs/nvidia_utility_v1.json"
DATA = "datasets/toy/prepared/panel.npz"
MANIFESTS = {
    "layer-sweep": ["results/layer_sweep/provenance/source_hashes.json"],
    "format": ["results/geometry/_diagnostics/format_factorial_v1/source_hashes.json"],
    "proof-twins": ["results/geometry/_diagnostics/proof_twins_v1/source_hashes.json"],
    "family-utility": ["results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer/source_hashes.json"],
    "inference": ["results/nvidia/provenance/input_hashes.json",
                  "results/nvidia/inference_cpu_v1/source_hashes.json",
                  "results/nvidia/inference_cpu_v1/audit_source_hashes.json"],
}
RESULTS = []

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def put(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n" if not isinstance(value, str) else value)

def read(root, path):
    return json.loads((root / path).read_text())

def fixture(parent, label):
    root = parent / label
    root.mkdir()
    (root / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/initialize_release.py", root / "scripts/initialize_release.py")
    shutil.copytree(ROOT / "src/nvidia_utility/vendor", root / "src/nvidia_utility/vendor")
    put(root, "src/fixture.py", "# synthetic orchestration fixture\n")
    put(root, "jobs/fixture.sbatch", "# synthetic source file\n")
    put(root, DATA, "small synthetic bytes; not a real NPZ dataset\n")
    put(root, "configs/geometry.json", {"fixture_only": True})
    put(root, REDUCED_CONFIG, {"source_hashes": {"jobs/fixture.sbatch": "old"},
        "original_source_hashes_sha256": "old", "method_fields": {"seed": 42, "folds": [[0], [1]]}})
    put(root, NV_CONFIG, {"reduced_mlp_config_sha256": "old", "method_fields": {"layers": [15, 28]}})
    put(root, UTILITY_CONFIG, {"input_hashes": {DATA: digest(root / DATA), NV_CONFIG: "old"},
        "method_fields": {"loss": "MSE", "seed": 42}})
    templates = {p: ["src/fixture.py"] for names in MANIFESTS.values() for p in names}
    templates[UTILITY] = ["src/fixture.py", UTILITY_CONFIG]
    put(root, "environment/runtime_manifest_paths.json", templates)
    return root

def run(root, stage, expected=None):
    config_before = {p: (root / p).read_bytes() for p in [REDUCED_CONFIG, NV_CONFIG, UTILITY_CONFIG]}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(root / ".cache/tmp"))
    process = subprocess.run([sys.executable, "-B", str(root / "scripts/initialize_release.py"),
                              "--stage", stage], cwd=root, env=env, text=True, capture_output=True)
    if expected:
        assert process.returncode != 0, (stage, "unexpected success")
        assert expected in process.stderr, (stage, expected, process.stderr)
        assert config_before == {p: (root / p).read_bytes() for p in config_before}, "failure changed config"
    else:
        assert process.returncode == 0, (stage, process.stderr)
    return process

def base_ready(root):
    put(root, BASE, {"src/fixture.py": digest(root / "src/fixture.py")})

def reduced_ready(root):
    base_ready(root)
    run(root, "reduced")

def nvidia_ready(root):
    reduced_ready(root)
    run(root, "nvidia")
    put(root, NV, {NV_CONFIG: digest(root / NV_CONFIG), "src/fixture.py": digest(root / "src/fixture.py")})

def passed(name):
    RESULTS.append({"check": name, "passed": True})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/reviewer_smoke_orchestration/report.json")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    assert output.is_relative_to(ROOT), "output must remain within this release"
    cache = ROOT / ".cache/reviewer_smoke_orchestration"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixtures-", dir=cache) as temporary:
        parent = Path(temporary)
        root = fixture(parent, "happy")
        methods = {p: copy.deepcopy(read(root, p)["method_fields"])
                   for p in [REDUCED_CONFIG, NV_CONFIG, UTILITY_CONFIG]}
        run(root, "layout"); run(root, "layout")
        for name in ["datasets", "embedders", "artifacts"]:
            link = root / "artifacts/format_factorial_runtime" / name
            assert link.is_symlink() and not os.path.isabs(os.readlink(link))
            assert link.resolve() == (root / name).resolve()
        passed("layout idempotency and relative internal runtime links")
        run(root, "reduced", "No such file or directory")
        run(root, "nvidia", "No such file or directory")
        run(root, "utility", "No such file or directory")
        passed("missing preparation fails without changing configs")
        base_ready(root)
        run(root, "nvidia", "Run the reduced provenance stage")
        passed("NVIDIA rejects missing reduced provenance stage")
        run(root, "reduced")
        run(root, "nvidia")
        put(root, NV, {NV_CONFIG: digest(root / NV_CONFIG), "src/fixture.py": digest(root / "src/fixture.py")})
        run(root, "utility")
        for stage in MANIFESTS: run(root, stage); run(root, stage)
        snapshots = {p: (root / p).read_bytes() for p in [REDUCED_CONFIG, NV_CONFIG, UTILITY_CONFIG]}
        run(root, "reduced"); run(root, "nvidia"); run(root, "utility")
        assert snapshots == {p: (root / p).read_bytes() for p in snapshots}
        assert methods == {p: read(root, p)["method_fields"] for p in methods}
        for path, value in read(root, UTILITY).items(): assert digest(root / path) == value
        passed("all provenance stages and exact config/hash idempotency")
        put(root, "results/baselines/toy/mlp/result.json", {})
        put(root, "results/nvidia/results/toy/result.json", {})
        put(root, "results/nvidia_layer_validation/utility_v1/toy/result.json", {})
        run(root, "reduced"); run(root, "nvidia"); run(root, "utility")
        passed("unchanged metadata can be rechecked after results exist")

        root = fixture(parent, "baseline-partial")
        base_ready(root)
        put(root, "results/baselines/toy/mlp/cache/seed_42.json", {})
        run(root, "reduced", "Refusing to change run metadata after output exists")
        passed("partial baseline checkpoint prevents provenance rebinding")

        root = fixture(parent, "pca-partial")
        reduced_ready(root)
        put(root, "results/nvidia/pca/toy/layer_21/ready.json", {})
        run(root, "nvidia", "Refusing to change run metadata after output exists")
        passed("PCA cache prevents NVIDIA provenance rebinding")

        root = fixture(parent, "utility-partial")
        nvidia_ready(root)
        put(root, "results/nvidia_layer_validation/utility_v1/toy/fits/seed_42.json", {})
        run(root, "utility", "Refusing to change run metadata after output exists")
        passed("partial utility checkpoint prevents provenance rebinding")

        root = fixture(parent, "utility-missing-source")
        nvidia_ready(root)
        templates = read(root, "environment/runtime_manifest_paths.json")
        templates[UTILITY].append("src/intentionally_missing.py")
        put(root, "environment/runtime_manifest_paths.json", templates)
        run(root, "utility", "Missing prerequisite")
        assert not (root / UTILITY).exists()
        passed("missing source leaves utility config and manifest untouched")

        root = fixture(parent, "utility-stale-manifest")
        nvidia_ready(root)
        put(root, UTILITY, {"old_source": "obsolete"})
        run(root, "utility", "Existing runtime manifest differs")
        assert read(root, UTILITY) == {"old_source": "obsolete"}
        passed("stale source manifest is preserved without partial config update")

        root = fixture(parent, "utility-changed-data")
        nvidia_ready(root)
        put(root, DATA, "changed synthetic dataset")
        run(root, "utility", "Frozen input mismatch")
        passed("changed frozen data rejected rather than rehashed")

        root = fixture(parent, "changed-prerequisite")
        nvidia_ready(root)
        put(root, "src/fixture.py", "# changed source")
        run(root, "utility", "Prerequisite source manifest differs")
        passed("stale NVIDIA source provenance rejected")

        root = fixture(parent, "bad-runtime-link")
        run(root, "layout")
        link = root / "artifacts/format_factorial_runtime/datasets"
        link.unlink()
        link.symlink_to(parent, target_is_directory=True)
        run(root, "layout", "Unsafe or unexpected runtime path")
        passed("external runtime link rejected")

        root = fixture(parent, "vendor-changed")
        vendor = root / "src/nvidia_utility/vendor/trunk.py"
        vendor.write_bytes(vendor.read_bytes() + b"\n# altered fixture")
        run(root, "layout", "Vendor source checksum mismatch")
        passed("altered upstream vendor code rejected")

        # Execute actual launchers against inert command stubs, never real compute.
        root = fixture(parent, "mock-launchers")
        run(root, "layout")
        shutil.copytree(ROOT / "jobs", root / "jobs", dirs_exist_ok=True)
        bindir = root / "mock-bin"; bindir.mkdir()
        for name, body in {
            "python": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$SMOKE_CALL_LOG"\n',
            "sbatch": '#!/bin/bash\nprintf "sbatch %s\\n" "$*" >> "$SMOKE_CALL_LOG"\nprintf "12345\\n"\n',
        }.items():
            path = bindir / name; path.write_text(body); path.chmod(0o755)
        env = dict(os.environ, PAPER_ROOT="/stale/inherited/root", PAPER_PYTHON=str(bindir / "python"),
                   SLURM_SUBMIT_DIR=str(root), SLURM_ARRAY_TASK_ID="0", SLURM_JOB_ID="999999",
                   PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                   SMOKE_CALL_LOG=str(root / "mock-calls.log"))
        launchers = sorted((root / "jobs").glob("*.sbatch"))
        for path in launchers:
            process = subprocess.run(["bash", str(path), "0"], cwd=root, env=env,
                                     capture_output=True, text=True, timeout=10)
            assert process.returncode == 0, (path.name, process.stderr)
        # Manual invocation within an older allocation must use the current copy.
        stale_env = dict(env, SLURM_SUBMIT_DIR="/stale/inherited/allocation")
        process = subprocess.run(["bash", str(root / "jobs/encode.sbatch"), "0"],
                                 cwd=root, env=stale_env, capture_output=True, text=True, timeout=10)
        assert process.returncode == 0, process.stderr
        calls = (root / "mock-calls.log").read_text()
        assert "src/encode.py" in calls and "src/nvidia_utility/train.py" in calls
        assert "/stale/inherited/root" not in calls
        passed(str(len(launchers)) + " Slurm launchers execute with inert Python/sbatch stubs")

        # Run the actual accounting function with minimal inputs and a forbidden
        # scheduler stub. Missing IDs must be explicit, never a broad sacct query.
        root = fixture(parent, "accounting")
        out = root / "results/nvidia"; out.mkdir(parents=True)
        tree = ast.parse((ROOT / "src/nvidia/compute_report.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        module = ast.Module(body=functions, type_ignores=[]); ast.fix_missing_locations(module)
        def forbidden_scheduler(*args, **kwargs):
            raise AssertionError("Attempted scheduler access during missing-ID smoke")
        def write_accounting(path, value):
            assert path.resolve().is_relative_to(root)
            put(root, str(path.relative_to(root)), value)
        ns = {"OUT": out, "CONFIG": root / NV_CONFIG, "verify": lambda: None,
              "cfg": lambda: {"tasks": [], "layers": [], "lambdas": {}},
              "sha": digest, "folds": lambda *args: [], "json": json,
              "collections": collections, "time": time,
              "atomic_json": write_accounting,
              "subprocess": types.SimpleNamespace(check_output=forbidden_scheduler,
                                                  CalledProcessError=subprocess.CalledProcessError)}
        exec(compile(module, "accounting-fixture", "exec"), ns)
        with contextlib.redirect_stdout(io.StringIO()): ns["main"]()
        assert "submission.json" in read(root, "results/nvidia/timing/summary.json")["scheduler_error"]
        assert "Scheduler accounting unavailable" in (out / "timing/summary.md").read_text()
        put(root, "results/nvidia/submission.json", {"training_workers": []})
        with contextlib.redirect_stdout(io.StringIO()): ns["main"]()
        assert "no study job IDs" in read(root, "results/nvidia/timing/summary.json")["scheduler_error"]
        passed("missing/empty scheduler metadata produces explicit unavailable accounting without sacct")

        docs = (ROOT / "docs/guides/RUN_COMMANDS.md").read_text()
        commands = sorted(set(re.findall(r"(?:src|scripts|jobs)/[A-Za-z0-9_./-]+", docs)))
        command_files = [s for s in commands if s.endswith((".py", ".sbatch", ".sh"))]
        assert all((ROOT / p).is_file() for p in command_files), [p for p in command_files if not (ROOT / p).is_file()]
        expected = {"geometry_tasks.json": 207, "baseline_tasks.json": 46,
                    "descriptive_clustering_tasks.json": 18, "additional_clustering_tasks.json": 90}
        for name, size in expected.items():
            assert len(json.loads((ROOT / "configs" / name).read_text())) == size
        source = ast.parse((ROOT / "src/run_geometry_batch.py").read_text())
        fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "assigned_tasks")
        module = ast.Module(body=[fn], type_ignores=[]); ast.fix_missing_locations(module)
        ns = {}; exec(compile(module, "geometry-scheduler", "exec"), ns)
        assert sorted(i for worker in range(20) for i in ns["assigned_tasks"](worker, 207)) == list(range(207))
        passed("documented paths, frozen task counts and complete geometry batch coverage")

    report = {"status": "passed", "scope": "synthetic orchestration only; no real jobs/data/model fits",
              "checks": RESULTS, "checks_passed": len(RESULTS),
              "initializer_sha256": digest(ROOT / "scripts/initialize_release.py"),
              "harness_sha256": digest(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(RESULTS),
                      "report": str(output.relative_to(ROOT))}))

if __name__ == "__main__":
    main()
