#!/usr/bin/env python3
"""Run the unchanged production PCA100 entrypoint on tiny synthetic CPU inputs.

This smoke needs the compute dependencies but no downloaded model or dataset.
Only temporary fixtures/results below this release's .cache are written.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = "routerbench-0shot"
DATASET = "routerbench"
LAYER = 21
FILES = [
    "src/nvidia/pca.py", "src/nvidia/common_nv.py", "src/nvidia/timing.py",
    "src/baselines/common.py", "src/baselines/run_mlp.py", "src/baselines/batched_mlp.py",
    "configs/baselines.json",
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def load(path):
    return json.loads(path.read_text())

def freeze_sources(root):
    # Same verification contract as the real entrypoint, over its fixture inputs.
    base_files = [p for p in FILES if p.startswith("src/baselines/") or p == "configs/baselines.json"]
    nv_files = [p for p in FILES if p.startswith("src/nvidia/")] + ["configs/nvidia.json"]
    for target, names in [
        ("results/baselines/_run/source_hashes.json", base_files),
        ("results/nvidia/provenance/source_hashes.json", nv_files),
    ]:
        write(root / target, {p: sha(root / p) for p in names})

def fixture(parent, name, raw, train):
    import numpy as np
    root = parent / name
    root.mkdir()
    for rel in FILES:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    prepared = root / "datasets" / DATASET / "prepared"
    prepared.mkdir(parents=True)
    n = len(raw)
    query_file = prepared / "queries.jsonl"
    query_file.write_text("".join(json.dumps({"key": str(i), "prompt": "synthetic row " + str(i)}) + "\n" for i in range(n)))
    np.savez_compressed(prepared / "panel.npz",
        reward=np.zeros((n, 2)), cost=np.zeros((n, 2)),
        family=np.array(["synthetic"] * n), topic=np.array(["fixture"] * n),
        subgroup=np.array(["synthetic::fixture"] * n), keys=np.array([str(i) for i in range(n)]))
    write(prepared / "manifest.json", {"models": ["fixture-a", "fixture-b"], "fixture_only": True})
    write(root / "configs/mlp_reduced.json", {"fixture_only": True})
    key = hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest()[:20]
    config = {"fixture_only": True, "pca_tasks": [[PANEL, LAYER]],
        "source_sets": {PANEL: {key: train.tolist()}},
        "reduced_mlp_config_sha256": sha(root / "configs/mlp_reduced.json")}
    write(root / "configs/nvidia.json", config)
    features = root / "artifacts/embeddings/qwen2.5-7b-instruct" / ("layer_" + str(LAYER)) / DATASET
    features.mkdir(parents=True)
    np.save(features / "vectors.npy", raw)
    write(features / "manifest.json", {
        "chat_template": False, "truncation": False, "layer": LAYER, "rows": n,
        "query_sha256": sha(query_file), "vector_sha256": sha(features / "vectors.npy")})
    freeze_sources(root)
    dest = root / "results/nvidia/pca" / DATASET / key / ("layer_" + str(LAYER))
    return root, features, dest, key

def invoke(root, logs, name, expect_failure=False):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TMPDIR=str(root / ".cache/tmp"),
               PYTHONPYCACHEPREFIX=str(root / ".cache/pycache"),
               XDG_CACHE_HOME=str(root / ".cache"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = [sys.executable, "-B", str(root / "src/nvidia/pca.py"), "--task", "0"]
    process = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=90)
    (logs / (name + ".stdout.txt")).write_text(process.stdout)
    (logs / (name + ".stderr.txt")).write_text(process.stderr)
    if expect_failure:
        assert process.returncode != 0 and "AssertionError" in process.stderr, (name, process.stdout, process.stderr)
    else:
        assert process.returncode == 0, (name, process.stdout, process.stderr)
    return {"name": name, "exit_code": process.returncode, "expected_failure": expect_failure,
            "wall_seconds": time.perf_counter() - started}

def inspect(root, features, dest, key, raw, train):
    import numpy as np
    ready = load(dest / "ready.json")
    projected = np.load(dest / "features.npy")
    with np.load(dest / "transform.npz", allow_pickle=False) as archive:
        transform = {k: archive[k].copy() for k in archive.files}
    assert projected.shape == (len(raw), 100) and projected.dtype == np.float32
    assert np.isfinite(projected).all()
    assert transform["components"].shape == (100, raw.shape[1])
    np.testing.assert_array_equal(transform["train_rows"], train)
    expected_identity = {"config_sha256": sha(root / "configs/nvidia.json"),
                         "feature_sha256": sha(features / "vectors.npy"),
                         "train_row_hash": key, "layer": LAYER, "panel": PANEL}
    assert ready["identity"] == expected_identity
    assert ready["feature_output_sha256"] == sha(dest / "features.npy")
    assert all(ready["timing"][k] >= 0 for k in
               ["scaling_seconds", "pca_fit_seconds", "projection_seconds", "total_seconds"])
    np.testing.assert_allclose(transform["scaler_mean"], raw[train].mean(axis=0, dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(transform["scaler_scale"], raw[train].astype(np.float64).std(axis=0), atol=1e-12)
    # Use the saved transform and sklearn's documented projected-mean order.
    scaled = raw.copy()
    scaled -= transform["scaler_mean"]
    scaled /= transform["scaler_scale"]
    reconstructed = scaled @ transform["components"].T - transform["pca_mean"] @ transform["components"].T
    np.testing.assert_allclose(reconstructed, projected, rtol=3e-4, atol=5e-4)
    return projected, transform, ready, float(np.max(np.abs(reconstructed - projected)))

def main():
    import numpy as np
    import sklearn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".cache/reviewer_pca_smoke/latest_report.json")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--source-rows", type=int, default=160)
    parser.add_argument("--features", type=int, default=512)
    args = parser.parse_args()
    if not (120 <= args.source_rows < args.rows and args.features >= 128):
        parser.error("Need at least 120 source rows, some heldout rows and 128 input features")
    output = (ROOT / args.output).resolve()
    assert output.is_relative_to(ROOT), "output must remain in this release"
    cache = ROOT / ".cache/reviewer_pca_smoke"
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=cache))
    logs = work / "logs"; logs.mkdir()
    report = {"status": "running", "scope": "actual production PCA100 CLI on synthetic CPU fixtures",
              "rows": args.rows, "source_rows": args.source_rows, "heldout_rows": args.rows - args.source_rows, "input_features": args.features, "components": 100,
              "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
              "harness_sha256": sha(Path(__file__)),
              "production_source_sha256": {p: sha(ROOT / p) for p in FILES},
              "commands": [], "checks": [], "run_directory": str(work.relative_to(ROOT))}
    try:
        rng = np.random.default_rng(20261006)
        raw = rng.normal(size=(args.rows, args.features)).astype(np.float32)
        train = np.sort(rng.choice(len(raw), size=args.source_rows, replace=False))
        heldout = np.setdiff1d(np.arange(len(raw)), train)
        poisoned = raw.copy()
        poisoned[heldout] = 1000 * raw[heldout] + 10000
        a, fa, da, key = fixture(work, "original-fixture", raw, train)
        b, fb, db, key_b = fixture(work, "poisoned-fixture", poisoned, train)
        assert key == key_b
        for src in FILES:
            assert sha(a / src) == sha(ROOT / src) == sha(b / src)
        report["commands"].append(invoke(a, logs, "initial-fit"))
        report["commands"].append(invoke(b, logs, "heldout-poison-fit"))
        xa, ta, ra, error_a = inspect(a, fa, da, key, raw, train)
        xb, tb, rb, error_b = inspect(b, fb, db, key, poisoned, train)
        for field in ta:
            np.testing.assert_array_equal(ta[field], tb[field])
        np.testing.assert_array_equal(xa[train], xb[train])
        assert np.max(np.abs(xa[heldout] - xb[heldout])) > 1
        report["checks"] += ["unchanged production entrypoint and dependency bytes",
            "PCA100 output shape and finite float32 projections",
            "saved source rows, scaler moments and PCA transform reconstructed",
            "ready config/vector/source-row hashes and projected-vector hash verified",
            "poisoning heldout vectors leaves all fitted transform arrays exactly unchanged",
            "poisoning heldout vectors leaves source-row projections exactly unchanged",
            "heldout projections change under poisoning"]
        cache_hashes = {p.name: sha(p) for p in da.iterdir() if p.is_file()}
        report["commands"].append(invoke(a, logs, "completed-cache-reuse"))
        assert cache_hashes == {p.name: sha(p) for p in da.iterdir() if p.is_file()}
        report["checks"].append("completed cache is reused without changing saved bytes")
        # Change only mock config metadata and re-freeze fixture source hashes.
        # The existing PCA identity must then prevent mixing old and new caches.
        changed = load(a / "configs/nvidia.json"); changed["fixture_revision"] = 2
        write(a / "configs/nvidia.json", changed); freeze_sources(a)
        report["commands"].append(invoke(a, logs, "stale-cache-rejected", expect_failure=True))
        assert cache_hashes == {p.name: sha(p) for p in da.iterdir() if p.is_file()}
        report["checks"].append("stale config-bound cache rejected without overwriting artifacts")
        # A raw-vector mutation not accompanied by its declared manifest must fail.
        np.save(fb / "vectors.npy", raw)
        report["commands"].append(invoke(b, logs, "vector-manifest-mismatch-rejected", expect_failure=True))
        report["checks"].append("raw-vector checksum mismatch rejected")
        report.update(status="passed", checks_passed=len(report["checks"]),
            solver=ra["solver"], maximum_reconstruction_error={"ordinary": error_a, "poisoned": error_b},
            training_projection_max_difference=float(np.max(np.abs(xa[train] - xb[train]))))
    except Exception:
        report["status"] = "failed"
        report["failure"] = traceback.format_exc()
        write(work / "report.json", report); write(output, report)
        print(json.dumps({"status": "failed", "report": str(output.relative_to(ROOT)),
                          "preserved_run": str(work.relative_to(ROOT))}))
        raise
    write(work / "report.json", report); write(output, report)
    # Keep small report and subprocess logs; discard only this run's mock inputs.
    shutil.rmtree(a); shutil.rmtree(b)
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"],
                      "report": str(output.relative_to(ROOT))}))

if __name__ == "__main__":
    main()
