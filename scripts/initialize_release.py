"""Initialize portable runtime folders and explicitly freeze new-run provenance.

This does not download assets, train models, submit jobs or change recorded evidence.
Only provenance fields in copied configs may be refreshed by reduced/nvidia/utility.
Every data/split hash in the utility protocol remains an exact-input check.
Run in a fresh reproduction copy: reproducibility/evidence is historical, runtime results
are generated anew. Mismatching existing runtime manifests are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel):
    return json.loads((ROOT / rel).read_text())

def sha(rel):
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError("Missing prerequisite: " + str(rel))
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_new(rel, obj):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if read(rel) != obj:
            raise RuntimeError("Existing runtime manifest differs; use a fresh reproduction copy: " + rel)
        return
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def replace_config(rel, obj):
    # Only callers below may change explicit metadata fields; no method fields change.
    p = ROOT / rel
    if read(rel) != obj:
        p.write_bytes(config_bytes(rel, obj))

def freeze_manifest(rel):
    paths = read("reproducibility/environment/runtime_manifest_paths.json")[rel]
    write_new(rel, {p: sha(p) for p in paths})

def verify_manifest(rel):
    for path, expected in read(rel).items():
        if sha(path) != expected:
            raise RuntimeError("Prerequisite source manifest differs: " + path)

def require_no_results(rel, metadata_dirs=()):
    directory = ROOT / rel
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if path.is_file():
            relative = path.relative_to(directory)
            if relative.parts[0] not in metadata_dirs:
                raise RuntimeError("Refusing to change run metadata after output exists: "
                                   + str(path.relative_to(ROOT)))

def config_bytes(rel, obj):
    # Preserve byte identity on idempotent calls (including original formatting).
    if read(rel) == obj:
        return (ROOT / rel).read_bytes()
    if rel == "configs/nvidia.json":
        # Keep the large row-index payload compact when refreshing provenance.
        return (json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode()

def validate_new_manifest(rel, obj):
    if (ROOT / rel).exists() and read(rel) != obj:
        raise RuntimeError("Existing runtime manifest differs; use a fresh reproduction copy: " + rel)

def layout():
    folders = ["logs", ".setup", ".cache/tmp", ".cache/huggingface", ".cache/matplotlib",
               "datasets", "embedders", "artifacts", "results/baselines/_logs",
               "results/baselines/_run", "results/nvidia/_logs", "results/nvidia/finished",
               "results/layer_sweep/logs", "results/nvidia_layer_validation/utility_v1/logs",
               "results/nvidia_layer_validation/utility_v1/finished"]
    for rel in folders:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    # Materialize the four bundled illustrative prompts only. Existing runtime
    # data must agree; layout initialization never replaces a changed dataset.
    example = (ROOT / "reproducibility/examples/proof_twins.jsonl").read_bytes()
    prompts = ROOT / "datasets/proof_twins_v1/prepared/queries.jsonl"
    prompts.parent.mkdir(parents=True, exist_ok=True)
    if prompts.is_symlink() or prompts.exists() and prompts.read_bytes() != example:
        raise RuntimeError("Illustrative prompt fixture differs; use a fresh reproduction copy")
    if not prompts.exists():
        prompts.write_bytes(example)
    runtime = ROOT / "artifacts/format_factorial_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ["datasets", "embedders", "artifacts"]:
        link = runtime / name
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or link.resolve() != (ROOT / name).resolve():
                raise RuntimeError("Unsafe or unexpected runtime path: " + str(link))
        else:
            link.symlink_to(os.path.relpath(ROOT / name, runtime), target_is_directory=True)
    # Third-party reference files retain their published byte hashes.
    vendor = "src/nvidia_utility/vendor/"
    for name, record in read(vendor + "sources.json")["files"].items():
        if sha(vendor + name) != record["sha256"]:
            raise RuntimeError("Vendor source checksum mismatch: " + name)

def reduced():
    rel = "configs/mlp_reduced.json"
    verify_manifest("results/baselines/_run/source_hashes.json")
    config = read(rel)
    config["source_hashes"] = {p: sha(p) for p in config["source_hashes"]}
    config["original_source_hashes_sha256"] = sha("results/baselines/_run/source_hashes.json")
    if read(rel) != config:
        require_no_results("results/baselines", ("_run", "_logs"))
    replace_config(rel, config)

def nvidia():
    verify_manifest("results/baselines/_run/source_hashes.json")
    reduced_config = read("configs/mlp_reduced.json")
    for path, expected in reduced_config["source_hashes"].items():
        if sha(path) != expected:
            raise RuntimeError("Run the reduced provenance stage before NVIDIA: " + path)
    if reduced_config["original_source_hashes_sha256"] != sha("results/baselines/_run/source_hashes.json"):
        raise RuntimeError("Reduced provenance does not match baseline preparation")
    config = read("configs/nvidia.json")
    config["reduced_mlp_config_sha256"] = sha("configs/mlp_reduced.json")
    if read("configs/nvidia.json") != config:
        require_no_results("results/nvidia", ("provenance", "_logs", "timing"))
    replace_config("configs/nvidia.json", config)
    # src/nvidia/prepare.py writes its source manifest and performs preflight checks.

def utility():
    verify_manifest("results/nvidia/provenance/source_hashes.json")
    config = read("configs/nvidia_utility_v1.json")
    for rel, expected in config["input_hashes"].items():
        actual = sha(rel)
        if rel == "configs/nvidia.json":
            config["input_hashes"][rel] = actual
        elif actual != expected:
            raise RuntimeError("Frozen input mismatch (not rehashed): " + rel)
    if read("configs/nvidia_utility_v1.json") != config:
        require_no_results("results/nvidia_layer_validation/utility_v1", ("provenance", "logs"))
    # Validate every dependency and any existing manifest before changing config.
    # This avoids a half-updated config when a source file is missing/stale.
    rel = "configs/nvidia_utility_v1.json"
    manifest = "results/nvidia_layer_validation/utility_v1/provenance/source_hashes.json"
    future_hash = hashlib.sha256(config_bytes(rel, config)).hexdigest()
    paths = read("reproducibility/environment/runtime_manifest_paths.json")[manifest]
    hashes = {path: (future_hash if path == rel else sha(path)) for path in paths}
    validate_new_manifest(manifest, hashes)
    replace_config(rel, config)
    write_new(manifest, hashes)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["layout", "reduced", "nvidia", "utility",
        "layer-sweep", "format", "proof-twins", "family-utility", "inference"], default="layout")
    args = parser.parse_args()
    layout()
    stages = {"reduced": reduced, "nvidia": nvidia, "utility": utility}
    targets = {
        "layer-sweep": ["results/layer_sweep/provenance/source_hashes.json"],
        "format": ["results/geometry/_diagnostics/format_factorial_v1/source_hashes.json"],
        "proof-twins": ["results/geometry/_diagnostics/proof_twins_v1/source_hashes.json"],
        "family-utility": ["results/geometry/gte-qwen2-7b/_diagnostics/family_utility_transfer/source_hashes.json"],
        "inference": ["results/nvidia/provenance/input_hashes.json",
                      "results/nvidia/inference_cpu_v1/source_hashes.json",
                      "results/nvidia/inference_cpu_v1/audit_source_hashes.json"]}
    if args.stage in stages:
        stages[args.stage]()
    for target in targets.get(args.stage, []):
        freeze_manifest(target)
    print("Release runtime initialized:", args.stage)

if __name__ == "__main__":
    main()
