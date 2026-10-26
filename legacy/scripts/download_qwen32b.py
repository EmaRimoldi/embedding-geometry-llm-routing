"""Download Qwen2.5-32B-Instruct from the official Qwen organisation on Hugging Face (PI-approved 2026-09-15), pinned to one commit,
weights/config/tokenizer files only (no README, no code files), into experiments4/models/; verify every safetensors shard's SHA-256 against
the repository's LFS manifest; write provenance.json. Nothing is executed from the download; the model is loaded later with trust_remote_code=False."""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
REPO = "Qwen/Qwen2.5-32B-Instruct"; COMMIT = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
DEST = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-32B-Instruct"
ALLOW = ["*.safetensors", "*.json", "*.txt", "LICENSE"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"; t0 = time.time()
    info = HfApi().model_info(REPO, revision=COMMIT, files_metadata=True); assert info.author == "Qwen", info.author
    expected = {s.rfilename: (s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    DEST.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=REPO, revision=COMMIT, allow_patterns=ALLOW, local_dir=str(DEST), max_workers=8)
    print(f"downloaded to {path} in {time.time() - t0:.0f}s", flush=True)
    report = {}; bad = []
    for name, (size, sha) in expected.items():
        p = DEST / name
        if not p.exists(): 
            if any(name.endswith(x) for x in (".safetensors", ".json", ".txt")) or name == "LICENSE": bad.append(f"missing {name}")
            continue
        ok_size = p.stat().st_size == size; ok_sha = (sha256(p) == sha) if sha else None
        report[name] = {"size_ok": ok_size, "sha256_ok": ok_sha}
        if not ok_size or ok_sha is False: bad.append(name)
    for py in DEST.rglob("*.py"): bad.append(f"unexpected code file {py}")
    (DEST / "provenance.json").write_text(json.dumps({"repo": REPO, "commit": COMMIT, "author": info.author, "license_tags": [t for t in info.tags if "license" in t], "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "allow_patterns": ALLOW, "verification": report, "problems": bad, "trust_remote_code": False}, indent=1))
    print("verification problems:", bad or "none", flush=True)
    if bad: sys.exit(1)


if __name__ == "__main__":
    main()
