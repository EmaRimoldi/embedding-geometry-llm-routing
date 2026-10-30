"""Download pinned public inputs and verify their remote checksums. Resumable."""
import datetime, json, os, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for folder in [".setup", ".cache/tmp", ".cache/huggingface"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
# Downloads require network; compute launchers default to offline mode.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
for key, value in {"HF_HOME": ".cache/huggingface", "HF_HUB_CACHE": ".cache/huggingface/hub", "HF_XET_CACHE": ".cache/huggingface/xet", "XDG_CACHE_HOME": ".cache", "TMPDIR": ".cache/tmp"}.items():
    os.environ[key] = str(ROOT / value)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
hf = str(ROOT / ".venv-download/bin/hf")
assets = json.loads((ROOT / "reproducibility/manifests/assets.json").read_text())
status_path = ROOT / ".setup/download-status.json"
status = {}
def save():
    status_path.write_text(json.dumps(status, indent=2) + "\n")
for asset in assets:
    name = asset["name"]
    status[name] = {"state": "downloading", "revision": asset["revision"]}
    save()
    print(f"START {name} {datetime.datetime.now(datetime.timezone.utc).isoformat()}", flush=True)
    base = [asset["repo"], "--repo-type", asset["type"], "--revision", asset["revision"], "--local-dir", str(ROOT / asset["destination"])]
    cmd = [hf, "download"] + base + ["--max-workers", "3"]
    for pattern in asset["include"]:
        cmd.extend(["--include", pattern])
    try:
        subprocess.run(cmd, check=True)
        status[name]["state"] = "verifying"
        save()
        with (ROOT / ".setup" / (name + "-verification.json")).open("w") as out:
            subprocess.run([hf, "cache", "verify"] + base + ["--format", "json"], stdout=out, check=True)
        status[name]["state"] = "verified"
        print(f"VERIFIED {name}", flush=True)
    except Exception as exc:
        status[name].update(state="failed", error=str(exc))
        save()
        raise
    save()
