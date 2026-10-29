#!/usr/bin/env python3
"""Create a deterministic allowlisted code ZIP; never touches the research workspace."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from verify_release import ROOT, MANIFEST, public_files, inspect, load_private_patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-patterns", type=Path,
                        help="Optional JSON regex array outside the public file set; never exported.")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Refresh and verify the current manifest without creating or replacing a ZIP.")
    args = parser.parse_args()
    private_patterns = load_private_patterns(args.private_patterns)
    files = {}
    for p in public_files():
        name = str(p.relative_to(ROOT))
        if name == MANIFEST:
            continue
        if p.is_symlink():
            raise RuntimeError("Refusing to archive a symlink: " + name)
        b = p.read_bytes()
        files[name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    target = ROOT / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"version": 1, "files": files}, indent=2, sort_keys=True) + "\n")
    check = inspect(private_patterns=private_patterns)
    if not check["ok"]:
        print(json.dumps(check, indent=2))
        return 1
    if args.manifest_only:
        print(json.dumps({"manifest": MANIFEST, "files": len(files),
                          "validation": True, "archive_updated": False}, indent=2))
        return 0
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / "routing_reproduction.zip"
    # Replace only this generated archive in this submission directory.
    temporary = dist / "routing_reproduction.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.comment = b""
        for p in public_files():
            if p.is_symlink():
                raise RuntimeError("Refusing to archive a symlink: " + str(p.relative_to(ROOT)))
            name = "routing-reproduction/" + str(p.relative_to(ROOT))
            # Neutral fixed metadata: no experiment or build date is encoded.
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info.comment = b""
            info.extra = b""
            z.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / "routing_reproduction.zip.sha256").write_text(digest + "  routing_reproduction.zip\n")
    print(json.dumps({"archive": archive.name, "sha256": digest, "bytes": archive.stat().st_size,
                      "files": len(files) + 1, "validation": check["ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
