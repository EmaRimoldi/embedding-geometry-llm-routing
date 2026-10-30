#!/usr/bin/env python3
"""Verify exported files, evidence hashes and generic disclosure indicators."""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TOP = {"README.md", "LICENSE", "Makefile", "requirements.lock",
              ".gitignore", ".github", "docs", "src", "configs", "tests", "scripts",
              "jobs", "reproducibility"}
# Generic user-home or machine-storage paths; no author-specific identifiers.
PATH_PATTERNS = [
    rb"/(?:Users|home|u)/[^/\s\"'<>]+",
    rb"/(?:scratch|gpfs|lustre|cluster|nfs)/[^\s\"'<>]+",
    rb"[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\\\s\"'<>]+",
]
SECRET_PATTERNS = [
    rb"AKIA[0-9A-Z]{16}",
    rb"gh[pousr]_[A-Za-z0-9]{20,}",
    rb"github_pat_[A-Za-z0-9_]{30,}",
    rb"hf_[A-Za-z0-9]{20,}",
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
]
MANIFEST = "reproducibility/provenance/release_manifest.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def public_files(root=ROOT):
    skip = {"__pycache__", ".pytest_cache", ".cache", ".venv", ".venv-download"}
    for base, dirs, names in os.walk(str(root), followlinks=False):
        here = Path(base).relative_to(root)
        if str(here) == ".":
            dirs[:] = sorted(d for d in dirs if d in PUBLIC_TOP)
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for name in sorted(names):
            if name == ".DS_Store":
                continue
            p = Path(base) / name
            rel = p.relative_to(root)
            parts = rel.parts
            if parts[0] in PUBLIC_TOP:
                yield p


def load_private_patterns(path, root=ROOT):
    """Read a JSON array of byte-compatible regex strings without exporting it."""
    if path is None:
        return []
    path = Path(path).resolve()
    if path in {p.resolve() for p in public_files(root)}:
        raise ValueError("Private scan rules must be outside the public file set.")
    rules = json.loads(path.read_text())
    if not isinstance(rules, list) or not all(isinstance(rule, str) and rule for rule in rules):
        raise ValueError("Private scan rules must be a JSON array of nonempty regex strings.")
    patterns = [rule.encode("utf-8") for rule in rules]
    for pattern in patterns:
        re.compile(pattern, re.I)
    return patterns


def scan_content(name, content, private_patterns=()):
    """Scan names and bytes uniformly, including this source file."""
    payload = name.encode("utf-8") + b"\n" + content
    problems = []
    for patterns, kind, flags in [
        (PATH_PATTERNS, "machine-specific path pattern", re.I),
        (SECRET_PATTERNS, "credential pattern", 0),
        (private_patterns, "private disclosure pattern", re.I),
    ]:
        if any(re.search(pattern, payload, flags) for pattern in patterns):
            problems.append({"file": name, "problem": kind})
    return problems


def inspect(root=ROOT, private_patterns=()):
    root = Path(root)
    problems = []
    if not (root / MANIFEST).exists():
        return {"ok": False, "errors": ["Missing release manifest; build the package first."]}
    manifest = json.loads((root / MANIFEST).read_text())
    expected = manifest["files"]
    actual = {str(p.relative_to(root)): p for p in public_files(root)}
    for name, rec in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            problems.append({"file": name, "problem": "unsafe manifest path"})
            continue
        p = root / name
        if not p.is_file() or p.is_symlink():
            problems.append({"file": name, "problem": "missing or symlink"})
        elif sha(p) != rec["sha256"] or p.stat().st_size != rec["bytes"]:
            problems.append({"file": name, "problem": "checksum or size mismatch"})
    unexpected = set(actual) - set(expected) - {MANIFEST}
    for name in sorted(unexpected):
        problems.append({"file": name, "problem": "unlisted public file"})
    for name, p in actual.items():
        if p.is_symlink():
            problems.append({"file": name, "problem": "public symlink"})
            continue
        problems.extend(scan_content(name, p.read_bytes(), private_patterns))
    return {"ok": not problems, "files_checked": len(expected),
            "bytes_checked": sum(x["bytes"] for x in expected.values()),
            "private_patterns_used": bool(private_patterns),
            "errors": problems,
            "scope": "Exact exported bytes and configured static patterns; no file exemptions. "
                     "This is not proof of scientific correctness, exhaustive anonymity or secret detection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-patterns", type=Path,
                        help="Optional JSON regex array outside the public file set; never exported.")
    args = parser.parse_args()
    result = inspect(private_patterns=load_private_patterns(args.private_patterns))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
