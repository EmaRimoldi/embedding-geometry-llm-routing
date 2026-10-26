"""Merge authoritative and validated Qwen subjects into the SPROUT row manifest."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "topics" / "authoritative_and_pending.jsonl"
QWEN = ROOT / "topics" / "qwen32_topics.jsonl"
OUT = ROOT / "topics" / "final_topics.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    base = read_jsonl(BASE)
    qwen_rows = read_jsonl(QWEN)
    qwen = {row["prompt"]: row for row in qwen_rows}
    if len(qwen) != len(qwen_rows):
        raise RuntimeError("Qwen output contains duplicate prompt records")
    final = []
    missing = []
    unresolved = []
    for row in base:
        row = dict(row)
        if row["topic"] is None:
            decision = qwen.get(row["prompt"])
            if decision is None:
                missing.append(row["key"])
                continue
            if decision["label"] is None:
                unresolved.append(row["key"])
                continue
            row["topic"] = decision["label"]
            row["topic_source"] = "Qwen2.5-32B-Instruct validated subject taxonomy"
            row["verification"] = decision["decision"]
        final.append(row)
    if missing or unresolved:
        raise RuntimeError(f"Cannot finalize: missing={len(missing)}, unresolved={len(unresolved)}")
    keys = [row["key"] for row in final]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate SPROUT keys in final topic manifest")
    prompt_labels = defaultdict(set)
    for row in final:
        prompt_labels[row["prompt"]].add((row["family"], row["topic"]))
    inconsistent = {p: x for p, x in prompt_labels.items() if len(x) > 1}
    # A prompt can occur in different released families, but it must still have one
    # semantic topic label when Qwen labeled it. Report rather than silently drop it.
    with OUT.open("w") as handle:
        for row in final:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cells = Counter((row["family"], row["topic"]) for row in final)
    summary = {
        "rows": len(final), "unique_keys": len(set(keys)),
        "families": Counter(row["family"] for row in final),
        "topics_per_family": {
            fam: len({row["topic"] for row in final if row["family"] == fam})
            for fam in sorted({row["family"] for row in final})
        },
        "cells": {f"{fam}::{topic}": n for (fam, topic), n in sorted(cells.items())},
        "verification": Counter(row["verification"] for row in final),
        "cross_family_prompt_label_pairs": len(inconsistent),
        "output": str(OUT),
    }
    (ROOT / "topics" / "final_topics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
