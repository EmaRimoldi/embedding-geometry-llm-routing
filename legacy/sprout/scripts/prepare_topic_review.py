"""Prepare human-review artifacts for Qwen-generated SPROUT subjects."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ROOT / "topics" / "qwen32_topics.jsonl"
REVIEW_CSV = ROOT / "topics" / "manual_review.csv"
AUDIT_JSONL = ROOT / "topics" / "accepted_audit_sample_100.jsonl"
AUDIT_CSV = ROOT / "topics" / "accepted_audit_sample_100.csv"
SEED = 20260919


def read_rows() -> list[dict]:
    with TOPICS.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flat(row: dict) -> dict:
    a = row["passes"].get("A", {}).get("parsed") or {}
    b = row["passes"].get("B", {}).get("parsed") or {}
    return {
        "item_id": row["item_id"],
        "families": ";".join(row["families"]),
        "review_reasons": ";".join(row.get("review_reasons", [])),
        "label_A": a.get("label", ""), "confidence_A": a.get("confidence", ""),
        "evidence_A": a.get("evidence", ""),
        "label_B": b.get("label", ""), "confidence_B": b.get("confidence", ""),
        "evidence_B": b.get("evidence", ""),
        "accepted_label": row.get("label") or "",
        "prompt": row["prompt"],
        "manual_correct": "", "manual_label": "", "manual_notes": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(flat(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat(row) for row in rows)


def main() -> None:
    rows = read_rows()
    review = [row for row in rows if row["decision"] == "manual_review"]
    accepted = [row for row in rows if row["decision"] == "two_pass_high_confidence_agreement"]
    write_csv(REVIEW_CSV, review)

    rng = random.Random(SEED)
    groups = defaultdict(list)
    for row in accepted:
        groups[(tuple(row["families"]), row["label"])].append(row)
    for group in groups.values():
        rng.shuffle(group)
    sample = []
    # Cover every nonempty family-subject group once before filling randomly.
    for key in sorted(groups):
        if len(sample) == 100:
            break
        sample.append(groups[key].pop())
    remaining = [row for group in groups.values() for row in group]
    rng.shuffle(remaining)
    sample.extend(remaining[: 100 - len(sample)])
    rng.shuffle(sample)
    with AUDIT_JSONL.open("w") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(AUDIT_CSV, sample)
    summary = {
        "seed": SEED, "manual_review": len(review),
        "accepted": len(accepted), "audit_sample": len(sample),
        "manual_review_csv": str(REVIEW_CSV), "audit_csv": str(AUDIT_CSV),
    }
    (ROOT / "topics" / "review_preparation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
