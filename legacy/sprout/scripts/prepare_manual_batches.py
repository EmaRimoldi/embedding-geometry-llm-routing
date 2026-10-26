"""Arrange flagged topics into stable, family-balanced manual review batches."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "topics" / "manual_review.jsonl"
ORDERED = ROOT / "topics" / "manual_review_ordered.jsonl"
BATCH_DIR = ROOT / "topics" / "manual_review_batches"


def main() -> None:
    rows = [json.loads(x) for x in SOURCE.open() if x.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["families"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (len(row["prompt"]), int(row["item_id"])))
    ordered = []
    while any(groups.values()):
        for family in sorted(groups):
            if groups[family]:
                ordered.append(groups[family].pop(0))
    for index, row in enumerate(ordered, 1):
        row["manual_review_order"] = index
    with ORDERED.open("w") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for path in BATCH_DIR.glob("batch_*.jsonl"):
        path.unlink()
    for start in range(0, len(ordered), 50):
        path = BATCH_DIR / f"batch_{start // 50 + 1:03d}.jsonl"
        with path.open("w") as handle:
            for row in ordered[start:start + 50]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows), "batch_size": 50,
        "batches": (len(rows) + 49) // 50,
        "families": {";".join(k): v for k, v in Counter(tuple(r["families"]) for r in rows).items()},
        "reasons": Counter(reason for r in rows for reason in r["review_reasons"]),
    }
    (ROOT / "topics" / "manual_review_batches_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
