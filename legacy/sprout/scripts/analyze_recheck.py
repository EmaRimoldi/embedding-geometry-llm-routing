"""Analyze blind pass D and expand manual review without automatic relabeling."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ROOT / "topics"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    accepted = {int(r["item_id"]): r for r in read_jsonl(TOPICS / "qwen32_topics.jsonl")
                if r["decision"] == "two_pass_high_confidence_agreement"}
    recheck = {int(r["item_id"]): r for r in read_jsonl(TOPICS / "accepted_recheck_pass_D.jsonl")}
    original_review = read_jsonl(TOPICS / "manual_review.jsonl")
    flagged = []
    for item_id, d in recheck.items():
        parsed = d.get("parsed")
        if not parsed or parsed["label"] != d["original_label"] or parsed["confidence"] != "high":
            base = accepted[item_id]
            flagged.append({
                **base,
                "decision": "manual_review_after_blind_recheck",
                "review_reasons": (["pass_D_invalid"] if not parsed else [])
                    + (["pass_D_disagreement"] if parsed and parsed["label"] != d["original_label"] else [])
                    + (["pass_D_non_high_confidence"] if parsed and parsed["confidence"] != "high" else []),
                "pass_D": d,
            })
    with (TOPICS / "accepted_recheck_flagged.jsonl").open("w") as handle:
        for row in flagged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    expanded = original_review + flagged
    with (TOPICS / "manual_review_expanded.jsonl").open("w") as handle:
        for row in expanded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    calibration = {}
    reviewed_path = TOPICS / "accepted_audit_sample_100_reviewed.csv"
    if reviewed_path.exists():
        human = {int(r["item_id"]): r for r in csv.DictReader(reviewed_path.open())}
        available = set(human) & set(recheck)
        known_wrong = {i for i in available if human[i]["manual_correct"] == "no"}
        known_right = available - known_wrong
        changed = {i for i in available if recheck[i].get("parsed") and
                   recheck[i]["parsed"]["label"] != recheck[i]["original_label"]}
        caught = {i for i in known_wrong if recheck[i].get("parsed") and
                  recheck[i]["parsed"]["label"] == human[i]["manual_label"]}
        calibration = {
            "audited_available": len(available), "known_wrong": len(known_wrong),
            "known_wrong_caught_with_correct_label": len(caught),
            "known_wrong_missed": len(known_wrong - caught),
            "known_correct_flagged_by_label_change": len(known_right & changed),
            "known_correct_unchanged": len(known_right - changed),
        }
    valid = [r for r in recheck.values() if r.get("parsed")]
    summary = {
        "accepted_total": len(accepted), "recheck_completed": len(recheck),
        "valid_rechecks": len(valid), "flagged_for_manual_review": len(flagged),
        "expanded_manual_review": len(expanded),
        "flag_reasons": Counter(reason for row in flagged for reason in row["review_reasons"]),
        "calibration": calibration,
        "policy": "Pass D only flags cases; it never changes labels automatically.",
    }
    (TOPICS / "accepted_recheck_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
