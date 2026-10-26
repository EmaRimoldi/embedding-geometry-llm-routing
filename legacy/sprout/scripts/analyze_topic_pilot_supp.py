"""Quantify and expose the two-pass Qwen topic pilot for audit."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "topics" / "pilot_qwen32_supplement.jsonl"
OUT = ROOT / "topics" / "pilot_audit_supplement"


def main() -> None:
    rows = [json.loads(x) for x in IN.open() if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["pilot_component"]].append(row)

    report = {}
    for name, subset in groups.items():
        valid = [r for r in subset if all(r["passes"].get(p, {}).get("parsed") for p in ("A", "B"))]
        agree = [r for r in valid if r["passes"]["A"]["parsed"]["label"] == r["passes"]["B"]["parsed"]["label"]]
        high_high = [r for r in valid if all(r["passes"][p]["parsed"]["confidence"] == "high" for p in ("A", "B"))]
        high_high_agree = [r for r in high_high if r["passes"]["A"]["parsed"]["label"] == r["passes"]["B"]["parsed"]["label"]]
        report[name] = {
            "n": len(subset), "valid_both": len(valid),
            "agreement": len(agree) / len(valid) if valid else None,
            "high_high": len(high_high),
            "high_high_agreement": len(high_high_agree) / len(high_high) if high_high else None,
            "pass_A_labels": Counter(r["passes"]["A"]["parsed"]["label"] for r in valid),
            "pass_B_labels": Counter(r["passes"]["B"]["parsed"]["label"] for r in valid),
            "confidence_pairs": Counter(
                r["passes"]["A"]["parsed"]["confidence"] + "/" + r["passes"]["B"]["parsed"]["confidence"]
                for r in valid
            ),
        }

    flagged = []
    for row in rows:
        a = row["passes"].get("A", {}).get("parsed")
        b = row["passes"].get("B", {}).get("parsed")
        reason = []
        if not a or not b:
            reason.append("invalid_or_excluded")
        else:
            if a["label"] != b["label"]:
                reason.append("label_disagreement")
            if a["confidence"] != "high" or b["confidence"] != "high":
                reason.append("non_high_confidence")
        if reason:
            flagged.append({"reason": reason, **row})
    with (OUT / "flagged.jsonl").open("w") as handle:
        for row in flagged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Deterministic examples for human inspection: up to 10 agreed examples per class.
    samples = defaultdict(list)
    for row in rows:
        a = row["passes"].get("A", {}).get("parsed")
        b = row["passes"].get("B", {}).get("parsed")
        if a and b and a["label"] == b["label"] and len(samples[a["label"]]) < 10:
            samples[a["label"]].append({
                "pilot_component": row["pilot_component"], "prompt": row["prompt"],
                "label": a["label"], "evidence_A": a["evidence"], "evidence_B": b["evidence"],
            })
    (OUT / "class_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    report["flagged"] = len(flagged)
    report["flagged_path"] = str(OUT / "flagged.jsonl")
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=lambda x: dict(x)))
    print(json.dumps(report, indent=2, default=lambda x: dict(x)))


if __name__ == "__main__":
    main()
