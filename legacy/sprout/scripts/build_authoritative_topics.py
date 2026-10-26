"""Build SPROUT family labels and all authoritative topic labels.

Rows without a defensible released subject label are sent to Qwen labeling.
This script reads no model outcomes, rewards, responses, costs, or gold answers.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "data"
SOURCE = ROOT / "data" / "source"
OUT = ROOT / "topics"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def family(dataset: str) -> str:
    if dataset == "TIGER-Lab/MMLU-Pro":
        return "mmlupro"
    if dataset.startswith("lighteval/MATH"):
        return "math"
    if dataset.startswith("Idavidrein/gpqa"):
        return "gpqa"
    if dataset.startswith("TAUR-Lab/MuSR"):
        return "musr"
    if dataset.startswith("rungalileo/ragbench"):
        return "ragbench"
    if dataset == "openhermes/teknium":
        return "openhermes"
    raise ValueError(dataset)


def main() -> None:
    mmlu = pq.read_table(SOURCE / "mmlu-pro" / "data" / "test-00000-of-00001.parquet").to_pylist()
    math_topics: dict[str, set[str]] = defaultdict(set)
    for path in sorted((SOURCE / "math").glob("*/*.parquet")):
        for row in pq.read_table(path, columns=["problem", "type"]).to_pylist():
            math_topics[norm(row["problem"])].add(str(row["type"]))

    rows = []
    for path in sorted(RAW.glob("*.parquet")):
        split = path.name.split("-")[0]
        for row in pq.read_table(path, columns=["key", "dataset", "dataset_idx", "prompt"]).to_pylist():
            dataset = str(row["dataset"])
            fam = family(dataset)
            topic = None
            source = None
            verification = None
            if fam == "mmlupro":
                src = mmlu[int(row["dataset_idx"])]
                topic = str(src["category"])
                source = "TIGER-Lab/MMLU-Pro category via dataset_idx"
                verification = "question_exact_substring" if str(src["question"]).strip() in str(row["prompt"]) else "dataset_idx_only"
                if topic.strip().lower() == "other":
                    topic = None
                    source = "released MMLU-Pro category is 'other', not a subject"
                    verification = "qwen_required"
            elif fam == "math":
                labels = sorted(math_topics.get(norm(row["prompt"]), set()))
                if len(labels) == 1:
                    topic = labels[0].lower().replace(" & ", "_").replace(" ", "_")
                    source = "EleutherAI/hendrycks_math type via exact normalized problem text"
                    verification = "exact_normalized_text"
                elif len(labels) > 1:
                    source = "ambiguous duplicate in EleutherAI/hendrycks_math: " + ",".join(labels)
                    verification = "qwen_required"
                else:
                    source = "no source match"
                    verification = "qwen_required"
            elif fam == "musr":
                topic = dataset.rsplit("/", 1)[-1] if dataset != "TAUR-Lab/MuSR" else "murder_mysteries"
                source = "released MuSR task identifier"
                verification = "release_field"
            elif fam in {"gpqa", "openhermes", "ragbench"}:
                source = (
                    "released RAGBench identifiers are source datasets, not a consistent subject variable"
                    if fam == "ragbench" else "no accessible authoritative subject in SPROUT release"
                )
                verification = "qwen_required"
            rows.append({
                "key": row["key"],
                "split": split,
                "dataset": dataset,
                "dataset_idx": int(row["dataset_idx"]),
                "family": fam,
                "prompt": row["prompt"],
                "topic": topic,
                "topic_source": source,
                "verification": verification,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "authoritative_and_pending.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pending_by_prompt = {}
    for row in rows:
        if row["topic"] is None:
            pending_by_prompt.setdefault(row["prompt"], {"prompt": row["prompt"], "families": set(), "keys": []})
            pending_by_prompt[row["prompt"]]["families"].add(row["family"])
            pending_by_prompt[row["prompt"]]["keys"].append(row["key"])
    with (OUT / "needs_qwen.jsonl").open("w") as handle:
        for prompt, item in pending_by_prompt.items():
            handle.write(json.dumps({"prompt": prompt, "families": sorted(item["families"]), "keys": item["keys"]}, ensure_ascii=False) + "\n")
    summary = {
        "rows": len(rows),
        "authoritative": sum(r["topic"] is not None for r in rows),
        "qwen_rows": sum(r["topic"] is None for r in rows),
        "qwen_unique_prompts": len(pending_by_prompt),
        "dataset_idx_only_mmlu": sum(r["verification"] == "dataset_idx_only" for r in rows),
        "ambiguous_or_unmatched_math": sum(r["family"] == "math" and r["topic"] is None for r in rows),
    }
    (OUT / "authoritative_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
