"""Materialize a hash-pinned numeric SPROUT panel in GTE row order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "data"
OUT = ROOT / "panel"
PRICES = json.loads((ROOT / "config" / "prices.json").read_text())
BASE = {"key", "dataset", "dataset_level", "dataset_idx", "prompt", "golden_answer"}


def ordered_files() -> list[Path]:
    rank = {"train": 0, "validation": 1, "test": 2}
    return sorted(RAW.glob("*.parquet"), key=lambda p: (rank[p.name.split("-")[0]], p.name))


def family(dataset: str) -> str:
    if dataset == "TIGER-Lab/MMLU-Pro": return "mmlupro"
    if dataset.startswith("lighteval/MATH"): return "math"
    if dataset.startswith("Idavidrein/gpqa"): return "gpqa"
    if dataset.startswith("TAUR-Lab/MuSR"): return "musr"
    if dataset.startswith("rungalileo/ragbench"): return "ragbench"
    if dataset == "openhermes/teknium": return "openhermes"
    raise ValueError(dataset)


def main() -> None:
    schema = pq.ParquetFile(ordered_files()[0]).schema_arrow
    models = [name for name in schema.names if name not in BASE]
    assert models == list(PRICES["models"]), (models, list(PRICES["models"]))
    rows = []
    reward_parts, in_parts, out_parts = [], [], []
    for path in ordered_files():
        split = path.name.split("-")[0]
        base = pq.read_table(path, columns=["key", "dataset", "dataset_level", "dataset_idx", "prompt"]).to_pylist()
        for row in base:
            rows.append({
                "key": str(row["key"]), "split": split, "dataset": str(row["dataset"]),
                "family": family(str(row["dataset"])), "dataset_level": str(row["dataset_level"]),
                "dataset_idx": int(row["dataset_idx"]), "prompt": str(row["prompt"]),
            })
        rewards, ins, outs = [], [], []
        for model in models:
            rewards.append(np.asarray(pq.read_table(path, columns=[f"{model}.score"])["score"].to_numpy(), dtype=np.float64))
            ins.append(np.asarray(pq.read_table(path, columns=[f"{model}.num_input_tokens"])["num_input_tokens"].to_numpy(), dtype=np.int32))
            outs.append(np.asarray(pq.read_table(path, columns=[f"{model}.num_output_tokens"])["num_output_tokens"].to_numpy(), dtype=np.int32))
        reward_parts.append(np.column_stack(rewards)); in_parts.append(np.column_stack(ins)); out_parts.append(np.column_stack(outs))
    rewards = np.vstack(reward_parts); input_tokens = np.vstack(in_parts); output_tokens = np.vstack(out_parts)
    pin = np.asarray([PRICES["models"][m]["input"] for m in models], dtype=np.float64)
    pout = np.asarray([PRICES["models"][m]["output"] for m in models], dtype=np.float64)
    costs = (input_tokens * pin[None, :] + output_tokens * pout[None, :]) / 1_000_000.0
    assert len(rows) == 44241 and rewards.shape == costs.shape == (44241, 13)
    assert np.isfinite(rewards).all() and np.isfinite(costs).all()
    assert len({r["key"] for r in rows}) == len(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "rows.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(
        OUT / "sprout.npz", rewards=rewards, costs=costs,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
    meta = {
        "rows": len(rows), "models": models,
        "families": sorted({r["family"] for r in rows}),
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
        "reward_sha256": hashlib.sha256(np.ascontiguousarray(rewards).tobytes()).hexdigest(),
        "cost_sha256": hashlib.sha256(np.ascontiguousarray(costs).tobytes()).hexdigest(),
        "price_source": PRICES["source"], "price_retrieved": PRICES["retrieved"],
        "cost_formula": "(input_tokens*input_USD_per_1M + output_tokens*output_USD_per_1M)/1e6",
    }
    (OUT / "panel.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
