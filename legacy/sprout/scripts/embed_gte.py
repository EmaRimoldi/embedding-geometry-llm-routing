"""Embed every SPROUT query that fits GTE's native 8192-token context.

No query is truncated. Row keys and the exclusion mask are stored beside the
matrix. Existing experiments4 representations are read-only and untouched.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from importlib.machinery import ModuleSpec
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parents[1]
RAW = ROOT / "data" / "raw" / "data"
MODEL = RUNTIME / "artifacts" / "models" / "gte-Qwen2-7B-instruct"
OUT = ROOT / "features"
MAX_SEQ = 8192


def flash_stub() -> None:
    try:
        version("flash_attn")
    except PackageNotFoundError:
        if "flash_attn" not in sys.modules:
            module = ModuleType("flash_attn")
            module.__spec__ = ModuleSpec("flash_attn", loader=None)
            sys.modules["flash_attn"] = module


def ordered_files() -> list[Path]:
    files = list(RAW.glob("*.parquet"))
    rank = {"train": 0, "validation": 1, "test": 2}
    return sorted(files, key=lambda p: (rank[p.name.split("-")[0]], p.name))


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    keys, prompts, splits = [], [], []
    for path in ordered_files():
        split = path.name.split("-")[0]
        table = pq.read_table(path, columns=["key", "prompt"])
        rows = table.to_pylist()
        keys.extend(str(r["key"]) for r in rows)
        prompts.extend(str(r["prompt"]) for r in rows)
        splits.extend([split] * len(rows))
    assert len(keys) == 44241 and len(keys) == len(set(keys))

    flash_stub()
    model = SentenceTransformer(
        str(MODEL), trust_remote_code=True, device="cuda",
        model_kwargs={"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = MAX_SEQ
    lengths = np.asarray([
        len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in prompts
    ], dtype=np.int32)
    keep = lengths <= MAX_SEQ
    kept_prompts = [text for text, ok in zip(prompts, keep) if ok]
    t0 = time.time()
    matrix = model.encode(
        kept_prompts, batch_size=4, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)
    assert matrix.shape[0] == int(keep.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "gte-qwen2-7b.npy", matrix, allow_pickle=False)
    np.savez_compressed(
        OUT / "gte-qwen2-7b_rows.npz",
        key=np.asarray(keys, dtype="U64"), split=np.asarray(splits, dtype="U10"),
        token_length=lengths, keep=keep,
    )
    meta = {
        "model": str(MODEL), "max_seq_length": MAX_SEQ, "prompt": None,
        "normalized": False, "truncation": "none; rows above context excluded",
        "rows_total": len(keys), "rows_kept": int(keep.sum()),
        "rows_excluded": int((~keep).sum()), "token_median": float(np.median(lengths)),
        "token_max": int(lengths.max()), "matrix_shape": list(matrix.shape),
        "matrix_sha256": hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest(),
        "elapsed_s": time.time() - t0,
    }
    (OUT / "gte-qwen2-7b.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
