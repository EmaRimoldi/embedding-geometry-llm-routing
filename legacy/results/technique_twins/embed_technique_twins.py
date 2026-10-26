"""Embed the 32 technique x story problems (raw text) with GTE; writes technique_twins/embeddings/gte-qwen2-7b__raw.npy."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from embed_gpu import run_st  # noqa: E402
from rb.panelctx import REP_NAME  # noqa: E402
HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "collection.json").read_text())["items"]
(HERE / "embeddings").mkdir(exist_ok=True)
run_st(REP_NAME, [it["text"] for it in items], HERE / "embeddings" / f"{REP_NAME}__raw.npy", "technique_twins/raw; collection.json order")
print("done", flush=True)
