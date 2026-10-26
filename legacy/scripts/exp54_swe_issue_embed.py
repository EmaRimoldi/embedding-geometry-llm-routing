"""Embed SWE-bench Verified issue statements in frozen LLMRouterBench row order.

Independent sensitivity for the human difficulty labels; uses the project's
unchanged GTE embedding routine and the SHA-pinned public annotation table.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rb.data import RUNTIME_ROOT, load_panel  # noqa: E402
from embed_gpu import run_st  # noqa: E402

SOURCE = ROOT / "results/difficulty_neighborhoods/source_labels/swe_verified_c104.parquet"
SHA = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    t = pd.read_parquet(SOURCE, columns=["instance_id", "problem_statement"])
    by_id = t.set_index("instance_id")
    p = load_panel("llmrouterbench-performance-cost")
    ix = [i for i, f in enumerate(p.dataset_ids) if f == "swe-bench"]
    src = glob.glob(str(RUNTIME_ROOT / "artifacts/llmrouterbench/bench-release/swe-bench/verified/*/*.json"))[0]
    rec = {str(z["index"]): z for z in json.load(open(src))["records"]}
    assert len(ix) == 500 and all(p.origin_queries[i] == rec[str(p.record_indices[i])]["origin_query"] for i in ix)
    ids = [rec[str(p.record_indices[i])]["instance_id"] for i in ix]
    assert set(ids) == set(by_id.index)
    texts = [str(by_id.loc[z, "problem_statement"]) for z in ids]
    out = ROOT / "llmrouterbench/results/performance-cost/difficulty_neighborhoods/exp54_swe_issues/gte-qwen2-7b.npy"
    if out.exists():
        raise FileExistsError(out)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    run_st("gte-qwen2-7b", texts, out,
           "500 SWE-bench Verified problem_statement texts in frozen LLMRouterBench row order; external SHA and exact join in exp54")


if __name__ == "__main__":
    main()
