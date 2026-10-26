"""Embed the 24 questions in three formats built from the panel's own templates (raw / mmlu / gsm).
Writes format_invariance/embeddings/gte-qwen2-7b__{raw,mmlu,gsm}.npy and the templates used."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from rb.data import load_panel  # noqa: E402
from embed_gpu import run_st  # noqa: E402
from rb.panelctx import REP_NAME  # noqa: E402

HERE = Path(__file__).resolve().parent


def templates(panel):
    g = str(panel.prompts[np.flatnonzero(panel.dataset_ids == "grade-school-math")[0]]); gsm_prefix = g[:g.rfind("Answer: 9") + len("Answer: 9")] + "\n\n"
    m = str(panel.prompts[np.flatnonzero(panel.dataset_ids == "mmlu")[0]])
    mmlu_prefix = m[:m.index("\n\n") + 2]; mmlu_suffix = m[m.rindex("\n", 0, m.index("Print only")):]
    return gsm_prefix, mmlu_prefix, mmlu_suffix


def main() -> None:
    items = json.loads((HERE / "collection.json").read_text())["items"]; panel = load_panel("routerbench-0shot")
    gsm_prefix, mmlu_prefix, mmlu_suffix = templates(panel)
    (HERE / "embeddings").mkdir(exist_ok=True)
    (HERE / "embeddings" / "templates.json").write_text(json.dumps({"gsm_prefix": gsm_prefix, "mmlu_prefix": mmlu_prefix, "mmlu_suffix": mmlu_suffix}, indent=1))
    variants = {"raw": [it["text"] for it in items],
                "mmlu": [mmlu_prefix + it["text"] + "\n" + "\n".join(f"{L}) {o}" for L, o in zip("ABCD", it["options"])) + mmlu_suffix for it in items],
                "gsm": [gsm_prefix + it["text"] for it in items]}
    (HERE / "embeddings" / "texts.json").write_text(json.dumps(variants, indent=1))
    for v, texts in variants.items():
        run_st(REP_NAME, texts, HERE / "embeddings" / f"{REP_NAME}__{v}.npy", f"format_invariance/{v}; ids in collection.json order")
    print("done", flush=True)


if __name__ == "__main__":
    main()
