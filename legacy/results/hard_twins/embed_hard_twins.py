"""Embed the hard/easy minimal-pair twins with the panel's encoder.
Raw statement, and wrapped in the panel's own GSM8K template (TASK line + fixed 5-shot block).
Writes hard_twins/embeddings/<rep>__{raw,gsm_template}.npy (+ .json). Mirrors embed_hard_examples.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from rb.data import load_panel  # noqa: E402
from embed_gpu import run_st  # noqa: E402
from rb.panelctx import REP_NAME  # noqa: E402

HERE = Path(__file__).resolve().parent


def gsm_prefix(panel) -> str:
    t = str(panel.prompts[np.flatnonzero(panel.dataset_ids == "grade-school-math")[0]])
    cut = t.rfind("Answer: 9") + len("Answer: 9")
    return t[:cut] + "\n\n"


def main() -> None:
    items = json.loads((HERE / "collection.json").read_text())["items"]
    panel = load_panel("routerbench-0shot"); prefix = gsm_prefix(panel)
    (HERE / "embeddings").mkdir(exist_ok=True)
    (HERE / "embeddings" / "gsm_prefix.txt").write_text(prefix)
    for variant, texts in (("raw", [it["text"] for it in items]),
                           ("gsm_template", [prefix + it["text"] for it in items])):
        run_st(REP_NAME, texts, HERE / "embeddings" / f"{REP_NAME}__{variant}.npy",
               f"hard_twins/{variant}; ids in collection.json order")
    print("done", flush=True)


if __name__ == "__main__":
    main()
