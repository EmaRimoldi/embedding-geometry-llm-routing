"""Embed the fixed IMO 1988 P6 triplet using the existing unchanged GTE routine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rb.data import load_panel  # noqa: E402
from embed_gpu import run_st  # noqa: E402


def main():
    meta = json.loads((HERE / "triplet.json").read_text())
    items = json.loads((HERE.parent / "collection.json").read_text())["items"]
    hard = next(z["text"] for z in items if z["pair"] == meta["hard_pair"] and z["arm"] == "hard")
    easy = next(z["text"] for z in items if z["pair"] == meta["hard_pair"] and z["arm"] == "easy")
    texts = [hard, easy, meta["easy_control_text"]]
    p = load_panel("routerbench-0shot")
    t = str(p.prompts[np.flatnonzero(p.dataset_ids == "grade-school-math")[0]])
    prefix = t[:t.rfind("Answer: 9") + len("Answer: 9")] + "\n\n"
    for name, inp in (("raw", texts), ("gsm_template", [prefix + z for z in texts])):
        out = HERE / f"gte-qwen2-7b__{name}.npy"
        if out.exists():
            raise FileExistsError(out)
        run_st("gte-qwen2-7b", inp, out, f"fixed IMO 1988 P6 triplet in order hard, easy twin, independent easy control; {name}")
        x = np.load(out).astype(np.float64)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        sims = {"hard_easy_twin": float(x[0] @ x[1]),
                "easy_twin_easy_control": float(x[1] @ x[2]),
                "hard_easy_control": float(x[0] @ x[2])}
        (HERE / f"{name}_cosines.json").write_text(json.dumps(sims, indent=2))
        print(name, sims, flush=True)


if __name__ == "__main__":
    main()
