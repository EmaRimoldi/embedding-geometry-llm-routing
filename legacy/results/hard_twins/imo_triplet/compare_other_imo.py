"""Compare the fixed IMO 1988 P6 triplet with nine independent IMO probes.

The independent probe collection predates the triplet. Cross-batch GTE cosines
can drift by about 0.001–0.003, so these values are illustrative only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
probes = ROOT / "results/hard_examples"
items = json.loads((probes / "collection.json").read_text())["items"]
imo = [i for i, z in enumerate(items) if z["category"] == "imo" and z["id"] != "imo1988p6"]
assert len(imo) == 9
out = {}
for variant in ("raw", "gsm_template"):
    x = np.load(HERE / f"gte-qwen2-7b__{variant}.npy").astype(np.float64)
    y = np.load(probes / "embeddings" / f"gte-qwen2-7b__{variant}.npy").astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    sims = x[0] @ y[imo].T
    j = int(np.argmax(sims))
    out[variant] = {"hard_to_easy_twin": float(x[0] @ x[1]),
                    "hard_to_closest_other_imo": float(sims[j]),
                    "closest_other_imo": items[imo[j]]["id"],
                    "hard_to_all_other_imo": {items[i]["id"]: float(v) for i, v in zip(imo, sims)},
                    "other_imo_n": len(imo)}
if (HERE / "other_imo_cosines.json").exists():
    raise FileExistsError(HERE / "other_imo_cosines.json")
(HERE / "other_imo_cosines.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
