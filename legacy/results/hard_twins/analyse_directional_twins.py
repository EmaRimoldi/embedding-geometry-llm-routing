"""Read-only directional comparison for the existing 21 authored hard/easy pairs.

For each arm, compare its own opposite-difficulty twin with *other* arms of
its own difficulty in the same mathematical field. No pair or comparator is
selected using the measured cosine. Writes a new analysis file only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "collection.json").read_text())["items"]
pairs = list(dict.fromkeys(x["pair"] for x in items))
assert len(pairs) == 21 and len(items) == 42
by_pair = {p: {x["arm"]: i for i, x in enumerate(items) if x["pair"] == p} for p in pairs}
assert all(set(z) == {"hard", "easy"} for z in by_pair.values())
field = {p: items[by_pair[p]["hard"]]["family"] for p in pairs}
assert all(items[by_pair[p]["easy"]]["family"] == field[p] for p in pairs)

result = {"definition": "For each pair, twin cosine versus every other easy arm viewed from its easy arm, and every other hard arm viewed from its hard arm; candidates restricted to the same field, no outcome-based selection.",
          "pairs": len(pairs), "representations": {}}
for model in ("gte-qwen2-7b", "minilm-l12"):
    for variant in ("raw", "gsm_template"):
        name = f"{model}__{variant}"
        x = np.load(HERE / "embeddings" / f"{name}.npy").astype(np.float64)
        assert x.shape[0] == len(items)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        s = x @ x.T
        per = []
        for p in pairs:
            h, e = by_pair[p]["hard"], by_pair[p]["easy"]
            others = [q for q in pairs if q != p and field[q] == field[p]]
            easy = sorted(((float(s[e, by_pair[q]["easy"]]), q) for q in others), reverse=True)
            hard = sorted(((float(s[h, by_pair[q]["hard"]]), q) for q in others), reverse=True)
            twin = float(s[h, e])
            per.append({"pair": p, "field": field[p], "twin": twin,
                        "easy_to_other_easy_mean": float(np.mean([z[0] for z in easy])),
                        "easy_to_other_easy_max": easy[0][0], "easy_closest_pair": easy[0][1],
                        "hard_to_other_hard_mean": float(np.mean([z[0] for z in hard])),
                        "hard_to_other_hard_max": hard[0][0], "hard_closest_pair": hard[0][1],
                        "easy_margin_vs_closest_easy": twin - easy[0][0],
                        "hard_margin_vs_closest_hard": twin - hard[0][0]})
        fields = {}
        for f in dict.fromkeys(field.values()):
            z = [a for a in per if a["field"] == f]
            fields[f] = {"n": len(z), "twin_mean": float(np.mean([a["twin"] for a in z])),
                         "other_easy_mean": float(np.mean([a["easy_to_other_easy_mean"] for a in z])),
                         "other_hard_mean": float(np.mean([a["hard_to_other_hard_mean"] for a in z])),
                         "closest_easy_mean": float(np.mean([a["easy_to_other_easy_max"] for a in z])),
                         "closest_hard_mean": float(np.mean([a["hard_to_other_hard_max"] for a in z])),
                         "easy_wins": sum(a["easy_margin_vs_closest_easy"] > 0 for a in z),
                         "hard_wins": sum(a["hard_margin_vs_closest_hard"] > 0 for a in z)}
        result["representations"][name] = {"per_pair": per, "fields": fields,
                                           "easy_wins": sum(a["easy_margin_vs_closest_easy"] > 0 for a in per),
                                           "hard_wins": sum(a["hard_margin_vs_closest_hard"] > 0 for a in per)}

out = HERE / "directional_similarity.json"
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(result, indent=2))
for name, z in result["representations"].items():
    print(name, "easy wins", z["easy_wins"], "hard wins", z["hard_wins"])
    for f, v in z["fields"].items():
        print(" ", f, "n", v["n"], "twin", round(v["twin_mean"], 3),
              "other easy", round(v["other_easy_mean"], 3),
              "other hard", round(v["other_hard_mean"], 3),
              "closest easy", round(v["closest_easy_mean"], 3),
              "closest hard", round(v["closest_hard_mean"], 3),
              "wins", v["easy_wins"], v["hard_wins"])
    print("top easy examples", [(a["pair"], a["easy_closest_pair"], round(a["twin"], 3),
                                round(a["easy_to_other_easy_max"], 3))
                               for a in sorted(z["per_pair"], key=lambda q: -q["easy_to_other_easy_max"])[:9]])
