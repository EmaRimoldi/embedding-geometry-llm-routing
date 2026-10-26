"""Build the refined SPROUT topic table (v2) from the label passes in results/sprout_labels/ and the existing sources.
Rules: authoritative labels unchanged (MMLU-Pro category, MATH type, MuSR task); RAGBench topic = released sub-dataset;
GPQA / MMLU-Pro 'other' / RAGBench-free OpenHermes rows: accepted two-pass subject, or the majority of passes A, B, C for
the formerly pending prompts (family if no majority); OpenHermes rows in the five large subjects: sub-topic = majority of
the sub-topic passes (subject if no majority). Writes results/sprout_labels/topics_v2.jsonl (key, family, topic_v1,
topic_v2, source_v2) and a summary. Reads only; nothing under experiments4/sprout is written."""
from __future__ import annotations
import collections, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sprout_common as S

OUT = S.RESULTS / "sprout_labels"


def passes(task):
    d = OUT / task; res = {}
    for p in "ABC":
        f = d / f"pass_{p}.jsonl"
        if f.exists(): res[p] = {int(json.loads(l)["item_id"]): (json.loads(l).get("parsed") or {}).get("label") for l in open(f)}
    return res


def majority(labels):
    labels = [x for x in labels if x]; c = collections.Counter(labels)
    if not c: return None
    lab, k = c.most_common(1)[0]; return lab if k >= 2 else None


def main():
    rows = [json.loads(l) for l in open(S.ROOT / "panel" / "rows.jsonl")]
    auth = {}; 
    for l in open(S.ROOT / "topics" / "authoritative_and_pending.jsonl"): a = json.loads(l); auth[a["key"]] = a["topic"]
    qwen = {}; qid = {}
    for l in open(S.ROOT / "topics" / "qwen32_topics.jsonl"):
        q = json.loads(l); qwen[q["prompt"]] = q; qid[int(q["item_id"])] = q
    P3 = passes("pending3"); pend_final = {}
    for i, q in qid.items():
        if q.get("label") is None:
            a = (q["passes"].get("A", {}).get("parsed") or {}).get("label"); b = (q["passes"].get("B", {}).get("parsed") or {}).get("label"); c = P3.get("C", {}).get(i); pend_final[q["prompt"]] = majority([a, b, c])
    ST = passes("subtopic"); items = json.load(open(OUT / "subtopic" / "items.json")) if (OUT / "subtopic" / "items.json").exists() else []
    sub_final = {}
    for it in items:
        i = it["item_id"]; sub_final[it["prompt"]] = (majority([ST.get(p, {}).get(i) for p in "ABC"]), it["subject"])
    out = []; src = collections.Counter(); cells = collections.Counter()
    for r in rows:
        k, fam, p = r["key"], r["family"], r["prompt"]; t1 = auth.get(k)
        if t1 is None: t1 = (qwen.get(p) or {}).get("label") or fam
        if auth.get(k) is not None: t2, s = auth[k], "authoritative"
        elif fam == "ragbench": t2, s = r["dataset"].rsplit("/", 1)[-1], "released sub-dataset"
        elif p in sub_final and sub_final[p][0]: t2, s = sub_final[p][1] + "/" + sub_final[p][0], "qwen subject + sub-topic majority"
        elif p in sub_final: t2, s = sub_final[p][1], "qwen subject (no sub-topic majority)"
        elif (qwen.get(p) or {}).get("label"): t2, s = qwen[p]["label"], "qwen subject (two-pass)"
        elif p in pend_final and pend_final[p]: t2, s = pend_final[p], "qwen subject (three-pass majority)"
        else: t2, s = fam, "family (no majority)"
        out.append({"key": k, "family": fam, "topic_v1": t1, "topic_v2": t2, "source_v2": s}); src[s] += 1; cells[(fam, t2)] += 1
    with open(OUT / "topics_v2.jsonl", "w") as h:
        for o in out: h.write(json.dumps(o, ensure_ascii=False) + "\n")
    summ = {"rows": len(out), "sources": dict(src), "cells": len(cells), "cells_ge50": sum(v >= 50 for v in cells.values()), "largest_cells": [[f"{a}/{b}", v] for (a, b), v in cells.most_common(15)],
            "pending3": {"items": len(pend_final), "resolved": sum(v is not None for v in pend_final.values())}, "subtopic": {"items": len(sub_final), "resolved": sum(v[0] is not None for v in sub_final.values()), "pass_sizes": {p: len(v) for p, v in ST.items()}}}
    json.dump(summ, open(OUT / "topics_v2_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
