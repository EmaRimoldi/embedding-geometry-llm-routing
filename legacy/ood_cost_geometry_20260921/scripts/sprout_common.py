"""SPROUT loader for this folder (reads experiments4/sprout/ only; writes nothing there).
Six conceptual families (gpqa, math, mmlupro, musr, openhermes, ragbench). Topic labels: the authoritative ones
(MMLU-Pro category, MATH type, MuSR task) plus the accepted two-pass Qwen2.5-32B labels; the 2,274 prompts still
pending manual review fall back to the family name (the rule the LRB 'audited' set used). Rows above GTE's context
(74, excluded by the frozen keep mask) are dropped; exact duplicate prompt strings are removed (first occurrence
kept) so that no identical prompt sits on both sides of a split. Success = reward >= 0.5 (judge score)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RESULTS = HERE.parent / "results"; E4 = HERE.parents[1]; ROOT = E4 / "sprout"


def unit(a):
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def load(dedup: bool = True, with_x: bool = True, topics: str | None = None, drop_models: tuple = ()):
    """topics: None/'v1' = authoritative + accepted Qwen subject (+ family fallback); 'v2' = refined table from
    results/sprout_labels/topics_v2.jsonl (build_sprout_topics_v2.py). Env SPROUT_TOPICS overrides when topics is None."""
    import os
    topics = topics or os.environ.get("SPROUT_TOPICS", "v1")
    rows = [json.loads(l) for l in open(ROOT / "panel" / "rows.jsonl")]; meta = json.load(open(ROOT / "panel" / "panel.json"))
    km = np.load(ROOT / "features" / "gte-qwen2-7b_rows.npz"); keep = km["keep"].astype(bool)
    assert np.array_equal(km["key"], np.asarray([r["key"] for r in rows])) and len(rows) == meta["rows"]
    z = np.load(ROOT / "panel" / "sprout.npz"); kept_idx = np.where(keep)[0]
    sel, seen = [], set()
    for j, i in enumerate(kept_idx):
        p = rows[i]["prompt"]
        if dedup and p in seen: continue
        seen.add(p); sel.append(j)
    sel = np.asarray(sel); idx = kept_idx[sel]
    x = None
    if with_x:
        xm = np.load(ROOT / "features" / "gte-qwen2-7b.npy", mmap_mode="r"); assert xm.shape[0] == int(keep.sum())
        x = np.asarray(xm[sel], dtype=np.float32)
    r = z["rewards"][idx].astype(np.float64); c = z["costs"][idx].astype(np.float64); keep_m = [j for j, m in enumerate(meta["models"]) if m not in drop_models]; r, c = r[:, keep_m], c[:, keep_m]
    fnames = sorted({rows[i]["family"] for i in idx}); fmap = {f: g for g, f in enumerate(fnames)}
    fam = np.asarray([fmap[rows[i]["family"]] for i in idx])
    auth = {}
    for l in open(ROOT / "topics" / "authoritative_and_pending.jsonl"):
        a = json.loads(l); auth[a["key"]] = a["topic"]
    qwen = {}
    for l in open(ROOT / "topics" / "qwen32_topics.jsonl"):
        q = json.loads(l); qwen[q["prompt"]] = q["label"]
    topic, src = [], []
    for i in idx:
        t = auth.get(rows[i]["key"])
        if t is not None: topic.append(t); src.append("authoritative")
        elif qwen.get(rows[i]["prompt"]) is not None: topic.append(qwen[rows[i]["prompt"]]); src.append("qwen")
        else: topic.append(rows[i]["family"]); src.append("pending->family")
    if topics == "v2":
        v2 = {}
        for l in open(RESULTS / "sprout_labels" / "topics_v2.jsonl"):
            o = json.loads(l); v2[o["key"]] = (o["topic_v2"], o["source_v2"])
        topic = [v2[rows[i]["key"]][0] for i in idx]; src = [v2[rows[i]["key"]][1] for i in idx]
    topic = np.asarray(topic, dtype=object); src = np.asarray(src, dtype=object)
    cell_str = np.asarray([f"{rows[i]['family']}/{t}" for i, t in zip(idx, topic)], dtype=object)
    return {"name": "sprout", "r": r, "c": c, "fam": fam, "fnames": fnames, "x": x, "names": [meta["models"][j] for j in keep_m], "topic": topic, "topic_src": src,
            "cell_str": cell_str, "dataset": np.asarray([rows[i]["dataset"] for i in idx], dtype=object), "split": np.asarray([rows[i]["split"] for i in idx], dtype=object),
            "prompts": np.asarray([rows[i]["prompt"] for i in idx], dtype=object), "keys": [rows[i]["key"] for i in idx], "n_total": len(rows), "n_kept": int(keep.sum()), "n": len(idx),
            "reward_sha256": meta["reward_sha256"], "cost_sha256": meta["cost_sha256"]}


def provenance(D):
    return {"panel": "sprout (experiments4/sprout, HF CARROT-LLM-Routing/SPROUT rev d54e3cd)", "rows_total": D["n_total"], "rows_in_gte_context": D["n_kept"], "rows_after_dedup": D["n"],
            "models": D["names"], "families": D["fnames"], "family_counts": {f: int((D["fam"] == g).sum()) for g, f in enumerate(D["fnames"])},
            "topic_sources": {str(k): int((D["topic_src"] == k).sum()) for k in np.unique(D["topic_src"])}, "n_cells": int(len(np.unique(D["cell_str"]))),
            "reward_sha256": D["reward_sha256"], "cost_sha256": D["cost_sha256"], "success_rule": "reward >= 0.5"}


def save(d: Path, result: dict, md_lines: list[str]):
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(result, indent=1, default=float)); (d / "result.md").write_text("\n".join(md_lines) + "\n"); print("wrote", d, flush=True)


def knn_idx(xq, xs, k, chunk=1024, exclude_self=False):
    """indices (into xs) of the k nearest by cosine (unit vectors), sorted by decreasing similarity."""
    out = np.empty((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), chunk):
        S = xq[s:s + chunk] @ xs.T
        if exclude_self: S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        p = np.argpartition(-S, k - 1, axis=1)[:, :k]; o = np.argsort(-np.take_along_axis(S, p, 1), axis=1); out[s:s + chunk] = np.take_along_axis(p, o, 1)
    return out
