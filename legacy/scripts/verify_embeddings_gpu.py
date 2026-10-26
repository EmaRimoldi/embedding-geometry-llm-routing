"""GPU verification: re-embed random rows and compare with the stored arrays;
check that left truncation really keeps the tail; check cached-array alignment.
Writes results/verification/gpu.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.reps import list_reps, load_rep  # noqa: E402
from embed_gpu import ST_MODELS  # noqa: E402
from embed_gpu import _register_optional_flash_attention_stub  # noqa: E402

OUT = ANALYSIS_ROOT / "results/verification"
report: dict = {"checks": []}


def check(name, ok, detail=""):
    report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def cos(a, b):
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    from sentence_transformers import SentenceTransformer

    panel = load_panel("routerbench-0shot")
    rng = np.random.default_rng(0)
    texts = [str(t) for t in panel.origin_queries]
    lengths = np.array([len(t) for t in texts])
    sample = np.concatenate([rng.choice(panel.n, 16, replace=False), np.argsort(-lengths)[:4]])  # include 4 longest

    for name in ST_MODELS:
        spec = next((s for s in list_reps("routerbench-0shot") if s.name == f"computed/{name}@prompt"), None)
        if spec is None:
            check(f"{name}: stored array present", False)
            continue
        stored = load_rep(spec, panel.n, panel.keep)[sample]
        path, max_seq, prompt, dtype, batch = ST_MODELS[name]
        try:
            kwargs = {"attn_implementation": "sdpa"} if "qwen" in path.lower() else {}
            if dtype:
                kwargs["torch_dtype"] = getattr(torch, dtype)
            if "gte-Qwen2" in path:
                _register_optional_flash_attention_stub()
            model = SentenceTransformer(path, trust_remote_code=True, device="cuda", model_kwargs=kwargs)
            model.max_seq_length = max_seq
            model.tokenizer.truncation_side = "left"
            enc = lambda ts: model.encode(  # noqa: E731
                [prompt + t for t in ts] if (prompt and prompt.endswith(": ")) else ts,
                prompt_name=None if (prompt is None or prompt.endswith(": ")) else prompt,
                batch_size=4, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
            fresh = enc([texts[i] for i in sample])
            c = cos(stored, fresh)
            check(f"{name}: re-embedded 20 rows match stored (min cosine)", c.min() > 0.995,
                  f"min={c.min():.5f} mean={c.mean():.5f} (bf16 models tolerate ~1e-3)")
            # Left-truncation: for the longest sampled text exceeding max_seq, tail-only must equal full.
            tok = model.tokenizer
            long_i = max(sample, key=lambda i: len(tok(texts[i], add_special_tokens=False)["input_ids"]))
            ids = tok(texts[long_i], add_special_tokens=False)["input_ids"]
            if len(ids) > max_seq + 20:
                tail = tok.decode(ids[-(max_seq - 8):])
                head = tok.decode(ids[: max_seq - 8])
                v_full, v_tail, v_head = enc([texts[long_i], tail, head])
                c_tail = cos(v_full[None], v_tail[None])[0]
                c_head = cos(v_full[None], v_head[None])[0]
                check(f"{name}: left truncation keeps the tail", c_tail > 0.98 and c_tail > c_head,
                      f"cos(full,tail)={c_tail:.4f} cos(full,head)={c_head:.4f} tokens={len(ids)} max_seq={max_seq}")
            else:
                check(f"{name}: left truncation test skipped (no sampled text > max_seq)", True, f"max tokens in sample={len(ids)}")
            del model
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            check(f"{name}: verification run", False, f"{type(exc).__name__}: {exc}")

    # Cached (normalised, right-truncated) GTE@prompt vs computed GTE@prompt: alignment on short prompts.
    cached = next((s for s in list_reps("routerbench-0shot") if s.name == "historical/gte-qwen2-7b@prompt"), None)
    comp = next((s for s in list_reps("routerbench-0shot") if s.name == "computed/gte-qwen2-7b@prompt"), None)
    if cached and comp:
        a = load_rep(cached, panel.n, panel.keep)
        b = load_rep(comp, panel.n, panel.keep)
        short = np.flatnonzero(np.array([len(p) for p in panel.prompts]) < 4000)
        idx = rng.choice(short, 500, replace=False)
        c = cos(a[idx], b[idx])
        check("cached GTE@prompt aligned with computed GTE@prompt on short prompts", np.median(c) > 0.98,
              f"median cosine={np.median(c):.4f} p05={np.percentile(c, 5):.4f} (cached=fp32 right-trunc, computed=bf16 left-trunc)")
        # And misaligned by one row should be much worse.
        c_shift = cos(a[idx], b[(idx + 1) % panel.n])
        check("cached/computed GTE: shifted rows are dissimilar (alignment is row-exact)",
              np.median(c_shift) < np.median(c) - 0.1, f"median shifted cosine={np.median(c_shift):.4f}")

    (OUT / "gpu.json").write_text(json.dumps(report, indent=1))
    print(f"\n{len(report['checks'])} checks, {sum(not c['ok'] for c in report['checks'])} failed")


if __name__ == "__main__":
    main()
