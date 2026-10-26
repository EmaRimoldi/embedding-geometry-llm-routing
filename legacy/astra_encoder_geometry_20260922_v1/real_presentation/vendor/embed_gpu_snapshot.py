"""Compute unnormalised representations for a routing panel on one GPU.

Scope: the encoders listed in ``ST_MODELS`` (see PLAN.md).

All encoders truncate from the LEFT (tail preserved) so that the question, which
sits at the end of every template in these panels, is never cut.  This is
deliberately different from the project's historical right-truncated caches and
is recorded in every sidecar.  Outputs are float32, never normalised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402

MODELS_DIR = RUNTIME_ROOT / "artifacts/models"


def _register_optional_flash_attention_stub() -> bool:
    """Let Transformers load GTE's optional FlashAttention branch without flash_attn.

    The pinned GTE remote module imports ``flash_attn`` only if Transformers
    reports it installed, but Transformers' static scanner rejects the file when
    the package is absent.  A metadata-free stub satisfies the scanner while
    ``is_flash_attn_2_available()`` stays false, so the SDPA path runs.
    (Same convention as the project's own feature builder; re-implemented here so
    this folder imports nothing from the shared source tree.)
    """
    from importlib.machinery import ModuleSpec
    from importlib.metadata import PackageNotFoundError, version
    from types import ModuleType
    try:
        version("flash_attn")
        return False
    except PackageNotFoundError:
        if "flash_attn" not in sys.modules:
            module = ModuleType("flash_attn")
            module.__spec__ = ModuleSpec("flash_attn", loader=None)
            sys.modules["flash_attn"] = module
        return True

ST_MODELS = {
    # name: (model path, max_seq_length, prompt_name/prefix, torch_dtype, batch)
    # Scope (PLAN.md): gte-Qwen2-7B-instruct used raw, as LLMRouterBench uses it.
    # Other encoders are added here only when the PI names them.
    "gte-qwen2-7b": (str(MODELS_DIR / "gte-Qwen2-7B-instruct"), 8192, None, "bfloat16", 4),
    # minilm/ study (PI, 2026-09-16): all-MiniLM-L12-v2 at its 512-token architectural window (the release config
    # says 128); rows above 512 BERT tokens are excluded by the panel, so nothing is truncated.  fp32, unnormalised
    # (the model's own Normalize module makes the vectors unit norm).
    "minilm-l12": (str(MODELS_DIR / "all-MiniLM-L12-v2__3392ab41"), 512, None, None, 256),
}



def sha256_of(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(out: Path, matrix: np.ndarray, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=np.float32)
    np.save(out, matrix, allow_pickle=False)
    meta = {**meta, "shape": list(matrix.shape), "sha256": sha256_of(matrix),
            "norm_mean": float(np.linalg.norm(matrix, axis=1).mean()),
            "norm_std": float(np.linalg.norm(matrix, axis=1).std())}
    out.with_suffix(".npy.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"wrote {out} {matrix.shape}", flush=True)


def run_st(name: str, texts: list[str], out: Path, note: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    path, max_seq, prompt, dtype, batch = ST_MODELS[name]
    # XLM-R / BERT-family encoders in this Transformers version reject the SDPA
    # switch; only the decoder-based encoders take it.
    kwargs = {"attn_implementation": "sdpa"} if ("Qwen" in path or "qwen" in path) else {}
    if dtype is not None:
        kwargs["torch_dtype"] = getattr(torch, dtype)
    if "gte-Qwen2" in path:
        _register_optional_flash_attention_stub()
    model = SentenceTransformer(path, trust_remote_code=True, device="cuda", model_kwargs=kwargs)
    model.max_seq_length = max_seq
    model.tokenizer.truncation_side = "left"
    start = time.time()
    encode_kwargs = dict(batch_size=batch, show_progress_bar=True, convert_to_numpy=True,
                         normalize_embeddings=False)
    if prompt is None:
        matrix = model.encode(texts, **encode_kwargs)
    elif prompt.endswith(": "):
        matrix = model.encode([prompt + t for t in texts], **encode_kwargs)
    else:
        matrix = model.encode(texts, prompt_name=prompt, **encode_kwargs)
    # Truncation accounting with the same tokenizer/limit used for encoding.
    lengths = np.asarray([len(model.tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"])
                          for t in texts])
    write(out, matrix, {
        "rep": name, "model": path, "max_seq_length": max_seq, "prompt": prompt,
        "torch_dtype": dtype, "truncation_side": "left", "normalized": False,
        "texts_truncated": int((lengths > max_seq).sum()), "token_len_median": float(np.median(lengths)),
        "token_len_max": int(lengths.max()), "elapsed_s": time.time() - start, "note": note,
    })
    del model
    torch.cuda.empty_cache()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True)
    parser.add_argument("--text-field", required=True, choices=["prompt", "origin_query"])
    parser.add_argument("--group", default="st", choices=["st"])
    parser.add_argument("--models", nargs="*", default=None, help="subset of ST model names")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    panel = load_panel(args.panel)
    texts = [str(t) for t in panel.text(args.text_field)]
    out_dir = ANALYSIS_ROOT / "features" / args.panel / args.text_field
    note = f"{args.panel}/{args.text_field}; computed by embed_gpu.py; left truncation; unnormalised"
    print(f"panel={args.panel} field={args.text_field} n={len(texts)} group={args.group}", flush=True)
    print("GPU:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    names = args.models or list(ST_MODELS)
    for name in names:
        out = out_dir / f"{name}.npy"
        if out.exists() and not args.force:
            print(f"skip {out} (exists)", flush=True)
            continue
        try:
            run_st(name, texts, out, note)
        except Exception as exc:  # keep going: a single encoder failure must not sink the job
            print(f"!! {name} failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
