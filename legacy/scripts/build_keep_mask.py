"""Build the row filter for a panel: keep a query iff it fits, whole, in the
architectural window of EVERY encoder used in this phase (prefix included).
Writes features/<panel>/keep_mask.npz with the per-encoder token counts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# encoder -> (tokenizer path, architectural window, prefix prepended before encoding)
ENCODERS = {
    "minilm-l6": ("sentence-transformers/all-MiniLM-L6-v2", 512, ""),
    "minilm-l12": (str(RUNTIME_ROOT / "artifacts/models/all-MiniLM-L12-v2__3392ab41"), 512, ""),
    "e5-large-v2": ("intfloat/e5-large-v2", 512, "query: "),
    "bge-m3": (str(RUNTIME_ROOT / "artifacts/models/bge-m3"), 8192, ""),
    "gte-qwen2-7b": (str(RUNTIME_ROOT / "artifacts/models/gte-Qwen2-7B-instruct"), 8192, ""),
    "gte-qwen2-7b-instr": (str(RUNTIME_ROOT / "artifacts/models/gte-Qwen2-7B-instruct"), 8192,
                           "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "),
}

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--panel", default="routerbench-0shot"); p.add_argument("--field", default="prompt")
    a = p.parse_args()
    panel = load_panel(a.panel, filtered=False)
    texts = [str(t) for t in panel.text(a.field)]
    keep = np.ones(panel.n, dtype=bool); counts = {}
    for name, (path, window, prefix) in ENCODERS.items():
        tok = AutoTokenizer.from_pretrained(path)
        n = np.array([len(tok(prefix + t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts])
        counts[name] = n; keep &= n <= window
        print(f"{name}: window={window} over={int((n > window).sum())}", flush=True)
    out = ANALYSIS_ROOT / "features" / a.panel / "keep_mask.npz"
    np.savez(out, keep=keep, **{f"tokens_{k}": v for k, v in counts.items()})
    meta = {"panel": a.panel, "field": a.field, "n_total": int(panel.n), "n_kept": int(keep.sum()),
            "removed": int((~keep).sum()), "removed_by_family": {d: int(((panel.dataset_ids == d) & ~keep).sum()) for d in panel.datasets},
            "rule": "kept iff tokens(prefix + text, with special tokens) <= architectural window for every encoder listed",
            "encoders": {k: {"window": v[1], "prefix": v[2]} for k, v in ENCODERS.items()}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))

if __name__ == "__main__":
    main()
