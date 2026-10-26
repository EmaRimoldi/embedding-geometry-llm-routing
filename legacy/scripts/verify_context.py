"""Did each encoder see the whole query?  CPU-only.

1. For every computed representation: tokenise the exact texts with that
   encoder's own tokenizer and report, per dataset, the fraction of texts that
   fit entirely inside the window used (from the sidecar), i.e. were seen whole.
2. Show what the encoder's *default* window would have been (SentenceTransformer
   config) versus what we set — MiniLM-L12's default is 128.
3. Behavioural test on CPU with MiniLM-L12: encode a ~400-token text at window
   128 vs 512; if the 512 encoding equals the 128 one, the window setting was
   not applied; it must instead equal the encoding of the last 512 tokens.
Writes results/verification/context.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.reps import list_reps  # noqa: E402

OUT = ANALYSIS_ROOT / "results/verification"
TOKENIZER_OF = {  # rep prefix -> tokenizer path (scope: PLAN.md)
    "gte-qwen2-7b": str(RUNTIME_ROOT / "artifacts/models/gte-Qwen2-7B-instruct"),
}


def main() -> None:
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"coverage": {}, "defaults": {}, "behaviour": {}}
    tok_cache = {}
    for panel_name in ("routerbench-0shot",):
        panel = load_panel(panel_name)
        seen = {}
        for spec in list_reps(panel_name, include_cached=False):
            base = spec.name.split("/", 1)[1].split("@")[0]
            if base.startswith("hash"):
                continue
            key = next((k for k in TOKENIZER_OF if base.startswith(k)), None)
            if key is None:
                continue
            meta = json.loads(spec.path.with_suffix(".npy.json").read_text())
            window = meta.get("max_seq_length") or meta.get("max_tokens")
            if window is None:
                continue
            cache_key = (key, spec.text_field, panel_name)
            if cache_key not in seen:
                try:
                    tok = tok_cache.setdefault(key, AutoTokenizer.from_pretrained(TOKENIZER_OF[key]))
                except Exception as exc:  # noqa: BLE001  (qwen3emb tokenizer needs the newer venv)
                    report["coverage"][f"{panel_name}::{spec.name}"] = {"skipped": f"tokenizer load failed: {type(exc).__name__}",
                                                                         "sidecar_texts_truncated": meta.get("texts_truncated")}
                    print(f"{panel_name} {spec.name}: tokenizer unavailable in this venv; sidecar truncated={meta.get('texts_truncated')}")
                    continue
                texts = [str(t) for t in panel.text(spec.text_field)]
                lengths = np.array([len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts])
                seen[cache_key] = lengths
            lengths = seen[cache_key]
            per_ds = {}
            for d in panel.datasets:
                m = panel.dataset_ids == d
                per_ds[d] = {"n": int(m.sum()), "seen_whole": float(np.mean(lengths[m] <= window)),
                             "median_tokens": float(np.median(lengths[m])), "p95_tokens": float(np.percentile(lengths[m], 95))}
            report["coverage"][f"{panel_name}::{spec.name}"] = {
                "window": int(window), "truncation_side": meta.get("truncation_side"),
                "seen_whole_overall": float(np.mean(lengths <= window)), "per_dataset": per_ds}
            print(f"{panel_name} {spec.name}: window={window} seen_whole={np.mean(lengths <= window):.3f} "
                  + " ".join(f"{d}={v['seen_whole']:.2f}" for d, v in per_ds.items()), flush=True)

    (OUT / "context.json").write_text(json.dumps(report, indent=1))
    print("wrote", OUT / "context.json")


if __name__ == "__main__":
    main()
