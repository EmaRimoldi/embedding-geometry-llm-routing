"""exp26 step 1 (GPU job): strip each family's harness template from the prompts and embed the stripped text ("core") with GTE.
Rule: results/deformatting/deformat.py (per-family longest common prefix / suffix, trimmed to line breaks). Writes into results/deformatting/panel/:
  texts.npz (core texts), rule.json (prefix/suffix per family + examples), gte-qwen2-7b.npy (+ .json sidecar)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(HERE))
from rb.data import load_panel  # noqa: E402
from embed_gpu import run_st  # noqa: E402
from deformat import PANEL_DIR, deformat  # noqa: E402
from rb.panelctx import REP_NAME  # noqa: E402


def main() -> None:
    panel = load_panel("routerbench-0shot"); prompts = [str(t) for t in panel.prompts]
    core, rule = deformat(prompts, panel.dataset_ids, panel.datasets)
    for f, r in rule.items(): print(f"{f}: prefix {r['prefix_chars']} suffix {r['suffix_chars']} core median {r['core_chars_median']:.0f}", flush=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    enc = [t.encode("utf-8") for t in core]; offsets = np.cumsum([0] + [len(e) for e in enc]).astype(np.int64)
    np.savez_compressed(PANEL_DIR / "texts.npz", utf8=np.frombuffer(b"".join(enc), dtype=np.uint8), offsets=offsets)
    (PANEL_DIR / "rule.json").write_text(json.dumps({**panel.provenance(), "rule": "per-family longest common prefix / suffix, trimmed to line breaks", "families": rule}, indent=1))
    run_st(REP_NAME, core, PANEL_DIR / f"{REP_NAME}.npy", "routerbench-0shot deformatted (results/deformatting/panel): prompts minus the per-family harness template (rule.json); same encoder run as the templated representation; left truncation; unnormalised")
    print("done", flush=True)


if __name__ == "__main__":
    main()
