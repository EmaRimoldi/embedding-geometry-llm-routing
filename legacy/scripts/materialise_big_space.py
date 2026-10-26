"""Materialise the 40k space of exp45 as a normal stored representation.

Writing it to features/<panel>/prompt/big40k.npy makes it visible to rb.reps as
``computed/big40k@prompt``, so every existing experiment runs on it unchanged through the
E4_REP switch: the geometry chapter (exp01 K-means against the families, exp04 MMLU subjects,
exp05 neighbour composition), the router chapter (exp08, exp09) and the baselines
(Avengers-Pro, RouteLLM-MF, GraphRouter).  That reuses validated code instead of re-deriving
the metrics here.

Difference from exp45, and it is deliberate: the per-block standardisation is fitted on the whole
panel rather than per split.  Standardisation uses no outcome, so this matches how every other
stored representation in the study is produced (GTE, hashed bag-of-words, the reader states are all
computed once over all rows).  Blocks are clipped at +-5 sd and divided by the square root of their
width, which is exp45's `big_balanced` scaling, the one that treats every source equally.
Usage: materialise_big_space.py [--target-dim 40000]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, lengths_features  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp45_big_space import BLOCKS, CLIP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--target-dim", type=int, default=40000); ap.add_argument("--name", default="big40k"); a = ap.parse_args()
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n
    HROOT = ANALYSIS_ROOT / "features" / PANEL / "prompt"

    def take(arr):
        arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr
        assert arr.shape[0] == n, arr.shape; return np.asarray(arr, dtype=np.float32)

    def load(spec):
        kind, mdl, lay = spec
        if kind == "hidden":
            f = HROOT / "hidden" / mdl / f"{lay}.npy"; return take(np.load(f, mmap_mode="r")) if f.exists() else None
        if kind == "delta":
            fa, fb = (HROOT / "hidden" / mdl / f"{x}.npy" for x in lay)
            return take(np.load(fa, mmap_mode="r")) - take(np.load(fb, mmap_mode="r")) if fa.exists() and fb.exists() else None
        if kind == "rep": return load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)
        if kind == "bow":
            f = HROOT / "hash-bow.npy"; return take(np.load(f, mmap_mode="r")) if f.exists() else None
        if kind == "surprise":
            f = HROOT / "qwen7b_hidden" / "prompt_logprob_stats.npz"; return take(np.load(f)["stats"]) if f.exists() else None
        if kind == "lengths": return lengths_features(panel)[0].astype(np.float32)
        return None

    parts, manifest, width = [], [], 0
    for name, spec, _ in BLOCKS:
        if width >= a.target_dim: break
        x = load(spec)
        if x is None:
            print(f"  skip {name} (missing)", flush=True); continue
        xs = np.clip(StandardScaler().fit_transform(x), -CLIP, CLIP).astype(np.float32)
        parts.append((xs / np.sqrt(xs.shape[1])).astype(np.float32))
        manifest.append({"block": name, "dim": int(xs.shape[1])}); width += xs.shape[1]
        print(f"  + {name}: {xs.shape[1]} -> {width} ({time.time() - t0:.0f}s)", flush=True); del x, xs
    X = np.column_stack(parts); del parts
    out = HROOT / f"{a.name}.npy"; np.save(out, X, allow_pickle=False)
    meta = {**panel.provenance(), "rep": a.name, "dims": int(X.shape[1]), "blocks": manifest, "clip": CLIP,
            "scaling": "per-block standardisation on the whole panel (no outcome used), clipped, divided by sqrt(block width)",
            "note": f"exp45 big_balanced space materialised; {len(manifest)} blocks; built by scripts/materialise_big_space.py",
            "normalized": False, "seconds": time.time() - t0}
    out.with_suffix(".npy.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
