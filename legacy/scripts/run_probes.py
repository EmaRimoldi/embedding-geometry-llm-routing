"""Run the probe grid for one representation of one panel.

Results: results/probes/<panel>/<rep>/<transform>__<decoder>__<split>.json (+ .npz per-query).
Existing units are skipped unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, build_targets, load_panel, text_controls  # noqa: E402
from rb.probes import evaluate, fit_predict_oof  # noqa: E402
from rb.reps import find_rep, load_rep, safe_name  # noqa: E402

DEFAULT_TRANSFORMS = ["raw", "unit", "whiten", "resid", "onehot+raw"]
DEFAULT_DECODERS = ["logistic", "logisticcv", "ridge", "knn5", "knn10", "knn20", "knn40", "knn50", "knn100", "kmeans7", "kmeans64", "kmeans256", "mlp", "mlp100", "mlp100x100"]
DEFAULT_SPLITS = ["random", "lodo", "within"]
# Decoders that make sense for a one-hot / few-feature control representation.
CONTROL_DECODERS = ["logistic", "logisticcv", "ridge", "knn20", "knn100", "mlp"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--rep", required=True, help="rep name from rb.reps.list_reps, or 'controls' / 'onehot'")
    p.add_argument("--transforms", nargs="*", default=None)
    p.add_argument("--decoders", nargs="*", default=None)
    p.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    panel = load_panel(args.panel)
    targets = build_targets(panel)
    cost = np.where(panel.available, panel.costs, np.nan)

    if args.rep in ("controls", "onehot"):
        x = np.zeros((panel.n, 1), dtype=np.float32)
        field = "origin_query"
        controls, _ = text_controls(panel.text(field))
        log_norm = np.zeros(panel.n)
        transforms = args.transforms or [args.rep]
        decoders = args.decoders or (["logistic", "ridge"] if args.rep == "onehot" else CONTROL_DECODERS)
        rep_name = args.rep
    else:
        spec = find_rep(args.panel, args.rep)
        x = load_rep(spec, panel.n, panel.keep)
        controls, _ = text_controls(panel.text(spec.text_field))
        log_norm = np.log(np.maximum(np.linalg.norm(x, axis=1), 1e-12))
        transforms = args.transforms or list(DEFAULT_TRANSFORMS)
        if not spec.normalized:
            transforms = transforms + ["norm"]
        decoders = args.decoders or DEFAULT_DECODERS
        rep_name = spec.name

    out_dir = ANALYSIS_ROOT / "results/probes" / args.panel / safe_name(rep_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"panel={args.panel} rep={rep_name} x={x.shape} transforms={transforms} decoders={decoders} splits={args.splits}",
          flush=True)

    for transform in transforms:
        for decoder in decoders:
            if transform in ("onehot", "controls", "norm") and decoder not in CONTROL_DECODERS:
                continue
            # Cosine decoders are the point of `unit`; linear probes on it duplicate `raw`.
            if transform == "unit" and not (decoder.startswith("knn") or decoder.startswith("kmeans")):
                continue
            # Concatenation / residualisation transforms are for linear probes (and one kNN check).
            if transform in ("onehot+raw", "surf+raw", "desurf") and decoder not in ("ridge", "logisticcv", "logistic", "mlp", "knn20"):
                continue
            if decoder.startswith("mlp") and transform not in ("raw", "controls", "onehot"):
                continue
            for split in args.splits:
                stem = f"{transform}__{decoder}__{split}"
                target = out_dir / f"{stem}.json"
                if target.exists() and not args.force:
                    continue
                t0 = time.time()
                try:
                    result = fit_predict_oof(x, targets, transform, decoder, split, controls, log_norm)
                    ev = evaluate(result, targets, cost)
                except Exception as exc:  # record and continue with the grid
                    target.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
                    print(f"!! {stem}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                per_query = ev.pop("per_query")
                np.savez_compressed(out_dir / f"{stem}.npz", **per_query)
                ev.update({**panel.provenance(), "rep": rep_name, "transform": transform, "decoder": decoder,
                           "split": split, "wall_s": time.time() - t0})
                target.write_text(json.dumps(ev, indent=1, sort_keys=True, default=float))
                s = ev["subsets"].get("all", {})
                print(f"{stem}: auroc={s.get('macro_auroc', float('nan')):.3f} diffR2={s.get('difficulty_r2', float('nan')):.3f} "
                      f"router={s.get('router_reward', float('nan')):.3f} best1={s.get('best_single_reward', float('nan')):.3f} "
                      f"oracle={s.get('oracle_reward', float('nan')):.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
