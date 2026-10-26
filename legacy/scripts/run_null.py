"""Permutation null for G: shuffle the representation rows within each dataset
(keeps dataset identity, destroys every within-dataset association), then run
the same decoders. Writes results/null/<panel>/<rep>/seed<k>__<decoder>__<split>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, build_targets, load_panel, text_controls  # noqa: E402
from rb.probes import evaluate, fit_predict_oof  # noqa: E402
from rb.reps import find_rep, load_rep, safe_name  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--rep", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--decoders", nargs="*", default=["ridge", "logisticcv", "knn20"])
    p.add_argument("--splits", nargs="*", default=["random", "within", "lodo"])
    p.add_argument("--transform", default="raw")
    args = p.parse_args()

    panel = load_panel(args.panel)
    t = build_targets(panel)
    cost = np.where(panel.available, panel.costs, np.nan)
    spec = find_rep(args.panel, args.rep)
    x = load_rep(spec, panel.n, panel.keep)
    controls, _ = text_controls(panel.text(spec.text_field))
    out_dir = ANALYSIS_ROOT / "results/null" / args.panel / safe_name(spec.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        xs = x.copy()
        for g in np.unique(t.dataset_index):
            rows = np.flatnonzero(t.dataset_index == g)
            xs[rows] = x[rng.permutation(rows)]
        for decoder in args.decoders:
            for split in args.splits:
                target = out_dir / f"seed{seed}__{args.transform}__{decoder}__{split}.json"
                if target.exists():
                    continue
                ev = evaluate(fit_predict_oof(xs, t, args.transform, decoder, split, controls), t, cost)
                ev.pop("per_query")
                ev.update({**panel.provenance(), "rep": spec.name, "null_seed": seed, "transform": args.transform,
                           "decoder": decoder, "split": split})
                target.write_text(json.dumps(ev, indent=1, sort_keys=True, default=float))
                s = ev["subsets"]["disagreement"]
                print(f"seed{seed} {decoder} {split}: G_null={s['gap_closed_vs_domain_best']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
