"""Outcome-free geometry + locality for every available representation of a panel.

Writes results/geometry/<panel>/<rep>.json and results/geometry/<panel>/cka.json.
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
from rb.geometry_tests import (cluster_alignment, dataset_identifiability, dataset_variance_fraction,  # noqa: E402
                         linear_cka, spectrum_stats)
from rb.geometry_tests import neighbour_analysis, pair_analysis  # noqa: E402
from rb.reps import list_reps, load_rep, safe_name  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--reps", nargs="*", default=None)
    p.add_argument("--skip-cka", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    panel = load_panel(args.panel)
    targets = build_targets(panel)
    out_dir = ANALYSIS_ROOT / "results/geometry" / args.panel
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [s for s in list_reps(args.panel) if not args.reps or s.name in args.reps]
    print(f"{args.panel}: {len(specs)} representations", flush=True)

    loaded = {}
    for spec in specs:
        target = out_dir / f"{safe_name(spec.name)}.json"
        x = load_rep(spec, panel.n, panel.keep)
        loaded[spec.name] = x
        if target.exists() and not args.force:
            continue
        t0 = time.time()
        controls, names = text_controls(panel.text(spec.text_field))
        norms = np.linalg.norm(x, axis=1)
        res = {
            **panel.provenance(),
            "rep": spec.name, "family": spec.family, "source": spec.source, "text_field": spec.text_field,
            "normalized": spec.normalized, "note": spec.note,
            "spectrum": spectrum_stats(x),
            "dataset_variance": dataset_variance_fraction(x, targets.dataset_index),
            "dataset_identifiability": dataset_identifiability(x, targets.dataset_index),
            "cluster_alignment": cluster_alignment(x, targets.dataset_index),
            "pairs": pair_analysis(x, targets),
            "neighbours": neighbour_analysis(x, targets),
        }
        if not spec.normalized:
            from scipy.stats import spearmanr
            res["norm"] = {
                "spearman_norm_difficulty": float(spearmanr(norms, targets.difficulty).correlation),
                "spearman_norm_log_chars": float(spearmanr(norms, controls[:, 0]).correlation),
                "spearman_norm_difficulty_within_dataset": float(np.mean([
                    spearmanr(norms[targets.dataset_index == g], targets.difficulty[targets.dataset_index == g]).correlation
                    for g in np.unique(targets.dataset_index) if (targets.dataset_index == g).sum() > 30])),
            }
        res["elapsed_s"] = time.time() - t0
        target.write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
        print(f"{spec.name}: eff_rank={res['spectrum']['effective_rank_exp_entropy']:.1f} eta2={res['dataset_variance']['eta2_total']:.3f} "
              f"ds_probe={res['dataset_identifiability']['linear_probe_accuracy']:.3f} "
              f"knn20_gain={res['neighbours']['20']['agreement_knn'] - res['neighbours']['20']['agreement_random_same_dataset']:.4f} "
              f"({res['elapsed_s']:.0f}s)", flush=True)

    if not args.skip_cka and len(loaded) > 1:
        names = list(loaded)
        cka = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cka[f"{a} || {b}"] = linear_cka(loaded[a], loaded[b])
        (out_dir / "cka.json").write_text(json.dumps(cka, indent=1, sort_keys=True))
        print("CKA written", flush=True)


if __name__ == "__main__":
    main()
