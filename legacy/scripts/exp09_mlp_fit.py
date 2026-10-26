"""exp09 worker: one (split, λ): fit RouterBench's per-model MLPs on u_λ (raw GTE input), store test predictions.

Usage: exp09_mlp_fit.py --seed S --lam-index L   (see experiments/exp09_mlp.yaml)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

ARCHS = {"mlp100": (100,), "mlp100x100": (100, 100)}
OUT = RESULTS_ROOT / "basic_model_tests" / "mlp" / "predictions"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); p.add_argument("--lam-index", type=int, required=True)
    a = p.parse_args(); lam = LAMBDAS[a.lam_index]
    out = OUT / f"seed{a.seed}_lam{a.lam_index}.npz"
    if out.exists():
        print("exists", out); return
    panel = load_panel(PANEL)
    x = load_rep(find_rep(PANEL, REP), panel.n, panel.keep)   # raw, as published
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    U = r - lam * c
    tr, te = split(panel.n, panel.dataset_index(), a.seed)
    preds = {}
    t0 = time.time()
    for name, hidden in ARCHS.items():
        P = np.zeros((len(te), panel.m), dtype=np.float32)
        for m in range(panel.m):
            net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam", learning_rate="constant",
                               learning_rate_init=1e-3, max_iter=200, random_state=1234)
            net.fit(x[tr], U[tr, m])
            P[:, m] = net.predict(x[te])
        preds[name] = P
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, test_idx=te, lam=lam, seed=a.seed, **preds)
    print("wrote", out)


if __name__ == "__main__":
    main()
