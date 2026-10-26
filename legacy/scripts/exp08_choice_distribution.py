"""Post-hoc read-out of exp08/exp09: which models each router actually picks (split 0), per λ.

Reports per method: share of each model, number of distinct models used, entropy (bits), and the
share of queries routed to the same model as the family lookup. Oracle included as the reference.
Writes results/basic_model_tests/choice_distribution_seed0.{json,md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.sweep import LAMBDAS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

BASE = RESULTS_ROOT / "basic_model_tests"
KS = (5, 10, 40, 100)
LAMS = (0.0, 10.0, 100.0, 1000.0)


def entropy(p):
    p = p[p > 0]; return float(-(p * np.log2(p)).sum())


def main() -> None:
    panel = load_panel(PANEL); names = [m[:18] for m in panel.models]; M = panel.m
    x = load_rep(find_rep(PANEL, REP), panel.n, panel.keep)
    u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0); fam = panel.dataset_index(); F = len(panel.datasets)
    tr, te = split(panel.n, fam, 0)
    kmax = max(KS); nb = np.zeros((len(te), kmax), dtype=np.int64)
    for s in range(0, len(te), 2048):
        sims = u[te[s:s + 2048]] @ u[tr].T; top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
        nb[s:s + 2048] = np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)
    out = {}; md = ["# Model-choice distribution, split 0 (test rows)", ""]
    for lam in LAMS:
        U = r - lam * c; Utr = U[tr]; table = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)])
        choices = {"oracle": U[te].argmax(1), "best_model_per_family": table[fam[te]], "best_single": np.full(len(te), int(Utr.mean(0).argmax()))}
        for k in KS: choices[f"knn{k}"] = Utr[nb[:, :k]].mean(1).argmax(1)
        li = LAMBDAS.index(lam); z = BASE / "mlp" / "predictions" / f"seed0_lam{li}.npz"
        if z.exists():
            zz = np.load(z); choices["mlp100 (exp09)"] = zz["mlp100"].argmax(1); choices["mlp100x100 (exp09)"] = zz["mlp100x100"].argmax(1)
        out[str(lam)] = {}
        md += [f"## λ = {lam:g}", "", "method | distinct models | entropy (bits) | agrees with family lookup | top models (share)", "---|---|---|---|---"]
        for name, ch in choices.items():
            share = np.bincount(ch, minlength=M) / len(ch); order = np.argsort(-share)
            out[str(lam)][name] = {"share": {names[j]: float(share[j]) for j in range(M)}, "distinct": int((share > 0).sum()), "entropy_bits": entropy(share),
                                   "agree_with_lookup": float((ch == choices["best_model_per_family"]).mean())}
            md.append(f"{name} | {int((share > 0).sum())} | {entropy(share):.2f} | {float((ch == choices['best_model_per_family']).mean()):.3f} | " + ", ".join(f"{names[j]} {share[j]:.2f}" for j in order[:4] if share[j] > 0))
        md.append("")
    (BASE / "choice_distribution_seed0.json").write_text(json.dumps(out, indent=1)); (BASE / "choice_distribution_seed0.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
