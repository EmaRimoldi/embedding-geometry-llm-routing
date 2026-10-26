"""Is embedding routing task classification? Decision-level analysis of stored units.

For every stored unit (rep, transform, decoder, split) with per-query choices:
  * majority_share      fraction of queries routed to their family's modal model
                        (the router's own mode), and to the domain-best model;
  * minority set M      queries routed away from the family mode;
  * on M: reward of the chosen model vs the counterfactual reward of the family
    mode on the same queries (delta), strict-better / strict-worse rates;
  * permutation null: keep each family's multiset of minority choices but assign
    them to random queries of the family (1,000 draws) -> null distribution of
    delta.  Observed delta above the null's 97.5th percentile = informed
    deviations; inside = indistinguishable from noise.
Writes results/decisions/<panel>/<rep>/<unit>.json and a summary CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, PANELS, build_targets, load_panel  # noqa: E402
from rb.report import baseline_rewards  # noqa: E402

PROBES = ANALYSIS_ROOT / "results/probes"
OUT = ANALYSIS_ROOT / "results/decisions"


def analyse(choice: np.ndarray, covered: np.ndarray, reward: np.ndarray, groups: np.ndarray,
            domain_best: np.ndarray, disagreement: np.ndarray, n_boot: int = 1000, seed: int = 0,
            model_strength: np.ndarray | None = None, model_cost: np.ndarray | None = None,
            difficulty: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    r = np.nan_to_num(reward, nan=0.0)
    n = len(choice)
    mode = np.full(n, -1)
    for g in np.unique(groups):
        rows = np.flatnonzero((groups == g) & covered)
        if len(rows):
            mode[rows] = np.bincount(choice[rows]).argmax()
    cov = covered & (mode >= 0)
    minority = cov & (choice != mode)
    out = {
        "n": int(cov.sum()),
        "majority_share": float(np.mean(choice[cov] == mode[cov])),
        "domain_best_share": float(np.mean(choice[cov] == domain_best[cov])),
        "mode_equals_domain_best_share": float(np.mean(mode[cov] == domain_best[cov])),
        "minority_n": int(minority.sum()),
        "minority_share": float(minority.mean() / max(cov.mean(), 1e-9)),
        "n_distinct_models_used": int(len(np.unique(choice[cov]))),
    }
    if minority.sum() < 20:
        out["minority_analysis"] = None
        return out
    idx = np.flatnonzero(minority)
    r_chosen = r[idx, choice[idx]]
    r_mode = r[idx, mode[idx]]
    delta = r_chosen - r_mode
    strict_better = (r_chosen > r_mode + 1e-9)
    strict_worse = (r_chosen < r_mode - 1e-9)
    # Null: same minority choices, random queries of the same family.
    null = []
    fam_rows = {g: np.flatnonzero((groups == g) & cov) for g in np.unique(groups[idx])}
    fam_idx = {g: idx[groups[idx] == g] for g in fam_rows}
    for _ in range(n_boot):
        tot = 0.0
        for g, rows_m in fam_idx.items():
            pick = rng.choice(fam_rows[g], size=len(rows_m), replace=False)
            tot += np.sum(r[pick, choice[rows_m]] - r[pick, mode[pick]])
        null.append(tot / len(idx))
    null = np.asarray(null)
    # Same statistic restricted to disagreement queries (where a deviation can matter).
    dis = disagreement[idx]
    # Where do deviations go, and are deviation queries harder?
    direction = {}
    if model_strength is not None:
        stronger = model_strength[choice[idx]] > model_strength[mode[idx]] + 1e-9
        weaker = model_strength[choice[idx]] < model_strength[mode[idx]] - 1e-9
        direction["to_stronger_frac"] = float(stronger.mean()); direction["to_weaker_frac"] = float(weaker.mean())
        direction["delta_when_stronger"] = float(delta[stronger].mean()) if stronger.any() else float("nan")
        direction["delta_when_weaker"] = float(delta[weaker].mean()) if weaker.any() else float("nan")
    if model_cost is not None and np.isfinite(model_cost).all():
        cheaper = model_cost[choice[idx]] < model_cost[mode[idx]] * (1 - 1e-9)
        direction["to_cheaper_frac"] = float(cheaper.mean())
        direction["delta_when_cheaper"] = float(delta[cheaper].mean()) if cheaper.any() else float("nan")
        direction["cost_ratio_chosen_over_mode"] = float(np.median(model_cost[choice[idx]] / np.maximum(model_cost[mode[idx]], 1e-12)))
    if difficulty is not None:
        stay = cov & (choice == mode)
        direction["difficulty_deviation_queries"] = float(difficulty[idx].mean())
        direction["difficulty_stay_queries"] = float(difficulty[stay].mean())
        direction["mode_fails_on_deviation_queries"] = float((r_mode < 0.5).mean())
        direction["mode_fails_on_stay_queries"] = float((r[np.flatnonzero(stay), mode[stay]] < 0.5).mean())
    out["minority_analysis"] = {
        "reward_chosen": float(r_chosen.mean()), "reward_family_mode_counterfactual": float(r_mode.mean()),
        "delta": float(delta.mean()), "strict_better_rate": float(strict_better.mean()),
        "strict_worse_rate": float(strict_worse.mean()),
        "null_delta_mean": float(null.mean()), "null_delta_p025": float(np.percentile(null, 2.5)),
        "null_delta_p975": float(np.percentile(null, 97.5)),
        "z_vs_null": float((delta.mean() - null.mean()) / max(null.std(), 1e-9)),
        "informed": bool(delta.mean() > np.percentile(null, 97.5)),
        "delta_on_disagreement": float(delta[dis].mean()) if dis.any() else float("nan"),
        "minority_disagreement_frac": float(dis.mean()),
        "direction": direction,
        "per_family": {int(g): {"n_minority": int(len(rows)), "family_n": int(len(fam_rows[g])),
                                "delta": float(np.mean(r[rows, choice[rows]] - r[rows, mode[rows]])),
                                "models_used": np.bincount(choice[rows]).nonzero()[0].tolist()}
                       for g, rows in fam_idx.items()},
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panels", nargs="*", default=list(PANELS))
    p.add_argument("--decoders", nargs="*", default=["ridge", "logisticcv", "logistic", "knn20", "knn100", "kmeans7", "kmeans64", "kmeans256", "mlp"])
    p.add_argument("--transforms", nargs="*", default=["raw", "unit", "whiten", "resid", "onehot+raw", "onehot", "controls"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = []
    for panel in args.panels:
        pnl = load_panel(panel)
        t = build_targets(pnl)
        base = {s: baseline_rewards(t, s) for s in ("random", "within", "lodo")}
        # domain_best model index per query per split (recomputed deterministically).
        for rep_dir in sorted((PROBES / panel).glob("*")):
            for npz in sorted(rep_dir.glob("*.npz")):
                transform, decoder, split = npz.stem.split("__")
                if decoder not in args.decoders or transform not in args.transforms:
                    continue
                target = OUT / panel / rep_dir.name / f"{npz.stem}.json"
                if target.exists() and not args.force:
                    res = json.loads(target.read_text())
                else:
                    z = np.load(npz)
                    # domain-best index: argmax of train-fold means -> reconstruct via baseline choice reward? We need
                    # the index; recompute from the same fold logic used in probes.
                    from rb.probes import make_folds
                    db = np.full(pnl.n, -1)
                    for tr, te in make_folds(split, t):
                        for g in np.unique(t.dataset_index[te]):
                            rows_tr = tr[t.dataset_index[tr] == g]
                            rows_te = te[t.dataset_index[te] == g]
                            db[rows_te] = int(np.nanargmax(np.nanmean(t.reward[rows_tr], axis=0))) if len(rows_tr) else \
                                int(np.nanargmax(np.nanmean(t.reward[tr], axis=0)))
                    strength = np.nanmean(t.reward, axis=0)
                    res = analyse(z["choice"].astype(int), z["covered"], t.reward, t.dataset_index, db, t.disagreement,
                                  model_strength=strength, model_cost=t.model_cost_scale, difficulty=t.difficulty)
                    res.update({**pnl.provenance(), "rep": rep_dir.name, "transform": transform, "decoder": decoder, "split": split})
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(res, indent=1, default=float))
                ma = res.get("minority_analysis") or {}
                rows.append({"panel": panel, "rep": rep_dir.name, "transform": transform, "decoder": decoder, "split": split,
                             "n": res["n"], "majority_share": res["majority_share"], "domain_best_share": res["domain_best_share"],
                             "minority_share": res["minority_share"], "models_used": res["n_distinct_models_used"],
                             "min_delta": ma.get("delta"), "min_null_p975": ma.get("null_delta_p975"), "min_z": ma.get("z_vs_null"),
                             "informed": ma.get("informed"), "strict_better": ma.get("strict_better_rate"),
                             "strict_worse": ma.get("strict_worse_rate"),
                             **{f"dir_{k}": v for k, v in (ma.get("direction") or {}).items()}})
                print(f"{panel} {rep_dir.name} {npz.stem}: majority={res['majority_share']:.3f} minority={res['minority_share']:.3f} "
                      f"delta={ma.get('delta', float('nan')):+.4f} null97.5={ma.get('null_delta_p975', float('nan')):+.4f} informed={ma.get('informed')}", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summary.csv", index=False)
    print(f"wrote {OUT / 'summary.csv'} ({len(df)} units)")


if __name__ == "__main__":
    main()
