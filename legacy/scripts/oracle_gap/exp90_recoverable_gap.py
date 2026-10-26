"""exp90: direct recoverability of the oracle-minus-cell-lookup gap.

For every outer split, split the 70% training portion again into fit/calibration.
The family-topic lookup is the safe baseline.  A ridge head predicts each model's
reward and cost residual relative to its fit-set cell mean from a frozen text
representation.  On calibration rows only, choose how many queries may switch
away from the lookup (a quantile of predicted advantage).  Evaluate on untouched
test rows.

An optimistic shortlist diagnostic always includes the lookup model and adds the
k highest-ranked alternatives.  A hindsight oracle picks the realised best model
inside that shortlist.  It is compared with random alternatives of the same size;
therefore simply giving the oracle more candidates is not mistaken for text signal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb.data import SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, topic_str  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from common import REP_LABEL, base, load_rep, out_dir, save  # noqa: E402

EXP = "exp90_recoverable_gap"
MIN_CELL = 50
REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0)
QGRID = np.array([0, .01, .02, .05, .10, .20, .35, .50, .75, 1.0])
KLIST = (0, 1, 2, 3, 5, 10)


def keys_from_strings(x):
    _, z = np.unique(np.asarray(x, dtype=str), return_inverse=True)
    return z


def cell_stats(Y, fam, key, F, K):
    """Per-model means, with family fallback, estimated on fit rows only."""
    global_mean = Y.mean(0)
    fm = np.vstack([Y[fam == g].mean(0) if (fam == g).any() else global_mean for g in range(F)])
    cm = np.full((K, Y.shape[1]), np.nan)
    for k in np.unique(key):
        m = key == k
        if m.sum() >= MIN_CELL:
            cm[k] = Y[m].mean(0)
    return fm, cm


def apply_stats(fam, key, fm, cm):
    z = cm[key].copy()
    bad = ~np.isfinite(z).all(1)
    z[bad] = fm[fam[bad]]
    return z


def fit_residual_head(Xfit, Yfit, Xout):
    sc = StandardScaler().fit(Xfit)
    model = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(sc.transform(Xfit), Yfit)
    return model.predict(sc.transform(Xout)).astype(np.float32), np.asarray(model.alpha_).tolist()


def aggregate(v, fam):
    return {"per_query": float(np.mean(v)),
            "per_family_equal_weight": float(np.mean([v[fam == g].mean() for g in np.unique(fam)]))}


def choice_utility(U, ch):
    return U[np.arange(len(ch)), ch]


def rank_alternatives(score, base_choice):
    s = score.copy()
    s[np.arange(len(s)), base_choice] = -np.inf
    return np.argsort(-s, axis=1)


def random_shortlist_utility(U, base_choice, k, seed, repeats=30):
    if k == 0:
        return choice_utility(U, base_choice)
    rng = np.random.default_rng(seed)
    n, M = U.shape
    vals = []
    for _ in range(repeats):
        noise = rng.random((n, M)); noise[np.arange(n), base_choice] = np.inf
        alt = np.argsort(noise, axis=1)[:, :k]
        cand = np.column_stack([base_choice, alt])
        vals.append(np.take_along_axis(U, cand, axis=1).max(1))
    return np.mean(vals, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="gte", choices=("bow", "gte", "gte_l21", "reader_l21"))
    a = ap.parse_args()
    t0 = time.time()
    panel = load_panel(PANEL)
    X = load_rep(a.rep, panel)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0)
    c = np.nan_to_num(panel.costs.astype(np.float64), nan=0.0)
    fam = panel.dataset_index().astype(int)
    topics = np.asarray(topic_str(panel), dtype=str)
    key = keys_from_strings(topics)
    F, K, M = len(np.unique(fam)), len(np.unique(key)), panel.m
    runs = []

    for seed in SEEDS:
        outer_tr, te = split(panel.n, fam, seed)
        fit_rel, cal_rel = split(len(outer_tr), fam[outer_tr], 1000 + seed)
        fit, cal = outer_tr[fit_rel], outer_tr[cal_rel]

        # Baselines used by the paper get all outer-training rows. Prediction heads
        # and calibration use disjoint fit/calibration rows.
        rf_all, rc_all = cell_stats(r[outer_tr], fam[outer_tr], key[outer_tr], F, K)
        cf_all, cc_all = cell_stats(c[outer_tr], fam[outer_tr], key[outer_tr], F, K)
        base_r_te = apply_stats(fam[te], key[te], rf_all, rc_all)
        base_c_te = apply_stats(fam[te], key[te], cf_all, cc_all)

        rf, rc = cell_stats(r[fit], fam[fit], key[fit], F, K)
        cf, cc = cell_stats(c[fit], fam[fit], key[fit], F, K)
        br_fit = apply_stats(fam[fit], key[fit], rf, rc)
        bc_fit = apply_stats(fam[fit], key[fit], cf, cc)
        br_cal = apply_stats(fam[cal], key[cal], rf, rc)
        bc_cal = apply_stats(fam[cal], key[cal], cf, cc)
        br_te = apply_stats(fam[te], key[te], rf, rc)
        bc_te = apply_stats(fam[te], key[te], cf, cc)

        Yr = r[fit] - br_fit
        Yc = c[fit] - bc_fit
        pred_r, ar = fit_residual_head(X[fit], Yr, np.vstack([X[cal], X[te]]))
        pred_c, ac = fit_residual_head(X[fit], Yc, np.vstack([X[cal], X[te]]))
        nr = len(cal)
        pr_cal, pr_te = br_cal + pred_r[:nr], br_te + pred_r[nr:]
        pc_cal, pc_te = bc_cal + pred_c[:nr], bc_te + pred_c[nr:]

        by_lam = {}
        for lam in LAMBDAS:
            Ucal, Ute = r[cal] - lam * c[cal], r[te] - lam * c[te]
            ph_cal, ph_te = pr_cal - lam * pc_cal, pr_te - lam * pc_te
            # Exact paper lookup, fitted on all outer-training rows.
            base_score_te = base_r_te - lam * base_c_te
            base_choice_te = base_score_te.argmax(1)
            # Calibration lookup uses fit rows only, to remain independent of cal outcomes.
            base_choice_cal = (br_cal - lam * bc_cal).argmax(1)
            pred_choice_cal, pred_choice_te = ph_cal.argmax(1), ph_te.argmax(1)
            adv_cal = ph_cal.max(1) - ph_cal[np.arange(len(cal)), base_choice_cal]
            adv_te = ph_te.max(1) - ph_te[np.arange(len(te)), base_choice_te]

            # Pick a switching fraction on calibration; top-q by predicted advantage.
            cal_rows = np.argsort(-adv_cal)
            q_scores = []
            for q in QGRID:
                ch = base_choice_cal.copy(); nsw = int(round(q * len(cal)))
                if nsw: ch[cal_rows[:nsw]] = pred_choice_cal[cal_rows[:nsw]]
                q_scores.append(choice_utility(Ucal, ch).mean())
            best_i = int(np.argmax(q_scores)); q = float(QGRID[best_i])
            ch = base_choice_te.copy(); nsw = int(round(q * len(te)))
            if nsw:
                rows = np.argsort(-adv_te)[:nsw]; ch[rows] = pred_choice_te[rows]

            ub = choice_utility(Ute, base_choice_te)
            ud = choice_utility(Ute, pred_choice_te)
            uh = choice_utility(Ute, ch)
            uo = Ute.max(1)
            denom = max(float(uo.mean() - ub.mean()), 1e-12)
            result = {
                "selected_switch_fraction": q,
                "cell_lookup": aggregate(ub, fam[te]),
                "direct_argmax": aggregate(ud, fam[te]),
                "calibrated_switch": aggregate(uh, fam[te]),
                "oracle": aggregate(uo, fam[te]),
                "fraction_gap_recovered_direct": float((ud.mean() - ub.mean()) / denom),
                "fraction_gap_recovered_switch": float((uh.mean() - ub.mean()) / denom),
                "shortlist": {},
            }
            rank = rank_alternatives(ph_te, base_choice_te)
            for k in KLIST:
                if k == 0:
                    up = ub
                else:
                    cand = np.column_stack([base_choice_te, rank[:, :k]])
                    up = np.take_along_axis(Ute, cand, axis=1).max(1)
                ur = random_shortlist_utility(Ute, base_choice_te, k, seed * 100 + int(lam * 10) + k)
                result["shortlist"][str(k)] = {
                    "predicted": aggregate(up, fam[te]),
                    "random": aggregate(ur, fam[te]),
                    "predicted_fraction_raw_gap": float((up.mean() - ub.mean()) / denom),
                    "random_fraction_raw_gap": float((ur.mean() - ub.mean()) / denom),
                    "excess_over_random": float(up.mean() - ur.mean()),
                }
            by_lam[str(lam)] = result
        runs.append({"seed": seed, "n_fit": len(fit), "n_cal": len(cal), "n_test": len(te),
                     "ridge_alpha_reward": ar, "ridge_alpha_cost": ac, "by_lambda": by_lam})
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)

    def vals(lam, path):
        out = []
        for run in runs:
            z = run["by_lambda"][str(lam)]
            for p in path: z = z[p]
            out.append(float(z))
        return out

    summary = {}
    for lam in LAMBDAS:
        methods = {}
        for method in ("cell_lookup", "direct_argmax", "calibrated_switch", "oracle"):
            methods[method] = {mode: float(np.mean(vals(lam, [method, mode]))) for mode in ("per_query", "per_family_equal_weight")}
        summary[str(lam)] = {
            "methods": methods,
            "switch_fraction": float(np.mean(vals(lam, ["selected_switch_fraction"]))),
            "fraction_gap_recovered_direct": float(np.mean(vals(lam, ["fraction_gap_recovered_direct"]))),
            "fraction_gap_recovered_switch": float(np.mean(vals(lam, ["fraction_gap_recovered_switch"]))),
            "shortlist": {str(k): {metric: float(np.mean(vals(lam, ["shortlist", str(k), metric])))
                                      for metric in ("predicted_fraction_raw_gap", "random_fraction_raw_gap", "excess_over_random")}
                          for k in KLIST},
        }

    result = base(panel, {"experiment": EXP, "representation": a.rep, "representation_label": REP_LABEL[a.rep],
                          "min_cell": MIN_CELL, "switch_fraction_grid": QGRID.tolist(), "shortlist_k": list(KLIST),
                          "protocol": "outer 70/30; inner fit/calibration; cell-residual multioutput ridge; test untouched",
                          "runs": runs, "summary": summary, "seconds": time.time() - t0})
    md = [f"# {EXP} — {REP_LABEL[a.rep]} — {panel.name}", "",
          "Family-topic lookup is the default. A ridge head predicts per-model reward and cost residuals. The switch fraction is chosen on disjoint calibration rows. The shortlist oracle is deliberately optimistic and is compared with a random shortlist of equal size.", "",
          "## End-to-end utility (mean of five splits, per query)", "",
          "lambda | cell | direct | calibrated switch | oracle | gap recovered (switch) | switch fraction", "---|---:|---:|---:|---:|---:|---:"]
    for lam in REPORT:
        s = summary[str(lam)]; m = s["methods"]
        md.append(f"{lam:g} | {m['cell_lookup']['per_query']:.4f} | {m['direct_argmax']['per_query']:.4f} | {m['calibrated_switch']['per_query']:.4f} | {m['oracle']['per_query']:.4f} | {s['fraction_gap_recovered_switch']:.1%} | {s['switch_fraction']:.1%}")
    md += ["", "## Optimistic shortlist: fraction of raw oracle gap captured", "",
           "lambda | k extra | predicted shortlist | random shortlist | utility excess over random", "---|---:|---:|---:|---:"]
    for lam in REPORT:
        for k in KLIST:
            z = summary[str(lam)]["shortlist"][str(k)]
            md.append(f"{lam:g} | {k} | {z['predicted_fraction_raw_gap']:.1%} | {z['random_fraction_raw_gap']:.1%} | {z['excess_over_random']:+.4f}")
    md += ["", "## Equal-family utility", "", "lambda | cell | direct | calibrated switch | oracle", "---|---:|---:|---:|---:"]
    for lam in REPORT:
        m = summary[str(lam)]["methods"]
        md.append(f"{lam:g} | {m['cell_lookup']['per_family_equal_weight']:.4f} | {m['direct_argmax']['per_family_equal_weight']:.4f} | {m['calibrated_switch']['per_family_equal_weight']:.4f} | {m['oracle']['per_family_equal_weight']:.4f}")
    save(out_dir(EXP + "_" + a.rep), result, md)


if __name__ == "__main__":
    main()
