"""exp96: diagnostics for recoverability of the oracle-minus-cell gap.

This script keeps the exact outer train/test and inner fit/calibration protocol
of exp90--92.  It adds the quantities needed to distinguish two failures:
identifying a query on which the family-topic choice should change, and choosing
the correct destination model once a change is attempted.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb.data import load_panel  # noqa: E402
from rb.panelctx import PANEL, topic_str  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from common import REP_LABEL, base, load_rep, out_dir, save  # noqa: E402

EXP = "exp96_switch_diagnostics"
MIN_CELL = 50
QGRID = np.array([0, .01, .02, .05, .10, .20, .35, .50, .75, 1.0])
EPS = 1e-12


def stats(Y, fam, key, F, K):
    glob = Y.mean(0)
    fm = np.vstack([Y[fam == g].mean(0) if (fam == g).any() else glob
                    for g in range(F)])
    cm = np.full((K, Y.shape[1]), np.nan)
    for k in np.unique(key):
        mask = key == k
        if mask.sum() >= MIN_CELL:
            cm[k] = Y[mask].mean(0)
    return fm, cm


def apply(fam, key, fm, cm):
    out = cm[key].copy()
    bad = ~np.isfinite(out).all(1)
    out[bad] = fm[fam[bad]]
    return out


def fit_ridge(X, Y, Xout):
    sc = StandardScaler().fit(X)
    model = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)
    pred = model.fit(sc.transform(X), Y).predict(sc.transform(Xout))
    return pred.astype(np.float32), {"alpha": np.asarray(model.alpha_).tolist()}


def fit_nonlinear(method, X, Y, Xout, seed):
    sc = StandardScaler().fit(X)
    A, B = sc.transform(X), sc.transform(Xout)
    if method == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-2,
                             early_stopping=True, validation_fraction=.1,
                             n_iter_no_change=12, max_iter=250,
                             random_state=seed)
        pred = model.fit(A, Y).predict(B)
        return pred.astype(np.float32), {"n_iter": int(model.n_iter_)}
    pca = PCA(128, svd_solver="randomized", random_state=seed).fit(A)
    A, B = pca.transform(A), pca.transform(B)
    pred = np.zeros((len(B), Y.shape[1]), np.float32)
    nit = []
    for j in range(Y.shape[1]):
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=.05, max_leaf_nodes=31,
            min_samples_leaf=20, early_stopping=True, random_state=seed)
        pred[:, j] = model.fit(A, Y[:, j]).predict(B)
        nit.append(int(model.n_iter_))
    return pred, {"pca_components": 128, "n_iter": nit}


def chosen(U, choice):
    return U[np.arange(len(choice)), choice]


def stratified_bootstrap(delta, fam, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(fam == g) for g in np.unique(fam)]
    pq, ef = [], []
    for _ in range(B):
        samples = [rng.choice(g, len(g), replace=True) for g in groups]
        ix = np.concatenate(samples)
        pq.append(delta[ix].mean())
        ef.append(np.mean([delta[x].mean() for x in samples]))
    return {
        "point": float(delta.mean()),
        "ci95": np.percentile(pq, [2.5, 97.5]).tolist(),
        "equal_family": float(np.mean([delta[g].mean() for g in groups])),
        "equal_family_ci95": np.percentile(ef, [2.5, 97.5]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("ridge", "hgb", "mlp"), required=True)
    ap.add_argument("--rep", choices=("gte", "gte_l21", "reader_l21"), default="gte")
    ap.add_argument("--lam", type=float, choices=(0.0, 100.0), required=True)
    args = ap.parse_args()
    t0 = time.time()

    panel = load_panel(PANEL)
    X = load_rep(args.rep, panel)
    R = np.nan_to_num(panel.rewards.astype(float), nan=0.0)
    C = np.nan_to_num(panel.costs.astype(float), nan=0.0)
    U = R - args.lam * C
    fam = panel.dataset_index().astype(int)
    _, key = np.unique(np.asarray(topic_str(panel), str), return_inverse=True)
    F, K = len(np.unique(fam)), key.max() + 1
    runs, boots = [], {}

    for seed in SEEDS:
        outer, te = split(panel.n, fam, seed)
        fi, ca = split(len(outer), fam[outer], 1000 + seed)
        fitix, cal = outer[fi], outer[ca]

        # The tested lookup receives the full outer-training split.
        fm_all, cm_all = stats(U[outer], fam[outer], key[outer], F, K)
        base_score_te = apply(fam[te], key[te], fm_all, cm_all)
        base_choice_te = base_score_te.argmax(1)

        if args.method == "ridge":
            # Match exp90/91 exactly: predict reward and cost residuals separately.
            rf, rc = stats(R[fitix], fam[fitix], key[fitix], F, K)
            cf, cc = stats(C[fitix], fam[fitix], key[fitix], F, K)
            brf = apply(fam[fitix], key[fitix], rf, rc)
            bcf = apply(fam[fitix], key[fitix], cf, cc)
            brc = apply(fam[cal], key[cal], rf, rc)
            bcc = apply(fam[cal], key[cal], cf, cc)
            brt = apply(fam[te], key[te], rf, rc)
            bct = apply(fam[te], key[te], cf, cc)
            xout = np.vstack([X[cal], X[te]])
            pr, mr = fit_ridge(X[fitix], R[fitix] - brf, xout)
            pc, mc = fit_ridge(X[fitix], C[fitix] - bcf, xout)
            nc = len(cal)
            score_cal = brc + pr[:nc] - args.lam * (bcc + pc[:nc])
            score_te = brt + pr[nc:] - args.lam * (bct + pc[nc:])
            base_choice_cal = (brc - args.lam * bcc).argmax(1)
            meta = {"reward": mr, "cost": mc}
        else:
            fm, cm = stats(U[fitix], fam[fitix], key[fitix], F, K)
            bf = apply(fam[fitix], key[fitix], fm, cm)
            bc = apply(fam[cal], key[cal], fm, cm)
            bt = apply(fam[te], key[te], fm, cm)
            pred, meta = fit_nonlinear(args.method, X[fitix], U[fitix] - bf,
                                       np.vstack([X[cal], X[te]]), seed)
            nc = len(cal)
            score_cal, score_te = bc + pred[:nc], bt + pred[nc:]
            base_choice_cal = bc.argmax(1)

        pred_choice_cal = score_cal.argmax(1)
        pred_choice_te = score_te.argmax(1)
        adv_cal = score_cal.max(1) - score_cal[np.arange(len(cal)), base_choice_cal]
        adv_te = score_te.max(1) - score_te[np.arange(len(te)), base_choice_te]

        order_cal = np.argsort(-adv_cal)
        cal_scores = []
        for q in QGRID:
            route = base_choice_cal.copy()
            n = int(round(q * len(cal)))
            if n:
                route[order_cal[:n]] = pred_choice_cal[order_cal[:n]]
            cal_scores.append(chosen(U[cal], route).mean())
        q = float(QGRID[int(np.argmax(cal_scores))])

        route = base_choice_te.copy()
        n = int(round(q * len(te)))
        if n:
            route[np.argsort(-adv_te)[:n]] = pred_choice_te[np.argsort(-adv_te)[:n]]
        switch = route != base_choice_te

        Ute = U[te]
        ub, uh, ud, uo = (chosen(Ute, z) for z in
                          (base_choice_te, route, pred_choice_te, Ute.argmax(1)))
        delta = uh - ub
        denom = max(float(uo.mean() - ub.mean()), EPS)

        # Same switched rows, but destinations chosen without the text head.
        cell_rank = np.argsort(-base_score_te, axis=1)
        second = np.array([next(m for m in cell_rank[i] if m != base_choice_te[i])
                           for i in range(len(te))])
        cell2_delta = np.zeros(len(te))
        cell2_delta[switch] = Ute[np.flatnonzero(switch), second[switch]] - ub[switch]

        random_dest_delta = np.zeros(len(te))
        for i in np.flatnonzero(switch):
            keep = np.arange(Ute.shape[1]) != base_choice_te[i]
            random_dest_delta[i] = Ute[i, keep].mean() - ub[i]

        # Same predicted destinations, but choose equally many eligible queries at random.
        eligible = pred_choice_te != base_choice_te
        p = float(switch.sum() / max(eligible.sum(), 1))
        random_query_delta = np.zeros(len(te))
        random_query_delta[eligible] = p * (chosen(Ute, pred_choice_te)[eligible] - ub[eligible])

        if switch.any():
            ds = delta[switch]
            beneficial = float(np.mean(ds > EPS))
            harmful = float(np.mean(ds < -EPS))
            tie = float(np.mean(np.abs(ds) <= EPS))
            conditional_gain = float(ds.mean())
            order_u = np.argsort(-Ute, axis=1)
            unique = Ute[np.arange(len(te)), order_u[:, 0]] > Ute[np.arange(len(te)), order_u[:, 1]] + EPS
            unique_sw = switch & unique
            exact_unique = (float(np.mean(route[unique_sw] == order_u[unique_sw, 0]))
                            if unique_sw.any() else float("nan"))
            opportunity = Ute.max(1) > ub + EPS
            opportunity_selected = float(np.mean(opportunity[switch]))
            opportunity_eligible = float(np.mean(opportunity[eligible])) if eligible.any() else float("nan")
            potential = (Ute.max(1) - ub)[switch & opportunity].sum()
            destination_capture = float(delta[switch & opportunity].sum() / potential) if potential > EPS else float("nan")
        else:
            beneficial = harmful = tie = conditional_gain = exact_unique = float("nan")
            opportunity_selected = opportunity_eligible = destination_capture = float("nan")

        metrics = {
            "selected_switch_fraction": q,
            "actual_switch_fraction": float(switch.mean()),
            "cell_utility": float(ub.mean()),
            "direct_utility": float(ud.mean()),
            "switch_utility": float(uh.mean()),
            "oracle_utility": float(uo.mean()),
            "utility_gain": float(delta.mean()),
            "fraction_gap_recovered": float(delta.mean() / denom),
            "conditional_gain_switched": conditional_gain,
            "beneficial_switch_fraction": beneficial,
            "harmful_switch_fraction": harmful,
            "tie_switch_fraction": tie,
            "exact_unique_oracle_destination": exact_unique,
            "oracle_opportunity_fraction_selected": opportunity_selected,
            "oracle_opportunity_fraction_eligible": opportunity_eligible,
            "destination_capture_on_opportunities": destination_capture,
            "control_same_rows_cell_second_gain": float(cell2_delta.mean()),
            "control_same_rows_random_destination_gain": float(random_dest_delta.mean()),
            "control_random_rows_predicted_destination_gain": float(random_query_delta.mean()),
        }
        if seed == SEEDS[0]:
            for name, vec in {
                "switch_minus_cell": delta,
                "same_rows_cell_second_minus_cell": cell2_delta,
                "same_rows_random_destination_minus_cell": random_dest_delta,
                "random_rows_predicted_destination_minus_cell": random_query_delta,
            }.items():
                boots[name] = stratified_bootstrap(vec, fam[te])
        runs.append({"seed": int(seed), "n_fit": len(fitix), "n_cal": len(cal),
                     "n_test": len(te), "fit_meta": meta, **metrics})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    names = [k for k in runs[0] if k not in {"seed", "n_fit", "n_cal", "n_test", "fit_meta"}]
    summary = {}
    for name in names:
        vals = np.asarray([z[name] for z in runs], float)
        summary[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    result = base(panel, {
        "experiment": EXP, "method": args.method, "representation": args.rep,
        "representation_label": REP_LABEL[args.rep], "lambda": args.lam,
        "protocol": "five outer 70/30 splits; disjoint inner fit/calibration; switch budget chosen only on calibration; untouched test evaluation",
        "controls": {
            "same_rows_cell_second": "same actually switched rows, destination is lookup's second-ranked model",
            "same_rows_random_destination": "same actually switched rows, expected utility of a uniformly random non-lookup model",
            "random_rows_predicted_destination": "same number of eligible switches in expectation, rows random, head's predicted destination",
        },
        "runs": runs, "summary": summary, "bootstrap_split0": boots,
        "seconds": time.time() - t0,
    })
    md = [
        f"# {EXP}: {args.method}, {REP_LABEL[args.rep]}, lambda={args.lam:g}", "",
        "Mean over five family-stratified outer splits.", "",
        "metric | value", "---|---:",
    ]
    for k, v in summary.items():
        md.append(f"{k} | {v:.6f}")
    md += ["", "## Split-0 family-stratified paired bootstrap", "",
           "comparison | point | 95% CI | equal-family point | 95% CI", "---|---:|---:|---:|---:"]
    for name, z in boots.items():
        md.append(f"{name} | {z['point']:+.6f} | [{z['ci95'][0]:+.6f}, {z['ci95'][1]:+.6f}] | {z['equal_family']:+.6f} | [{z['equal_family_ci95'][0]:+.6f}, {z['equal_family_ci95'][1]:+.6f}]")
    save(out_dir(f"{EXP}_{args.rep}_{args.method}_lam{args.lam:g}"), result, md)


if __name__ == "__main__":
    main()
