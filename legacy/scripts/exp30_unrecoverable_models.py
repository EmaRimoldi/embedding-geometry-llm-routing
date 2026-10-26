"""exp30: is the oracle's advantage made of per-query model successes that no text-based router can recover?
A. Oracle pool ablation: oracle utility with the full pool, without the panel's cheap model (CHEAP_MODEL), and without the k cheapest models; share of the (oracle - cell lookup) gap closed.
   Also: share of the gap attributable to queries where the oracle picks each model.
B. Model-specific predictability: for each model m, y_m = 1[r_m >= 0.5]. AUC of predicting y_m on test rows from (i) ridge on the representation, (ii) kNN-100 vote, (iii) ridge on bag-of-words,
   (iv) logistic on the other models' outcomes (non-text reference). Unconditional AUC and WITHIN-DIFFICULTY AUC (computed inside strata of the count of models
   that solve the query, averaged weighted by stratum size) -- the latter is the model-specific part with general easiness factored out.
C. GSM8K sampling-noise ceiling: rewards are fractions of 4 samples; taking the reward as the success probability p_i, the best possible accuracy for predicting
   y = 1[r >= 0.5] is mean(max(p_i, 1 - p_i)); share of coin-flip queries (0 < r < 1).
D. Oracle without the models whose within-difficulty AUC (ridge on the representation) is below thresholds; gap closed.
Writes results/oracle_gap/exp30_unrecoverable_models/."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, N_FAM, TOPIC_FAMILIES, CHEAP_MODEL, topic_str, small_families, lengths_features, core_rep  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import split  # noqa: E402
import os
SEEDS = tuple(int(v) for v in os.environ.get("E4_SEEDS", "0,1,2,3,4").split(","))    # E4_SEEDS overrides the default 5 seeds (hedge runs)
from rb.reps import find_rep, load_rep  # noqa: E402
OUT = RESULTS_ROOT / "oracle_gap" / ("exp30_unrecoverable_models" + os.environ.get("E4_OUT_SUFFIX", "")); LAMS = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0); MIN_CELL = 50; AGGS = ("per_query", "per_family_equal_weight")


def strat_auc(y, s, strata):
    """AUC within strata (count of models solving the query), weighted by stratum size; strata with one class are skipped."""
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        if m.sum() < 20 or y[m].min() == y[m].max(): continue
        num += roc_auc_score(y[m], s[m]) * m.sum(); den += m.sum()
    return float(num / den) if den else float("nan")


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); ds = panel.dataset_ids
    models = [str(m).split("/")[-1] for m in panel.models]; M = len(models); mi = models.index(CHEAP_MODEL)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0); Y = (r >= SUCCESS_THRESHOLD).astype(np.float64); ns = Y.sum(1)
    x = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32); u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    bow = load_rep(find_rep(PANEL, "computed/hash-bow@prompt"), n, panel.keep).astype(np.float32)
    cell_str = topic_str(panel); _, cell = np.unique(cell_str, return_inverse=True)
    mean_cost = c.mean(0); cheap_order = list(np.argsort(mean_cost))                       # cheapest first
    def agg(v, fte, mode): return float(v.mean()) if mode == "per_query" else float(np.mean([v[fte == g].mean() for g in range(F) if (fte == g).any()]))
    # ---------------- B (and neighbours for A): per split
    aucs = {m: {k: {"uncond": [], "within": [], "within_fc": []} for k in ("ridge_gte", "knn100_vote", "ridge_bow", "others_outcomes", "boosting_gte")} for m in models}
    E = {"bins": [], "pairs_near": [], "pairs_random": []}
    A = {L: {mode: {"cell_lookup": [], "oracle_full": [], "oracle_no_mistral": [], **{f"oracle_minus_{k}_cheapest": [] for k in (2, 3, 4, 5)}, "gap_share_by_oracle_pick": []} for mode in AGGS} for L in [f"{l:g}" for l in LAMS]}
    seeds_done = []
    def write_results():
        # ---------------- C: GSM8K sampling ceiling
        g = ds == "grade-school-math"; C = {}
        for m_ in range(M):
            rv = r[g, m_]; C[models[m_]] = {"share_coin_flip(0<r<1)": float(((rv > 0) & (rv < 1)).mean()), "share_r_in_[0.25,0.75]": float(((rv >= 0.25) & (rv <= 0.75)).mean()),
                                            "best_possible_accuracy_if_p=r": float(np.maximum(rv, 1 - rv).mean()), "solve_rate": float((rv >= 0.5).mean())}
        # ---------------- D: oracle without low-predictability models
        within_gte = {m: float(np.nanmean(aucs[m]["ridge_gte"]["within_fc"])) for m in models}
        D = {}
        for thr in (0.55, 0.60, 0.65):
            drop = [m for m in models if within_gte[m] < thr]; keep = [j for j in range(M) if models[j] not in drop]; D[str(thr)] = {"dropped": drop, "oracle": {}}
            if not keep: D[str(thr)]["note"] = "every model is below the threshold; no oracle"; continue
            for lam in LAMS:
                L = f"{lam:g}"; Ul = r - lam * c; D[str(thr)]["oracle"][L] = {}
                for seed in SEEDS:
                    tr, te = split(n, fam, seed); te = np.sort(te); fte = fam[te]; Ute = Ul[te]
                    for mode in AGGS: D[str(thr)]["oracle"][L].setdefault(mode, []).append(agg(Ute[:, keep].max(1), fte, mode))
                for mode in AGGS: D[str(thr)]["oracle"][L][mode] = float(np.mean(D[str(thr)]["oracle"][L][mode]))
        OUT.mkdir(parents=True, exist_ok=True); ms = lambda v: float(np.mean(v))
        Aout = {L: {mode: {k: (ms(v) if k != "gap_share_by_oracle_pick" else np.mean(np.array(v), axis=0).tolist()) for k, v in d.items()} for mode, d in dd.items()} for L, dd in A.items()}
        out = {**panel.provenance(), "seeds_done": seeds_done, "experiment": "exp30_unrecoverable_models", "representation": str(REP), "models": models, "mean_cost": mean_cost.tolist(), "cheapest_order": [models[j] for j in cheap_order],
               "A_oracle_ablation": Aout, "B_auc": {m: {k: {"uncond": ms(v["uncond"]), "within_difficulty": float(np.nanmean(v["within"])), "within_family_and_difficulty": float(np.nanmean(v["within_fc"]))} for k, v in d.items() if v["uncond"]} for m, d in aucs.items()}, "C_gsm8k_ceiling": C, "D_oracle_without_low_predictability": D, "E_near_duplicate_consistency_mistral": E["bins"]}
        (OUT / "result.json").write_text(json.dumps(out, indent=1))
        md = ["# exp30 — the oracle's advantage and whether it is recoverable", "", "Splits covered by this file: " + ", ".join(map(str, seeds_done)) + " (checkpointed after each). Means over these splits. Cell lookup = (family, topic) lookup with family fallback.", "",
              "## A. Oracle pool ablation (utility; gap = oracle_full − cell lookup; 'closed' = share of the gap removed by shrinking the oracle's pool)", ""]
        for mode in AGGS:
            md += [f"### {mode}", "", f"λ | cell lookup | oracle ({M} models) | gap | oracle without {CHEAP_MODEL} (gap closed) | without 2 cheapest (closed) | without 3 cheapest | without 4 cheapest | without 5 cheapest", "---|---|---|---|---|---|---|---|---"]
            for L in Aout:
                d = Aout[L][mode]; gap = d["oracle_full"] - d["cell_lookup"]; f_ = lambda k: f"{d[k]:.4f} ({(d['oracle_full'] - d[k]) / gap:.2f})"
                md.append(f"{L} | {d['cell_lookup']:.4f} | {d['oracle_full']:.4f} | {gap:.4f} | {f_('oracle_no_mistral')} | {f_('oracle_minus_2_cheapest')} | {f_('oracle_minus_3_cheapest')} | {f_('oracle_minus_4_cheapest')} | {f_('oracle_minus_5_cheapest')}")
            md += ["", "Share of the gap attributable to queries on which the oracle picks each model:", "", "λ | " + " | ".join(models), "---|" + "---|" * M]
            md += [f"{L} | " + " | ".join(f"{v:.2f}" for v in Aout[L][mode]["gap_share_by_oracle_pick"]) for L in Aout] + [""]
        md += [f"Cheapest models in order: {', '.join(out['cheapest_order'])}.", "", "## B. Predictability of 'model m solves the query' (AUC on test rows; 'within difficulty' = AUC inside strata of the count of models that solve the query)", "",
               f"model | solve rate | ridge on {REP}: uncond. / within difficulty / within family x difficulty | kNN-100 vote | ridge on bag-of-words | other {M - 1} models' outcomes", "---|---|---|---|---|---"]
        for m_ in range(M):
            d = out["B_auc"][models[m_]]; md.append(f"{models[m_]} | {Y[:, m_].mean():.2f} | " + " | ".join(f"{d[k]['uncond']:.3f} / {d[k]['within_difficulty']:.3f} / {d[k]['within_family_and_difficulty']:.3f}" for k in ("ridge_gte", "knn100_vote", "ridge_bow", "others_outcomes")))
        b_ = out["B_auc"][CHEAP_MODEL]["boosting_gte"]; md += ["", f"{CHEAP_MODEL}, gradient boosting on {REP} (nonlinear check): unconditional AUC {b_['uncond']:.3f}, within difficulty {b_['within_difficulty']:.3f}, within family x difficulty {b_['within_family_and_difficulty']:.3f}."]
        md += ["", f"## E. Consistency of {CHEAP_MODEL}'s outcome between a test query and its nearest training row (same family, same difficulty count) vs a random training row of the same family and count", "",
               "cosine of the nearest row | pairs (5 splits) | P(same outcome) nearest | P(same outcome) random | share with count in 1..10", "---|---|---|---|---"]
        agg_bins = {}
        for lo, hi, n_, a, b, c_ in E["bins"]: agg_bins.setdefault((lo, hi), []).append((n_, a, b, c_))
        for (lo, hi), rows in sorted(agg_bins.items(), reverse=True):
            N = sum(r_[0] for r_ in rows); md.append(f"[{lo:.2f}, {min(hi, 1.0):.2f}) | {N} | {sum(r_[0] * r_[1] for r_ in rows) / N:.3f} | {sum(r_[0] * r_[2] for r_ in rows) / N:.3f} | {sum(r_[0] * r_[3] for r_ in rows) / N:.2f}")
        md += ["", "## C. [VOID] GSM8K rewards are a partial-credit rubric (0.25 / 0.5 / 0.75 in 96% of cells), not sampling fractions; the ceiling below assumes p = r and is not valid", "", "model | solve rate | share of coin-flip queries (0 < r < 1) | share with r in [0.25, 0.75] | best possible accuracy for y = 1[r >= 0.5] if p = r", "---|---|---|---|---"]
        md += [f"{m} | {v['solve_rate']:.2f} | {v['share_coin_flip(0<r<1)']:.2f} | {v['share_r_in_[0.25,0.75]']:.2f} | {v['best_possible_accuracy_if_p=r']:.3f}" for m, v in C.items()]
        md += ["", f"## D. Oracle without the models whose within-(family x difficulty) AUC (ridge on {REP}) is below a threshold", "", "AUC used: " + ", ".join(f"{m} {within_gte[m]:.3f}" for m in models), ""]
        for thr, d in D.items():
            if "note" in d: md += [f"threshold {thr}: dropped = {d['dropped']} — {d['note']}", ""]; continue
            md += [f"threshold {thr}: dropped = {d['dropped']}", "", "λ | " + " | ".join(f"oracle ({mode}) | gap closed" for mode in AGGS), "---|---|---|---|---"]
            for L in d["oracle"]:
                row = []
                for mode in AGGS:
                    full = Aout[L][mode]["oracle_full"]; cl = Aout[L][mode]["cell_lookup"]; v = d["oracle"][L][mode]; row.append(f"{v:.4f} | {(full - v) / (full - cl):.2f}")
                md.append(f"{L} | " + " | ".join(row))
            md.append("")
        (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
        print(f"[checkpoint] results written for seeds {seeds_done}", flush=True)

    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fte = fam[te]; idx = np.arange(len(te)); strata = ns[te]; strata_fc = fte * 100 + ns[te]
        sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te]); sb = StandardScaler().fit(bow[tr]); btr, bte = sb.transform(bow[tr]), sb.transform(bow[te])
        nb = tr[NearestNeighbors(n_neighbors=100, metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te])[1]]
        for m_ in range(M):
            y_tr, y_te = Y[tr, m_], Y[te, m_]
            scores = {"ridge_gte": RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, y_tr).predict(xte), "knn100_vote": Y[nb, m_].mean(1), "ridge_bow": RidgeCV(alphas=RIDGE_ALPHAS).fit(btr, y_tr).predict(bte)}
            if m_ == mi:
                from sklearn.ensemble import HistGradientBoostingClassifier
                scores["boosting_gte"] = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=seed).fit(xtr, y_tr).predict_proba(xte)[:, 1]
            others = [j for j in range(M) if j != m_]
            scores["others_outcomes"] = LogisticRegression(C=1.0, max_iter=2000).fit(Y[tr][:, others], y_tr).predict_proba(Y[te][:, others])[:, 1]
            for k, s_ in scores.items(): aucs[models[m_]][k]["uncond"].append(float(roc_auc_score(y_te, s_))); aucs[models[m_]][k]["within"].append(strat_auc(y_te, s_, strata)); aucs[models[m_]][k]["within_fc"].append(strat_auc(y_te, s_, strata_fc))
        # ---------------- E: consistency of Mistral's outcome between a test query and its nearest training row (same family and same difficulty count),
        #                    vs a random training row of the same family and count; by cosine bin of the nearest row
        d1, i1 = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=8).fit(u[tr]).kneighbors(u[te]); cos1 = 1 - d1[:, 0]; j1 = tr[i1[:, 0]]
        same = (fam[j1] == fte) & (ns[j1] == ns[te]); rng_e = np.random.default_rng(seed)
        rnd = np.full(len(te), -1)
        for g in range(F):
            for k in np.unique(ns[te][fte == g]):
                pool = tr[(fam[tr] == g) & (ns[tr] == k)]; sel = np.flatnonzero((fte == g) & (ns[te] == k))
                if len(pool): rnd[sel] = rng_e.choice(pool, len(sel))
        ok = same & (rnd >= 0)
        for lo, hi in ((0.95, 1.01), (0.9, 0.95), (0.8, 0.9), (0.7, 0.8), (0.0, 0.7)):
            b = ok & (cos1 >= lo) & (cos1 < hi)
            if b.sum() >= 30:
                E["bins"].append([lo, hi, int(b.sum()), float((Y[te, mi][b] == Y[j1, mi][b]).mean()), float((Y[te, mi][b] == Y[rnd, mi][b]).mean()), float(((ns[te][b] >= 1) & (ns[te][b] <= 10)).mean())])
        # ---------------- A: oracle ablations
        for lam in LAMS:
            L = f"{lam:g}"; Ul = r - lam * c; Utr, Ute = Ul[tr], Ul[te]; ftab = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)]); ch = ftab[fte].copy()
            for g in np.unique(cell[te]):
                rows = cell[tr] == g
                if rows.sum() >= MIN_CELL: ch[cell[te] == g] = Utr[rows].mean(0).argmax()
            u_cell = Ute[idx, ch]; u_full = Ute.max(1); pick = Ute.argmax(1)
            pools = {"oracle_no_mistral": [j for j in range(M) if j != mi], **{f"oracle_minus_{k}_cheapest": [j for j in range(M) if j not in cheap_order[:k]] for k in (2, 3, 4, 5)}}
            for mode in AGGS:
                A[L][mode]["cell_lookup"].append(agg(u_cell, fte, mode)); A[L][mode]["oracle_full"].append(agg(u_full, fte, mode))
                for k, pool in pools.items(): A[L][mode][k].append(agg(Ute[:, pool].max(1), fte, mode))
                gap = u_full - u_cell; tot = agg(gap, fte, mode)
                A[L][mode]["gap_share_by_oracle_pick"].append([agg(np.where(pick == j, gap, 0.0), fte, mode) / tot if tot else 0.0 for j in range(M)])
        print(f"seed {seed} done", flush=True)
        seeds_done.append(seed); write_results()



if __name__ == "__main__":
    main()
