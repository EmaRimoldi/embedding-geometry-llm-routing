"""exp07 - calibrated verbosity-axis router: minimal grid, de-shrinking, fixed a-priori config.

Lessons from exp03/exp05: (i) the rank-1 cost axis is the right object; (ii) the ridge prediction of the
family coordinate gamma shrinks toward the middle (RB GSM8K 1.34 -> 0.65, MT-Bench 1.44 -> 0.04); (iii)
nested selection over large grids with a worst-family rule is noisier than the effect it selects.
This version:
  cost estimator   verb1 only (log c_m(x) = alpha_m + b_m * gamma_hat(x)), features from --features:
                   gtel = GTE-Qwen2-7B + log(1+n_in) (default), reader = Qwen2.5-7B-Instruct layer-21 last
                   token + log(1+n_in), drafts = behavioural probe (64-token greedy draft of Qwen2.5-7B-Instruct:
                   log draft length, finished, first-token entropy/margin, mean log-prob/entropy) + log(1+n_in)
  calibration      inner LOFO gives |S| pairs (true family gamma_s, mean predicted gamma over that family);
                   a line gamma_true = a + b * gamma_hat is fitted on them (no target information) and
                   applied to the outer predictions; reported with and without ("raw")
  pessimism        z in {0, 1, 2} on the calibrated coordinate, sigma = rms inner error after calibration
  reward side      soft family prior only
  selection        inner MEAN utility over source families (worst family as tie-break), grid = {raw,
                   calibrated} x {z}; the a-priori fixed config (calibrated, z = 1) is the primary row
References: safe, soft prior, cost oracle, family ceiling, oracle; verb1_batch (transductive) separate.
"""
from __future__ import annotations
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp05_verbosity_axis_router import axes, coords, fit_scalar

EXP = "exp07_calibrated_verbosity_router"; ZS = (0.0, 1.0, 2.0); CALS = ("raw", "cal"); GRID = list(itertools.product(CALS, ZS))
READER = {"routerbench-0shot": "features/routerbench-0shot/prompt/hidden/qwen2.5-7b-instruct/L21_last.npy",
          "llmrouterbench-performance-cost": "features/llmrouterbench-performance-cost/prompt/hidden/qwen2.5-7b-instruct/L21_last.npy"}


def build(D, X, Y, S, target_rows):
    x, r, c, fam = D["x"], D["r"], D["c"], D["fam"]; tr = np.flatnonzero(np.isin(fam, S))
    w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
    rt, ct = C.family_tables(r, c, fam, S); logct = np.log(ct + C.EPS); alpha, B, G = axes(logct, 1)
    v = coords(Y[tr], alpha, B)[:, 0]; ghat = fit_scalar(X[tr], v, w)(X[target_rows])[:, 0]
    P = C.fit_posterior(x, fam, S, target_rows, True, 42)
    return {"Pr": P @ rt, "log_transferred": np.log(P @ ct + C.EPS), "alpha": alpha, "b": B[0], "G": G[:, 0], "ghat": ghat, "gtrue": float(coords(Y[target_rows], alpha, B)[:, 0].mean())}


def decide(blk, lam, cal, z, a, b_, sigma, batch=False):
    g = blk["ghat"]; g = a + b_ * g if cal == "cal" else g
    lo, hi = blk["G"].min(), blk["G"].max(); span = hi - lo; g = np.clip(g, lo - .25 * span, hi + .25 * span)
    if batch: g = np.full_like(g, g.mean())
    logc = blk["alpha"][None, :] + g[:, None] * blk["b"][None, :] + z * sigma * np.abs(blk["b"])[None, :]
    return (blk["Pr"] - lam * np.exp(logc)).argmax(1)


def main():
    panel_name = sys.argv[1]; feat = sys.argv[2] if len(sys.argv) > 2 else "gtel"; t0 = time.time(); D = C.load(panel_name)
    x, r, c, fam, fn, names = D["x"], D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M = len(fn), len(names)
    ln = np.log1p(D["n_in"])[:, None].astype(np.float32)
    if feat.startswith("drafts"):
        # drafts | drafts256 | drafts_gtel | drafts256_gtel : behavioural probe stats (+ GTE)
        dname = "drafts256" if "256" in feat else "drafts"
        z = np.load(C.RESULTS / dname / panel_name / "stats.npz")
        Xb = np.column_stack([np.log1p(z["draft_len"]), z["finished"].astype(np.float32), z["first_entropy"], z["first_margin"], z["mean_logprob"], z["mean_entropy"], ln[:, 0]]).astype(np.float32)
        X = np.hstack([Xb, x]) if feat.endswith("_gtel") else Xb
    elif feat == "reader":
        h = np.load(C.E4 / READER[panel_name], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(r) else h, dtype=np.float32); X = np.hstack([h, ln])
    else:
        X = np.hstack([x, ln])
    Y = np.log(c + C.EPS)
    methods = ["safe", "soft_prior", "verb_cal_z1_fixed", "verb_selected", "verb_raw_z0", "verb_cal_z0", "verb_cal_z1_batch", "cost_oracle", "family_ceiling", "oracle"]
    per = {m: {str(l): {} for l in C.LAMS} for m in methods}; n_by = {}; selected = {}; diag = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        inner = {s: build(D, X, Y, [q for q in S if q != s], np.flatnonzero(fam == s)) for s in S}
        gh = np.array([inner[s]["ghat"].mean() for s in S]); gt = np.array([inner[s]["gtrue"] for s in S])
        A = np.c_[np.ones(len(S)), gh]; a, b_ = np.linalg.lstsq(A, gt, rcond=None)[0]
        sig_raw = float(np.sqrt(np.mean((gh - gt) ** 2))); sig_cal = float(np.sqrt(np.mean((a + b_ * gh - gt) ** 2)))
        outer = build(D, X, Y, S, te); rt, ct = C.family_tables(r, c, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0)
        diag[fn[g]] = {"calibration_a_b": [float(a), float(b_)], "sigma_raw": sig_raw, "sigma_cal": sig_cal, "gamma_true": outer["gtrue"], "gamma_hat_raw": float(outer["ghat"].mean()), "gamma_hat_cal": float(a + b_ * outer["ghat"].mean()),
                       "inner_pairs": [[fn[s], float(gt[i]), float(gh[i])] for i, s in enumerate(S)]}
        selected[fn[g]] = {}
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te)); score = {}
            for cal, z in GRID:
                vals = []
                for s in S:
                    rows = np.flatnonzero(fam == s); Us = r[rows] - lam * c[rows]
                    # inner calibration for inner fold s must itself exclude s: reuse (a, b_) fitted on all S pairs (leave-in for the inner
                    # score only; the OUTER application uses only source families) — documented approximation
                    vals.append(Us[np.arange(len(rows)), decide(inner[s], lam, cal, z, a, b_, sig_cal if cal == "cal" else sig_raw)].mean())
                score[(cal, z)] = (float(np.mean(vals)), float(np.min(vals)))
            best = max(GRID, key=lambda k: score[k]); selected[fn[g]][L] = {"config": list(best), "inner_mean_worst": score[best]}
            ch = {"safe": np.full(len(te), int((rt - lam * ct).mean(0).argmax())), "soft_prior": (outer["Pr"] - lam * np.exp(outer["log_transferred"])).argmax(1),
                  "verb_cal_z1_fixed": decide(outer, lam, "cal", 1.0, a, b_, sig_cal), "verb_selected": decide(outer, lam, best[0], best[1], a, b_, sig_cal if best[0] == "cal" else sig_raw),
                  "verb_raw_z0": decide(outer, lam, "raw", 0.0, a, b_, sig_raw), "verb_cal_z0": decide(outer, lam, "cal", 0.0, a, b_, sig_cal),
                  "verb_cal_z1_batch": decide(outer, lam, "cal", 1.0, a, b_, sig_cal, batch=True),
                  "cost_oracle": (outer["Pr"] - lam * cbar[None, :]).argmax(1), "family_ceiling": np.full(len(te), int((rbar - lam * cbar).argmax())), "oracle": U.argmax(1)}
            for m in methods: per[m][L][fn[g]] = float(U[ii, ch[m]].mean())
        d = diag[fn[g]]
        print(f"{fn[g]:18s} gamma true {d['gamma_true']:+.2f} raw {d['gamma_hat_raw']:+.2f} cal {d['gamma_hat_cal']:+.2f} (a={a:+.2f} b={b_:.2f} sig {sig_raw:.2f}->{sig_cal:.2f}) | lam100 soft {per['soft_prior']['100.0'][fn[g]]:.3f} fixed {per['verb_cal_z1_fixed']['100.0'][fn[g]]:.3f} sel {per['verb_selected']['100.0'][fn[g]]:.3f}{selected[fn[g]]['100.0']['config']} raw0 {per['verb_raw_z0']['100.0'][fn[g]]:.3f} cal0 {per['verb_cal_z0']['100.0'][fn[g]]:.3f} c-oracle {per['cost_oracle']['100.0'][fn[g]]:.3f} ceil {per['family_ceiling']['100.0'][fn[g]]:.3f} | lam50 soft {per['soft_prior']['50.0'][fn[g]]:.3f} fixed {per['verb_cal_z1_fixed']['50.0'][fn[g]]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in methods}
    def wl(a_, L): return {"wins": int(sum(per[a_][L][f] > per["soft_prior"][L][f] + 1e-9 for f in fn)), "losses": int(sum(per[a_][L][f] < per["soft_prior"][L][f] - 1e-9 for f in fn)), "worst_delta": float(min(per[a_][L][f] - per["soft_prior"][L][f] for f in fn))}
    wtl = {L: {m: wl(m, L) for m in ("verb_cal_z1_fixed", "verb_selected", "verb_cal_z0")} for L in agg["soft_prior"]}
    res = C.provenance(D); res.update({"experiment": EXP, "features": feat, "per_family": per, "aggregate_qw_macro": agg, "selected": selected, "diagnostics": diag, "vs_soft_prior": wtl})
    md = [f"# {EXP} ({panel_name}, features = {feat})", "", "Strict held-out family. Cost = rank-1 verbosity axis with inner-LOFO de-shrinking calibration; primary row = a-priori fixed (calibrated, z=1); 'selected' = inner-mean choice over {raw, cal} x z. Query-weighted [macro].", "",
          "lambda | " + " | ".join(methods) + " | fixed vs soft W/L (worst) | selected vs soft W/L", "---|" + "---|" * (len(methods) + 2)]
    for lam in C.LAMS:
        L = str(lam); w1, w2 = wtl[L]["verb_cal_z1_fixed"], wtl[L]["verb_selected"]
        md.append(f"{lam:g} | " + " | ".join(f"{agg[m][L][0]:.3f} [{agg[m][L][1]:.3f}]" for m in methods) + f" | {w1['wins']}/{w1['losses']} ({w1['worst_delta']:+.3f}) | {w2['wins']}/{w2['losses']} ({w2['worst_delta']:+.3f})")
    for lam in (10.0, 50.0, 100.0, 1000.0):
        L = str(lam); md += ["", f"## per family, lambda = {lam:g}", "", "family | " + " | ".join(methods) + " | selected", "---|" + "---|" * (len(methods) + 1)]
        for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{per[m][L][fn[g]]:.3f}" for m in methods) + f" | {selected[fn[g]][L]['config']}")
    md += ["", "## verbosity coordinate: true, raw prediction, calibrated prediction; inner calibration line and sigma", "", "family | true | raw | cal | a | b | sigma raw -> cal", "---|---|---|---|---|---|---"]
    for g in range(F): d = diag[fn[g]]; md.append(f"{fn[g]} | {d['gamma_true']:+.2f} | {d['gamma_hat_raw']:+.2f} | {d['gamma_hat_cal']:+.2f} | {d['calibration_a_b'][0]:+.2f} | {d['calibration_a_b'][1]:.2f} | {d['sigma_raw']:.2f} -> {d['sigma_cal']:.2f}")
    C.save(C.RESULTS / EXP / panel_name / feat, res, md)


if __name__ == "__main__":
    main()
