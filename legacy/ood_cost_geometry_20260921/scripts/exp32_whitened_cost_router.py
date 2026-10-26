"""exp32 - does GTE read the verbosity coordinate once whitened? Same reader-router pipeline as exp07 (rank-1 cost axis,
raw, z=0, soft-prior reward), with the coordinate regressed from PCA-whitened features fitted on the TRAINING families of
each fold (k = 64, 256 components): emitted GTE, the reader L21 state, and GTE's trunk L21; plus the unwhitened emitted GTE
and reader as references. Held-out family; qw [macro] at C.LAMS. Usage: exp32 <panel>."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.decomposition import PCA
import common as C
from exp05_verbosity_axis_router import axes, coords, fit_scalar
from exp07_calibrated_verbosity_router import READER
from exp25_gated_cost_router import reader_logcost

EXP = "exp32_whitened_cost_router"; KS = (64, 256)


def build_w(D, Xraw, Y, S, te, k):
    """exp07.build with whitening fitted on training rows only."""
    x, r, c, fam = D["x"], D["r"], D["c"], D["fam"]; tr = np.flatnonzero(np.isin(fam, S))
    pw = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=0).fit(Xraw[tr]); Xtr, Xte = pw.transform(Xraw[tr]), pw.transform(Xraw[te])
    ln = np.log1p(D["n_in"])[:, None].astype(np.float32); Xtr = np.hstack([Xtr, ln[tr]]); Xte = np.hstack([Xte, ln[te]])
    w = np.array([1.0 / (fam[tr] == fam[i]).sum() for i in tr]); w *= len(w) / w.sum()
    rt, ct = C.family_tables(r, c, fam, S); alpha, B, G = axes(np.log(ct + C.EPS), 1); v = coords(Y[tr], alpha, B)[:, 0]; ghat = fit_scalar(Xtr, v, w)(Xte)[:, 0]
    P = C.fit_posterior(x, fam, S, te, True, 42)
    return {"Pr": P @ rt, "log_transferred": np.log(P @ ct + C.EPS), "alpha": alpha, "b": B[0], "G": G[:, 0], "ghat": ghat, "gtrue": float(coords(Y[te], alpha, B)[:, 0].mean())}


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name); x, r, c, fam, fn = D["x"], D["r"], D["c"], D["fam"], D["fnames"]; F = len(fn); Y = np.log(c + C.EPS)
    h = np.load(C.E4 / READER[panel_name], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(r) else h, dtype=np.float32)
    g21 = np.load(C.E4 / "features" / panel_name / "prompt" / "hidden" / "gte-qwen2-7b" / "L21_last.npy", mmap_mode="r"); g21 = np.asarray(g21[D["panel"].keep] if g21.shape[0] != len(r) else g21, dtype=np.float32)
    RAW = {"gte": x, "reader_L21": h, "gte_trunk_L21": g21}; ln = np.log1p(D["n_in"])[:, None].astype(np.float32)
    from exp07_calibrated_verbosity_router import build
    per = {}; n_by = {}; gerr = {}
    def rec(m, L, g, v): per.setdefault(m, {}).setdefault(L, {})[fn[g]] = float(v)
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te); ii = np.arange(len(te)); rt, ct = C.family_tables(r, c, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0)
        blks = {"gte_std": build(D, np.hstack([x, ln]), Y, S, te), "reader_std": build(D, np.hstack([h, ln]), Y, S, te)}
        for name, Xr in RAW.items():
            for k in KS: blks[f"{name}_whiten{k}"] = build_w(D, Xr, Y, S, te, k)
        gerr[fn[g]] = {m: round(float(b["ghat"].mean() - b["gtrue"]), 2) for m, b in blks.items()}
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; b0 = blks["gte_std"]
            rec("soft_prior", L, g, U[ii, (b0["Pr"] - lam * np.exp(b0["log_transferred"])).argmax(1)].mean()); rec("cost_oracle", L, g, U[ii, (b0["Pr"] - lam * cbar[None, :]).argmax(1)].mean()); rec("family_ceiling", L, g, U[ii, int((rbar - lam * cbar).argmax())].mean())
            for m, b in blks.items(): rec(m, L, g, U[ii, (b["Pr"] - lam * np.exp(reader_logcost(b))).argmax(1)].mean())
        print(fn[g], gerr[fn[g]], round(time.time() - t0), flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in per}; res = C.provenance(D); res.update({"experiment": EXP, "aggregate_qw_macro": agg, "per_family": per, "gamma_error_per_family": gerr})
    md = [f"# {EXP} ({panel_name})", "", "Held-out family; reader-router cost from the coordinate regressed on standardised or PCA-whitened features (whitening fitted on training families). qw [macro].", "", "features | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)]
    for m in per: md.append(f"{m} | " + " | ".join(f"{agg[m][str(l)][0]:.3f} [{agg[m][str(l)][1]:.3f}]" for l in C.LAMS))
    md += ["", "## held-out coordinate error (gamma_hat mean - gamma_true) per family", ""] + [f"- {f}: {v}" for f, v in gerr.items()]
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
