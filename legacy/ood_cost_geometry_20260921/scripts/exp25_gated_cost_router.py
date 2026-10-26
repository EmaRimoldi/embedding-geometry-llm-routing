"""exp25 - gated reader-state cost router. Cost estimate = (1 - beta) * transferred family cost (soft prior's p.C)
+ beta * reader cost (rank-1 axis at the coordinate read from the Qwen2.5-7B layer-21 prefill state, raw, z=0, exp07's
clipping). beta = 0 is the soft prior, beta = 1 the reader router. beta is chosen per lambda on TRAINING families only:
  gated_mean : inner leave-one-training-family-out, beta maximising the mean inner utility over {0, .25, .5, .75, 1}
  gated_worst: same with the worst inner family as criterion
  gated_diag : swap diagnostic on the training tables - beta = 1 if routing a training family with another training
               family's COST row loses more than with another family's REWARD row, else beta = 0 (no fitting at all)
References: soft prior, reader (beta = 1), cost oracle (soft prior + true held-out cost vector), family ceiling, best
single of the training rows, oracle. Strict outer holdout; query-weighted [macro]. Usage: exp25 <panel>."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp07_calibrated_verbosity_router import build, READER

EXP = "exp25_gated_cost_router"; BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ("safe", "soft_prior", "reader", "gated_mean", "gated_worst", "gated_diag", "cost_oracle", "family_ceiling", "oracle")


def reader_logcost(blk):
    g = blk["ghat"]; lo, hi = blk["G"].min(), blk["G"].max(); span = hi - lo; g = np.clip(g, lo - .25 * span, hi + .25 * span)
    return blk["alpha"][None, :] + g[:, None] * blk["b"][None, :]


def choose(blk, lam, beta):
    cost = (1 - beta) * np.exp(blk["log_transferred"]) + beta * np.exp(reader_logcost(blk)); return (blk["Pr"] - lam * cost).argmax(1)


def swap_losses(rt, ct, lam):
    U = rt - lam * ct; best = U.max(1); lc, lr = [], []
    for f in range(len(rt)):
        for g in range(len(rt)):
            if f != g: lc.append(best[f] - U[f, (rt[f] - lam * ct[g]).argmax()]); lr.append(best[f] - U[f, (rt[g] - lam * ct[f]).argmax()])
    return float(np.mean(lc)), float(np.mean(lr))


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name); x, r, c, fam, fn = D["x"], D["r"], D["c"], D["fam"], D["fnames"]; F = len(fn)
    ln = np.log1p(D["n_in"])[:, None].astype(np.float32); h = np.load(C.E4 / READER[panel_name], mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(r) else h, dtype=np.float32); X = np.hstack([h, ln]); Y = np.log(c + C.EPS)
    per = {m: {str(l): {} for l in C.LAMS} for m in METHODS}; n_by = {}; sel = {}; diag = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        inner = {s: build(D, X, Y, [q for q in S if q != s], np.flatnonzero(fam == s)) for s in S}; outer = build(D, X, Y, S, te)
        rt, ct = C.family_tables(r, c, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0); sel[fn[g]] = {}; diag[fn[g]] = {}
        for lam in C.LAMS:
            L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te))
            score = {}
            for beta in BETAS:
                vals = []
                for s in S:
                    rows = np.flatnonzero(fam == s); Us = r[rows] - lam * c[rows]; vals.append(float(Us[np.arange(len(rows)), choose(inner[s], lam, beta)].mean()))
                score[beta] = (float(np.mean(vals)), float(np.min(vals)))
            b_mean = max(BETAS, key=lambda b: score[b][0]); b_worst = max(BETAS, key=lambda b: (score[b][1], score[b][0]))
            lc, lr = swap_losses(rt, ct, lam); b_diag = 1.0 if lc > lr else 0.0
            sel[fn[g]][L] = {"beta_mean": b_mean, "beta_worst": b_worst, "beta_diag": b_diag, "inner_scores": {str(b): score[b] for b in BETAS}}; diag[fn[g]][L] = {"swap_loss_cost_row": lc, "swap_loss_reward_row": lr}
            ch = {"safe": np.full(len(te), int((rt - lam * ct).mean(0).argmax())), "soft_prior": choose(outer, lam, 0.0), "reader": choose(outer, lam, 1.0), "gated_mean": choose(outer, lam, b_mean), "gated_worst": choose(outer, lam, b_worst), "gated_diag": choose(outer, lam, b_diag),
                  "cost_oracle": (outer["Pr"] - lam * cbar[None, :]).argmax(1), "family_ceiling": np.full(len(te), int((rbar - lam * cbar).argmax())), "oracle": U.argmax(1)}
            for m in METHODS: per[m][L][fn[g]] = float(U[ii, ch[m]].mean())
        print(fn[g], {L: (sel[fn[g]][L]["beta_mean"], sel[fn[g]][L]["beta_worst"], sel[fn[g]][L]["beta_diag"]) for L in sel[fn[g]]}, round(time.time() - t0), flush=True)
    agg = {m: {L: C.aggregate(per[m][L], n_by) for L in per[m]} for m in METHODS}
    res = C.provenance(D); res.update({"experiment": EXP, "features": "reader (Qwen2.5-7B-Instruct L21 last token + log n_in)", "betas": list(BETAS), "per_family": per, "aggregate_qw_macro": agg, "selected_beta": sel, "swap_diagnostic": diag})
    md = [f"# {EXP} ({panel_name})", "", "Held-out family; cost = (1-beta) transferred + beta reader; beta chosen on training families (inner mean / inner worst / swap diagnostic). Query-weighted [macro].", "", "method | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)]
    for m in METHODS: md.append(f"{m} | " + " | ".join(f"{agg[m][str(l)][0]:.3f} [{agg[m][str(l)][1]:.3f}]" for l in C.LAMS))
    md += ["", "## chosen beta per held-out family (mean / worst / diag)", "", "family | " + " | ".join(f"λ={l:g}" for l in C.LAMS), "---|" + "---|" * len(C.LAMS)] + [f"{f} | " + " | ".join(f"{sel[f][str(l)]['beta_mean']:g} / {sel[f][str(l)]['beta_worst']:g} / {sel[f][str(l)]['beta_diag']:g}" for l in C.LAMS) for f in fn]
    md += ["", "## swap diagnostic on training tables (loss from wrong cost row / wrong reward row), first fold", ""] + [f"- λ={l:g}: {diag[fn[0]][str(l)]['swap_loss_cost_row']:.3f} / {diag[fn[0]][str(l)]['swap_loss_reward_row']:.3f}" for l in C.LAMS]
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
