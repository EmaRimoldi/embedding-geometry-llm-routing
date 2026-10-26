"""exp28 - which representations read the verbosity coordinate? Held-out family, the exp07 reader router (rank-1 axis,
raw, z=0) with the coordinate regressed from each candidate representation (+ log n_in): Qwen2.5-7B-Instruct last-token
states at layers 0..28 (step 4) and 21, GTE-Qwen2-7B's own trunk at the same layers plus its emitted vector, the 1.5B
coder decoder (L14, L28), Qwen3-embedding-0.6B (L14, L28). Reports the routing utility at C.LAMS (qw [macro]) and the
held-out coordinate error (RMS of gamma_hat mean vs gamma_true over families), against soft prior / cost oracle / ceiling.
Usage: exp28 <panel>."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp07_calibrated_verbosity_router import build
from exp25_gated_cost_router import choose

EXP = "exp28_verbosity_layer_sweep"
REPS = [("gte_emitted", None)] + [(f"reader_L{L}", f"hidden/qwen2.5-7b-instruct/L{L}_last") for L in (0, 4, 8, 12, 16, 20, 21, 24, 28)] + [(f"gte_trunk_L{L}", f"hidden/gte-qwen2-7b/L{L}_last") for L in (8, 16, 21, 24, 28)] + [(f"coder1.5b_L{L}", f"hidden/qwen2.5-coder-1.5b-instruct/L{L}_last") for L in (14, 28)] + [(f"qwen3emb0.6b_L{L}", f"hidden/qwen3-embedding-0.6b/L{L}_last") for L in (14, 28)]


def main():
    panel_name = sys.argv[1]; t0 = time.time(); D = C.load(panel_name); x, r, c, fam, fn = D["x"], D["r"], D["c"], D["fam"], D["fnames"]; F = len(fn); ln = np.log1p(D["n_in"])[:, None].astype(np.float32); Y = np.log(c + C.EPS)
    res = C.provenance(D); res.update({"experiment": EXP, "reps": {}}); n_by = {fn[g]: int((fam == g).sum()) for g in range(F)}; refs = {m: {str(l): {} for l in C.LAMS} for m in ("soft_prior", "cost_oracle", "family_ceiling")}
    for name, path in REPS:
        p = C.E4 / "features" / panel_name / "prompt" / (path + ".npy") if path else None
        if p is not None and not p.exists(): print("missing", name, flush=True); continue
        h = x if path is None else np.load(p, mmap_mode="r"); h = np.asarray(h[D["panel"].keep] if h.shape[0] != len(r) else h, dtype=np.float32); X = np.hstack([h, ln])
        per = {str(l): {} for l in C.LAMS}; err = []
        for g in range(F):
            te = np.flatnonzero(fam == g); S = [s for s in range(F) if s != g]; blk = build(D, X, Y, S, te); err.append(float(blk["ghat"].mean() - blk["gtrue"])); rt, ct = C.family_tables(r, c, fam, S); rbar, cbar = r[te].mean(0), c[te].mean(0)
            for lam in C.LAMS:
                L = str(lam); U = r[te] - lam * c[te]; ii = np.arange(len(te)); per[L][fn[g]] = float(U[ii, choose(blk, lam, 1.0)].mean())
                if name == REPS[0][0]: refs["soft_prior"][L][fn[g]] = float(U[ii, choose(blk, lam, 0.0)].mean()); refs["cost_oracle"][L][fn[g]] = float(U[ii, (blk["Pr"] - lam * cbar[None, :]).argmax(1)].mean()); refs["family_ceiling"][L][fn[g]] = float(U[ii, int((rbar - lam * cbar).argmax())].mean())
        res["reps"][name] = {"dim": int(X.shape[1]), "utility": {L: C.aggregate(per[L], n_by) for L in per}, "gamma_rms_error": float(np.sqrt(np.mean(np.square(err)))), "gamma_error_per_family": dict(zip(fn, err))}
        print(name, {L: round(res["reps"][name]["utility"][L][0], 3) for L in per}, "rms", round(res["reps"][name]["gamma_rms_error"], 2), round(time.time() - t0), flush=True)
    res["references"] = {m: {L: C.aggregate(v, n_by) for L, v in d.items()} for m, d in refs.items()}
    md = [f"# {EXP} ({panel_name})", "", "Held-out family; reader router with the coordinate read from each representation (+ log n_in). qw [macro]; gamma RMS = held-out coordinate error over families.", "", "representation | dim | " + " | ".join(f"λ={l:g}" for l in C.LAMS) + " | gamma RMS", "---|---|" + "---|" * len(C.LAMS) + "---"]
    for name, v in res["reps"].items(): md.append(f"{name} | {v['dim']} | " + " | ".join(f"{v['utility'][str(l)][0]:.3f} [{v['utility'][str(l)][1]:.3f}]" for l in C.LAMS) + f" | {v['gamma_rms_error']:.2f}")
    for m, d in res["references"].items(): md.append(f"{m} | - | " + " | ".join(f"{d[str(l)][0]:.3f} [{d[str(l)][1]:.3f}]" for l in C.LAMS) + " | -")
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
