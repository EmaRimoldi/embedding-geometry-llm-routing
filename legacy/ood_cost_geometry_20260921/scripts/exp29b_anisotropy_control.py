"""exp29b - is the reader's format axis information or metric anisotropy? For each representation (RB): effective
dimensionality of the centred space (participation ratio of the covariance spectrum; share of variance in the top 1 / 10
components); the same-format vs different-format centroid distance ratio on unit vectors and after PCA-256 WHITENING
(unit variance per component, fitted on all rows); the held-out-family format probe after whitening (fitted on training
families only, PCA/whitening fitted on training rows). Usage: exp29b <panel>."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import common as C
from sprout_common import unit
from exp29_format_geometry import FORMAT, load_rep

EXP = "exp29b_anisotropy_control"; REPS = [("gte_emitted", "gte"), ("gte_trunk_L21", "hidden/gte-qwen2-7b/L21_last"), ("reader_L21", "hidden/qwen2.5-7b-instruct/L21_last"), ("reader_L12", "hidden/qwen2.5-7b-instruct/L12_last")]


def ratio(Z, fam, fkey, fn):
    F = len(fn); cen = np.stack([Z[fam == g].mean(0) for g in range(F)]); spread = np.mean([np.linalg.norm(Z[fam == g] - cen[g], axis=1).mean() for g in range(F)]); ds, dd = [], []
    for a in range(F):
        for b in range(a + 1, F): (ds if fkey[fn[a]] == fkey[fn[b]] else dd).append(np.linalg.norm(cen[a] - cen[b]) / spread)
    return float(np.mean(ds)), float(np.mean(dd))


def main():
    panel_name = sys.argv[1]; D = C.load(panel_name); r, fam, fn = D["r"], D["fam"], D["fnames"]; F, n = len(fn), len(r)
    fmap = FORMAT[panel_name]; fkey = {f: next((v for k, v in fmap.items() if k in f.lower()), "other") for f in fn}; classes = sorted(set(fkey.values())); fmt = np.array([classes.index(fkey[fn[g]]) for g in fam])
    res = C.provenance(D); res.update({"experiment": EXP, "reps": {}}); md = [f"# {EXP} ({panel_name})", "", "rep | eff. dim (participation ratio) | var share top-1 / top-10 comp. | centroid ratio diff/same format, unit vectors | same after PCA-256 whitening | held-out format probe, unit | after whitening", "---|---|---|---|---|---|---"]
    for name, rep in REPS:
        xu = unit(load_rep(D, panel_name, rep)); Xc = xu - xu.mean(0); ev = np.linalg.svd(Xc[np.random.default_rng(0).choice(n, min(n, 8000), replace=False)], compute_uv=False) ** 2; ev /= ev.sum(); pr = float(1 / (ev ** 2).sum())
        ds, dd = ratio(xu, fam, fkey, fn); pw = PCA(n_components=256, whiten=True, svd_solver="randomized", random_state=0).fit(xu); Zw = pw.transform(xu); dsw, ddw = ratio(Zw, fam, fkey, fn)
        acc_u, acc_w, tot = 0, 0, 0
        for g in range(F):
            te = fam == g; tr = ~te
            if len(np.unique(fmt[tr])) < 2: continue
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(xu[tr]); cu = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(sc.transform(xu[tr]), fmt[tr]); acc_u += (cu.predict(sc.transform(xu[te])) == fmt[te]).sum()
            pwt = PCA(n_components=256, whiten=True, svd_solver="randomized", random_state=0).fit(xu[tr]); cw = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced").fit(pwt.transform(xu[tr]), fmt[tr]); acc_w += (cw.predict(pwt.transform(xu[te])) == fmt[te]).sum(); tot += te.sum()
        res["reps"][name] = {"participation_ratio": pr, "top1_var": float(ev[0]), "top10_var": float(ev[:10].sum()), "centroid_same_unit": ds, "centroid_diff_unit": dd, "centroid_same_whitened": dsw, "centroid_diff_whitened": ddw, "probe_unit": float(acc_u / tot), "probe_whitened": float(acc_w / tot)}
        md.append(f"{name} | {pr:.0f} | {ev[0]:.2f} / {ev[:10].sum():.2f} | {dd:.2f}/{ds:.2f} = {dd/ds:.2f} | {ddw:.2f}/{dsw:.2f} = {ddw/dsw:.2f} | {acc_u/tot:.3f} | {acc_w/tot:.3f}"); print(md[-1], flush=True)
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
