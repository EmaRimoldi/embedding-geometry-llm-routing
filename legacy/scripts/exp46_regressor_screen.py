"""exp46: which regressor extracts the most from the big space?  Stage one of two.

The space is the one exp45 builds (every frozen representation on the cluster, no reduction,
per-block standardised and clipped, each block divided by the square root of its width), about
37k dimensions.  On split 0 only, and on the three targets that matter for routing (solve count,
mean reward, and whether the pool's cheapest model solves the query), every regressor below is fitted
on the training rows and scored on the test rows by R² beyond the family mean, with its wall-clock
cost recorded.  The family one-hot is given to every model, so the scores are comparable.

  ridge                RidgeCV, the current baseline (linear, L2)
  pls16/64/256         partial least squares, built for p >> n: latent directions that covary with y
  rff_ridge            random Fourier features (4,096) then ridge: an RBF kernel at this scale
  nystroem_ridge       Nyström approximation (2,048 landmarks) then ridge
  pcr256               principal components (256) then ridge
  hgb_pca256           histogram gradient boosting on those components (non-linear, low dimension)
  hgb_topcorr          the same on the 2,000 raw features most correlated with the target
  mlp_pca512           a 256-unit network on 512 components
  knn_reg              average of the 100 nearest training rows' target values (no fitting at all)
  sgd_elasticnet       sparse linear model over all dimensions
  stack_mean           the mean of the ridge, RFF and boosting predictions

Stage two takes the winners into exp45's routing evaluation on all targets and all splits.
Writes results[_root]/difficulty_llm/exp46_regressor_screen/."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from exp36_augmented_knn import topk, unit  # noqa: E402
from exp42_proxy_keys import build_targets  # noqa: E402
from exp45_big_space import BLOCKS, CLIP  # noqa: E402

OUT = RESULTS_ROOT / "difficulty_llm" / "exp46_regressor_screen"
TARGETS = ("n_solved", "mean_reward", "cheap_solves")


def main() -> None:
    import exp45_big_space as big
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); F = len(panel.datasets)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    Y, pnames, M, _ = build_targets(r, c, panel)
    loader = big.__dict__  # reuse exp45's block loader through a local copy of its logic
    sys.argv = [sys.argv[0]]
    # --- build the same block list exp45 uses (no reduction), balanced scaling
    from rb.data import ANALYSIS_ROOT
    from rb.panelctx import REP, lengths_features
    from rb.reps import find_rep, load_rep
    HROOT = ANALYSIS_ROOT / "features" / PANEL / "prompt"

    def take(arr):
        arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr
        assert arr.shape[0] == n; return np.asarray(arr, dtype=np.float32)

    def load(spec):
        kind, mdl, lay = spec
        if kind == "hidden":
            f = HROOT / "hidden" / mdl / f"{lay}.npy"; return take(np.load(f, mmap_mode="r")) if f.exists() else None
        if kind == "delta":
            fa, fb = (HROOT / "hidden" / mdl / f"{x}.npy" for x in lay)
            return take(np.load(fa, mmap_mode="r")) - take(np.load(fb, mmap_mode="r")) if fa.exists() and fb.exists() else None
        if kind == "rep": return load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)
        if kind == "bow":
            f = HROOT / "hash-bow.npy"; return take(np.load(f, mmap_mode="r")) if f.exists() else None
        if kind == "surprise":
            f = HROOT / "qwen7b_hidden" / "prompt_logprob_stats.npz"; return take(np.load(f)["stats"]) if f.exists() else None
        if kind == "lengths": return lengths_features(panel)[0].astype(np.float32)
        return None

    seed = SEEDS[0]; tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
    parts, names_used, width = [], [], 0
    for name, spec, in_ridge in BLOCKS:
        if width >= 40000: break
        x = load(spec)
        if x is None: continue
        xs = np.clip(StandardScaler().fit(x[tr]).transform(x), -CLIP, CLIP).astype(np.float32)
        parts.append((xs / np.sqrt(xs.shape[1])).astype(np.float32)); names_used.append(name); width += xs.shape[1]; del x, xs
    onehot = np.eye(F, dtype=np.float32)[fam]
    X = np.column_stack(parts + [onehot]); del parts
    print(f"big space {X.shape} from {len(names_used)} blocks ({time.time() - t0:.0f}s)", flush=True)
    Xu = unit(X)                                        # unit rows, for the kernel approximations and kNN
    nb = topk(Xu[te], Xu[tr], 100)
    print(f"neighbours ready ({time.time() - t0:.0f}s)", flush=True)

    def r2_beyond(y, yhat):
        floor = np.array([y[tr][ftr == g].mean() for g in range(F)])[fte]
        return float(1 - ((y[te] - yhat) ** 2).sum() / max(((y[te] - floor) ** 2).sum(), 1e-12))

    res = {}
    for tname in TARGETS:
        y = Y[:, pnames.index(tname)]; res[tname] = {}

        def run(label, fn):
            t = time.time()
            try:
                yhat = fn(y)
                res[tname][label] = {"r2_beyond_family": r2_beyond(y, yhat), "seconds": time.time() - t}
            except Exception as exc:
                res[tname][label] = {"error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - t}
            print(f"  {tname:14s} {label:16s} {res[tname][label].get('r2_beyond_family', 'ERR')} ({res[tname][label]['seconds']:.0f}s)", flush=True)

        run("ridge", lambda y: RidgeCV(alphas=RIDGE_ALPHAS).fit(X[tr], y[tr]).predict(X[te]))
        for k in (16, 64, 256):
            run(f"pls{k}", lambda y, k=k: PLSRegression(n_components=k, scale=False).fit(X[tr], y[tr]).predict(X[te]).ravel())
        def rff(y, D=4096, gamma=1.0):
            m = RBFSampler(gamma=gamma, n_components=D, random_state=0).fit(Xu[tr])
            return RidgeCV(alphas=RIDGE_ALPHAS).fit(m.transform(Xu[tr]), y[tr]).predict(m.transform(Xu[te]))
        run("rff_ridge", rff)
        def nys(y, D=2048):
            m = Nystroem(gamma=1.0, n_components=D, random_state=0).fit(Xu[tr])
            return RidgeCV(alphas=RIDGE_ALPHAS).fit(m.transform(Xu[tr]), y[tr]).predict(m.transform(Xu[te]))
        run("nystroem_ridge", nys)
        p256 = PCA(256, svd_solver="randomized", random_state=0).fit(X[tr]); Z256tr, Z256te = p256.transform(X[tr]), p256.transform(X[te])
        run("pcr256", lambda y: RidgeCV(alphas=RIDGE_ALPHAS).fit(Z256tr, y[tr]).predict(Z256te))
        run("hgb_pca256", lambda y: HistGradientBoostingRegressor(max_iter=300, early_stopping=True, random_state=0).fit(Z256tr, y[tr]).predict(Z256te))
        def hgb_top(y, k=2000):
            yc = y[tr] - y[tr].mean(); num = (X[tr] * yc[:, None]).sum(0); den = np.linalg.norm(X[tr], axis=0) * np.linalg.norm(yc) + 1e-9
            idx = np.argsort(-np.abs(num / den))[:k]
            return HistGradientBoostingRegressor(max_iter=300, early_stopping=True, random_state=0).fit(X[tr][:, idx], y[tr]).predict(X[te][:, idx])
        run("hgb_topcorr", hgb_top)
        p512 = PCA(512, svd_solver="randomized", random_state=0).fit(X[tr]); Z512tr, Z512te = p512.transform(X[tr]), p512.transform(X[te])
        run("mlp_pca512", lambda y: MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, max_iter=300, random_state=0).fit(Z512tr, y[tr]).predict(Z512te))
        run("knn_reg", lambda y: y[tr][nb].mean(1))
        run("sgd_elasticnet", lambda y: SGDRegressor(penalty="elasticnet", alpha=1e-4, l1_ratio=0.15, max_iter=50, random_state=0).fit(X[tr], y[tr]).predict(X[te]))
        def stack(y):
            a = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[tr], y[tr]).predict(X[te]); b = rff(y)
            d = HistGradientBoostingRegressor(max_iter=300, early_stopping=True, random_state=0).fit(Z256tr, y[tr]).predict(Z256te)
            return (a + b + d) / 3.0
        run("stack_mean", stack)

    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": "exp46_regressor_screen", "split": int(seed), "space_dims": int(X.shape[1]), "blocks": names_used, "targets": list(TARGETS), "results": res, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    labels = list(res[TARGETS[0]])
    md = [f"# exp46 — which regressor extracts most from the {X.shape[1]}-dimensional space ({panel.name}, split 0)", "",
          f"{len(names_used)} blocks, no reduction, per-block standardised and clipped at ±{CLIP}, each divided by the square root of its width, family one-hot appended. R² beyond the family mean on the test rows; seconds are wall clock.", "",
          "regressor | " + " | ".join(f"{t} R² (s)" for t in TARGETS), "---|" + "---|" * len(TARGETS)]
    for lab in labels:
        cells = []
        for t in TARGETS:
            v = res[t].get(lab, {})
            cells.append(f"{v['r2_beyond_family']:+.3f} ({v['seconds']:.0f})" if "r2_beyond_family" in v else f"ERR ({v.get('seconds', 0):.0f})")
        md.append(f"{lab} | " + " | ".join(cells))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
