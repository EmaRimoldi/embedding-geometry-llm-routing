"""exp47: the binary targets deserve classifiers, not a linear regression on a 0/1 column.

exp43's ceiling table showed that the cell label worth the most is a binary — whether the pool's
cheapest model solves the query — and that almost all of its value is lost to prediction error
(LLMRouterBench, equal weight, lambda=0: true label 0.7236, predicted label 0.6789, family lookup
0.6749).  Every binary in the study so far (that one, and the per-model success vector that drives
the winning (family, predicted best model) key) has been predicted by ridge regression on a 0/1
target, which is neither calibrated nor a proper scoring rule.

Compared here, all cross-fitted on the training rows exactly as before, input = the frozen
Qwen2.5-7B-Instruct layer-21 last-token state:
  ridge_full     ridge on the raw 3,584 dimensions              (the established baseline)
  ridge_pca512   ridge on 512 unsupervised components           (input control for the classifiers)
  logistic       L2 logistic regression on the same components  (proper likelihood, calibrated)
  hgb            histogram gradient boosting classifier         (non-linear, cheap_solves only)
Targets: cheap_solves and the per-model success vector.  Quality is reported as AUC, Brier score and
R^2 beyond the family mean; routing is then re-run with each predictor feeding (a) the
(family, cheap-solves bin) lookup and (b) the (family, predicted best model) lookup, against the
family lookup, the (family, topic) lookup and the true-label ceilings.
Writes results[_root]/difficulty_llm/exp47_binary_targets/."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, topic_str  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split, summarise  # noqa: E402
from exp36_augmented_knn import boot  # noqa: E402
from exp42_proxy_keys import cell_choice, codes, qbins  # noqa: E402

OUT = RESULTS_ROOT / "difficulty_llm" / "exp47_binary_targets"
NPCA = 512; QS = (2, 3); REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight")


def crossfit_binary(kind, Xfull, Xp, B, onehot, tr, te, fam, seed):
    """Out-of-fold predictions on the training rows, full-fit predictions on the test rows."""
    ptr = np.zeros((len(tr), B.shape[1]), dtype=np.float32); pte = np.zeros((len(te), B.shape[1]), dtype=np.float32)

    def fit_predict(Xa, ya, Xb):
        out = np.zeros((Xb.shape[0], ya.shape[1]), dtype=np.float32)
        if kind == "ridge_full" or kind == "ridge_pca512":
            m = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(Xa, ya); return m.predict(Xb).astype(np.float32)
        for j in range(ya.shape[1]):
            col = ya[:, j]
            if col.min() == col.max():
                out[:, j] = col[0]; continue
            if kind == "logistic":
                mdl = LogisticRegression(C=1.0, max_iter=400).fit(Xa, col)
            else:
                mdl = HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=0).fit(Xa, col)
            out[:, j] = mdl.predict_proba(Xb)[:, 1]
        return out

    X = Xfull if kind == "ridge_full" else Xp
    Xd = np.column_stack([onehot, X])          # every predictor gets the family one-hot, so the comparison is fair
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, fam[tr]):
        ptr[b_] = fit_predict(Xd[tr[a_]], B[tr[a_]], Xd[tr[b_]])
    pte[:] = fit_predict(Xd[tr], B[tr], Xd[te])
    return ptr, pte


def main() -> None:
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); M = panel.m
    onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    S = (r >= SUCCESS_THRESHOLD).astype(np.float32)
    cheapest = int(np.median(c, axis=0).argmin()); cheap = S[:, cheapest:cheapest + 1]
    B = np.column_stack([cheap, S])                       # target 0 = cheap_solves, then per-model success
    logc = np.log(c + 1e-6).astype(np.float32)
    topics_all = np.array([str(t) for t in topic_str(panel)], dtype=object); _, topic_code = np.unique(topics_all, return_inverse=True)
    h = np.load(ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / "qwen2.5-7b-instruct" / "L21_last.npy", mmap_mode="r")
    h = h[panel.keep] if panel.keep is not None and h.shape[0] == len(panel.keep) else h; h = np.asarray(h, dtype=np.float32)
    print(f"panel {PANEL}: n={n} M={M} cheapest={panel.models[cheapest]} base rate cheap_solves={cheap.mean():.3f}", flush=True)
    KINDS = ("ridge_full", "ridge_pca512", "logistic", "hgb")
    per = {}; quality = {}; pq = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
        dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        sc = StandardScaler().fit(h[tr]); hs = sc.transform(h).astype(np.float32)
        pca = PCA(NPCA, svd_solver="randomized", random_state=0).fit(hs[tr]); hp = pca.transform(hs).astype(np.float32)
        cost_tr, cost_te = None, None
        mc = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(np.column_stack([onehot[tr], hs[tr]]), logc[tr])
        cost_te = np.exp(mc.predict(np.column_stack([onehot[te], hs[te]])))
        ctr = np.zeros((len(tr), M), dtype=np.float32)
        for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, ftr):
            m2 = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(np.column_stack([onehot[tr[a_]], hs[tr[a_]]]), logc[tr[a_]])
            ctr[b_] = m2.predict(np.column_stack([onehot[tr[b_]], hs[tr[b_]]]))
        cost_tr = np.exp(ctr)
        preds = {}
        for kind in KINDS:
            tk = time.time()
            cols = slice(0, 1) if kind == "hgb" else slice(0, B.shape[1])
            ptr, pte = crossfit_binary(kind, hs, hp, B[:, cols], onehot, tr, te, fam, seed)
            preds[kind] = (ptr, pte, cols)
            if seed == SEEDS[0]:
                q = {}
                for j in range(pte.shape[1]):
                    yt = B[te, cols][:, j] if isinstance(cols, slice) else B[te, j]
                    nm = "cheap_solves" if j == 0 else f"model_{j - 1}"
                    if yt.min() < yt.max():
                        fl = np.array([B[tr, cols][:, j][ftr == g].mean() for g in range(F)])[fte]
                        q[nm] = {"auc": float(roc_auc_score(yt, pte[:, j])), "brier": float(brier_score_loss(yt, np.clip(pte[:, j], 0, 1))),
                                 "r2_beyond_family": float(1 - ((yt - pte[:, j]) ** 2).sum() / max(((yt - fl) ** 2).sum(), 1e-12))}
                quality[kind] = q
            print(f"seed {seed} {kind}: {time.time() - tk:.0f}s ({time.time() - t0:.0f}s total)", flush=True)
        res = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            keys = {"lookup::family": codes([(ftr, fte)]), "lookup::family+topic": codes([(ftr, fte), (topic_code[tr], topic_code[te])])}
            # the true binary is used as its own label; quantile bins on a 0/1 column are degenerate
            keys["CEILING::family+cheap_solves_TRUE"] = codes([(ftr, fte), (B[tr, 0].astype(int), B[te, 0].astype(int))])
            keys["CEILING::family+true_argmax"] = codes([(ftr, fte), (Utr.argmax(1), U[te].argmax(1))])
            for kind, (ptr, pte, cols) in preds.items():
                for Q in QS:
                    bt, be = qbins(ptr[:, 0], pte[:, 0], ftr, fte, F, Q)
                    keys[f"lookup::family+cheap_solves[{kind}]/Q{Q}"] = codes([(ftr, fte), (bt, be)])
                if pte.shape[1] > 1:
                    am_tr = (ptr[:, 1:] - lam * cost_tr).argmax(1); am_te = (pte[:, 1:] - lam * cost_te).argmax(1)
                    keys[f"lookup::family+predicted_argmax[{kind}]"] = codes([(ftr, fte), (am_tr, am_te)])
            choices = {nm: cell_choice(ct, ce, ftr, fte, Utr, F, nc) for nm, (ct, ce, nc) in keys.items()}
            for kind, (ptr, pte, cols) in preds.items():
                if pte.shape[1] > 1: choices[f"direct_argmax[{kind}]"] = (pte[:, 1:] - lam * cost_te).argmax(1)
            choices["best_single"] = np.full(len(te), int(Utr.mean(0).argmax())); choices["oracle"] = U[te].argmax(1)
            for nm, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); u = s.pop("_per_query_utility")
                res.setdefault(nm, {})[str(lam)] = s["all"]
                if seed == SEEDS[0] and lam in REPORT: pq[(nm, lam)] = u
            if lam in REPORT: print(f"  seed {seed} lam={lam:g} done ({time.time() - t0:.0f}s)", flush=True)
        per[seed] = res
    methods = list(per[SEEDS[0]])
    table = {str(lam): {nm: {m_: float(np.mean([per[s][nm][str(lam)][m_]["utility"] for s in SEEDS])) for m_ in AGGS} for nm in methods} for lam in LAMBDAS}
    rng = np.random.default_rng(0); _, te0 = split(n, fam, SEEDS[0]); te0 = np.sort(te0); fte0 = fam[te0]
    boots = {}
    for lam in REPORT:
        for nm in methods:
            if nm.startswith("lookup::family+") and (nm, lam) in pq:
                boots[f"{nm} - family @ λ={lam:g}"] = boot(pq[(nm, lam)] - pq[("lookup::family", lam)], fte0, F, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": "exp47_binary_targets", "cheapest_model": str(panel.models[cheapest]), "n_pca": NPCA,
           "quality_split0": quality, "by_lambda": table, "bootstrap_split0": boots, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# exp47 — classifiers instead of ridge for the binary targets ({panel.name})", "",
          f"Input: frozen Qwen2.5-7B-Instruct layer 21, last token. Cheapest model: {panel.models[cheapest]}. Cross-fitted on the training rows; 5 splits.", "",
          "## Prediction quality on the binary targets (split 0)", "", "predictor | cheap_solves AUC | Brier | R² beyond family | mean per-model AUC", "---|---|---|---|---"]
    for kind in KINDS:
        q = quality.get(kind, {})
        if "cheap_solves" not in q: continue
        cs = q["cheap_solves"]; pm = [v["auc"] for k, v in q.items() if k != "cheap_solves"]
        md.append(f"{kind} | {cs['auc']:.3f} | {cs['brier']:.4f} | {cs['r2_beyond_family']:+.3f} | " + (f"{np.mean(pm):.3f}" if pm else "—"))
    for mode in AGGS:
        md += ["", f"## Utility, {mode}", "", "router | " + " | ".join(f"λ={l:g}" for l in REPORT), "---|" + "---|" * len(REPORT)]
        for nm in sorted(methods): md.append(f"{nm} | " + " | ".join(f"{table[str(l)][nm][mode]:.4f}" for l in REPORT))
    md += ["", "## Paired bootstraps against the family lookup, split 0 (per query / equal weight)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
