"""exp44: significance and deployability of the key that won exp42, (family, predicted best model).

The cell label is the argmax over models of predicted reward minus lambda times predicted cost, both
predicted from the frozen reader by the cross-fitted ridge of exp42; the lookup then stores, per cell,
the model that is actually best on the training rows of that cell.  exp42's permutation null skipped
this key because it carries no quantile marker, so it is done here:
  * within-family permutation of the cell label (the label is shuffled among the rows of each family,
    training and test alike), 200 draws, so the null keeps the family structure and destroys only the
    information the reader adds;
  * a deployable variant in which the family part of the key is itself predicted from the embedding by
    a multinomial logistic fitted on the training rows;
  * the direct router (take the predicted argmax as the answer, no lookup) and the true-argmax ceiling.
5 splits, both aggregations, paired bootstraps on split 0.
Writes results[_root]/difficulty_llm/exp44_argmax_key_null/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, topic_str  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split, summarise  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp36_augmented_knn import boot, crossfit  # noqa: E402
from exp42_proxy_keys import build_targets, cell_choice, codes  # noqa: E402

OUT = RESULTS_ROOT / "difficulty_llm" / "exp44_argmax_key_null"; N_PERM = 200; REPORT = (0.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--rep", default="computed/qwen7b-l21@prompt")
    _a = ap.parse_args()
    _tag = _a.rep.split("/")[-1].split("@")[0]
    if _tag != "qwen7b-l21":
        OUT = OUT.with_name(OUT.name + "_" + _tag)
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names)
    onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    Y, pnames, M, _ = build_targets(r, c, panel); P = len(pnames)
    topics_all = np.array([str(t) for t in topic_str(panel)], dtype=object); _, topic_code = np.unique(topics_all, return_inverse=True)
    h = load_rep(find_rep(PANEL, _a.rep), n, panel.keep).astype(np.float32)
    gte = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)
    per = {}; pq = {}; perm_out = {}; acc_log = []
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
        dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        Ptr, Pte = crossfit(h, Y, onehot, tr, te, fam, seed); Ptr = np.asarray(Ptr); Pte = np.asarray(Pte)
        rhat = (Ptr[:, P:P + M], Pte[:, P:P + M]); chat = (np.exp(Ptr[:, P + M:]), np.exp(Pte[:, P + M:]))
        sc_g = StandardScaler().fit(gte[tr])
        clf = LogisticRegression(C=1.0, max_iter=300, n_jobs=-1).fit(sc_g.transform(gte[tr]), ftr)
        pf_tr, pf_te = clf.predict(sc_g.transform(gte[tr])), clf.predict(sc_g.transform(gte[te]))
        acc_log.append(float((pf_te == fte).mean()))
        res = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            am_tr = (rhat[0] - lam * chat[0]).argmax(1); am_te = (rhat[1] - lam * chat[1]).argmax(1)
            true_tr, true_te = Utr.argmax(1), U[te].argmax(1)
            keys = {"family": codes([(ftr, fte)]), "family+topic": codes([(ftr, fte), (topic_code[tr], topic_code[te])]),
                    "family+predicted_argmax": codes([(ftr, fte), (am_tr, am_te)]),
                    "DEPLOY::predfamily+predicted_argmax": codes([(pf_tr, pf_te), (am_tr, am_te)]),
                    "CEILING::family+true_argmax": codes([(ftr, fte), (true_tr, true_te)])}
            choices = {nm: cell_choice(ct, ce, ftr, fte, Utr, F, nc) for nm, (ct, ce, nc) in keys.items()}
            choices["direct_argmax_router"] = am_te; choices["oracle"] = true_te
            choices["best_single"] = np.full(len(te), int(Utr.mean(0).argmax()))
            for nm, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); u = s.pop("_per_query_utility")
                res.setdefault(nm, {})[str(lam)] = s["all"]
                if seed == SEEDS[0] and lam in REPORT: pq[(nm, lam)] = u
            if seed == SEEDS[0] and lam in REPORT:
                base = res["family+predicted_argmax"][str(lam)]["per_query"]["utility"]; fam_u = res["family"][str(lam)]["per_query"]["utility"]
                nulls = []
                for d in range(N_PERM):
                    rng = np.random.default_rng(2000 + d); at, ae = am_tr.copy(), am_te.copy()
                    for g in range(F):
                        mt = ftr == g; me = fte == g
                        at[mt] = am_tr[mt][rng.permutation(mt.sum())]; ae[me] = am_te[me][rng.permutation(me.sum())]
                    ct, ce, nc = codes([(ftr, fte), (at, ae)])
                    ch = cell_choice(ct, ce, ftr, fte, Utr, F, nc)
                    nulls.append(float((r[te][np.arange(len(te)), ch] - lam * c[te][np.arange(len(te)), ch]).mean()))
                nulls = np.array(nulls)
                perm_out[str(lam)] = {"observed_gain_over_family": base - fam_u, "null_mean_gain": float(nulls.mean() - fam_u),
                                      "null_p975_gain": float(np.percentile(nulls, 97.5) - fam_u), "p_value": float((nulls >= base).mean()), "n_perm": N_PERM}
                print(f"  lam={lam:g}: observed {base - fam_u:+.4f}, null {nulls.mean() - fam_u:+.4f}, p={perm_out[str(lam)]['p_value']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        per[seed] = res; print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    methods = list(per[SEEDS[0]])
    table = {str(lam): {nm: {m_: float(np.mean([per[s][nm][str(lam)][m_]["utility"] for s in SEEDS])) for m_ in AGGS} for nm in methods} for lam in LAMBDAS}
    rng = np.random.default_rng(0); _, te0 = split(n, fam, SEEDS[0]); te0 = np.sort(te0); fte0 = fam[te0]
    boots = {}
    for lam in REPORT:
        for nm in ("family+predicted_argmax", "DEPLOY::predfamily+predicted_argmax", "direct_argmax_router", "CEILING::family+true_argmax"):
            for ref in ("family", "family+topic"):
                boots[f"{nm} - {ref} @ λ={lam:g}"] = boot(pq[(nm, lam)] - pq[(ref, lam)], fte0, F, rng)
        boots[f"family+predicted_argmax - direct_argmax_router @ λ={lam:g}"] = boot(pq[("family+predicted_argmax", lam)] - pq[("direct_argmax_router", lam)], fte0, F, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": "exp44_argmax_key_null", "representation": _a.rep, "family_classifier_accuracy": acc_log, "by_lambda": table, "bootstrap_split0": boots, "permutation_split0": perm_out, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# exp44 — (family, predicted best model) as a lookup key: null, deployability, ceiling ({panel.name})", "",
          f"Cell label = argmax over models of predicted reward − λ·predicted cost, from the frozen Qwen2.5-7B L21 last-token state, cross-fitted on training rows. Family classifier accuracy on test rows: {np.mean(acc_log):.4f} (mean over splits). 5 splits, mean.", ""]
    for mode in AGGS:
        md += [f"## Utility, {mode}", "", "router | " + " | ".join(f"λ={l:g}" for l in REPORT), "---|" + "---|" * len(REPORT)]
        for nm in methods: md.append(f"{nm} | " + " | ".join(f"{table[str(l)][nm][mode]:.4f}" for l in REPORT))
        md.append("")
    md += ["## Paired bootstraps, split 0 (per query / equal weight)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    md += ["", f"## Within-family permutation null ({N_PERM} draws, cell label shuffled inside each family, split 0)", "", "λ | observed gain over family | null mean | null 97.5th | p", "---|---|---|---|---"]
    for lam, v in perm_out.items():
        md.append(f"{float(lam):g} | {v['observed_gain_over_family']:+.4f} | {v['null_mean_gain']:+.4f} | {v['null_p975_gain']:+.4f} | {v['p_value']:.3f}")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
