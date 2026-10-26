"""exp37: which frozen reader, which layer, which pooling, which transformation makes per-query
difficulty recoverable?  Same target and protocol as exp18/exp35 (ridge on family one-hot +
standardised representation predicting the solve count; R² beyond the family mean on test rows;
5 stratified 70/30 splits; both aggregations; bootstrap CI on split 0; within-family shuffle null
for the best row), applied to every layer x pooling stored by scripts/extract_hidden_all_layers.py
for one model, plus transformations of the best layer:
  * PCA to 64 and 256 dimensions (fitted on training rows)      - a cheap-to-store version
  * concatenation of the layers at 1/2, 3/4 and full depth       - multi-layer read-out
  * a small MLP probe (256 hidden units, early stopping)          - is the linear read-out the limit?
  * for qwen2.5-7b-instruct: the vocabulary projection at the last token (entropy + top-64 log-probs)
    and the prompt surprise scalars of exp35 (mean/last-32 log-prob and entropy, token count).
Writes results[_root]/difficulty_llm/exp37_layer_sweep/<model>/result.{json,md}.
Usage: exp37_layer_sweep.py --model qwen3-8b   (panel from E4_PANEL)"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT  # noqa: E402
from rb.probes.constants import RIDGE_ALPHAS  # noqa: E402
from rb.probes.sweep import SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp18_recovering_difficulty import boot_r2b, metrics  # noqa: E402

EXP = "exp37_layer_sweep"; N_PERM = 3


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    HID = ANALYSIS_ROOT / "features" / PANEL / "prompt" / "hidden" / a.model; meta = json.load(open(HID / "meta.json"))
    OUT = RESULTS_ROOT / "difficulty_llm" / EXP / a.model; OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names); onehot = np.eye(F)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); y = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
    splits = []
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); splits.append((tr, te, np.array([y[tr][fam[tr] == g].mean() for g in range(F)])[fam[te]]))

    def load(name):
        arr = np.load(HID / f"{name}.npy", mmap_mode="r"); arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr
        assert arr.shape[0] == n, (name, arr.shape); return np.asarray(arr, dtype=np.float32)

    def evaluate(x, probe="ridge", pca=None, only_seed0=False):
        rows = []
        for si, (tr, te, floor) in enumerate(splits):
            if only_seed0 and si > 0: break
            sc = StandardScaler().fit(x[tr]); xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
            if pca:
                p = PCA(n_components=pca, random_state=0).fit(xtr); xtr, xte = p.transform(xtr), p.transform(xte)
            xtr = np.column_stack([onehot[tr], xtr]); xte = np.column_stack([onehot[te], xte])
            if probe == "ridge":
                yhat = RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, y[tr]).predict(xte)
            else:
                yhat = MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=si).fit(xtr, y[tr]).predict(xte)
            m = metrics(y[te], yhat, floor, fam[te], F)
            if si == 0: m["ci_split0"] = boot_r2b(y[te], yhat, floor, np.random.default_rng(0))
            rows.append(m)
        pf = {names[g]: float(np.mean([d["per_family"][g]["r2_beyond_family"] for d in rows if g in d["per_family"]])) for g in range(F)}
        return {"r2_beyond_family": float(np.mean([d["r2_beyond_family"] for d in rows])), "r2_beyond_family_sd": float(np.std([d["r2_beyond_family"] for d in rows])),
                "equal_weight": float(np.mean([np.mean([v["r2_beyond_family"] for v in d["per_family"].values() if np.isfinite(v["r2_beyond_family"])]) for d in rows])),
                "r2": float(np.mean([d["r2"] for d in rows])), "spearman": float(np.mean([d["spearman"] for d in rows])), "ci_split0": rows[0]["ci_split0"], "per_family": pf, "n_splits": len(rows)}

    t0 = time.time(); res = {"gte_reference": evaluate(load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32))}
    print(f"gte {res['gte_reference']['r2_beyond_family']:+.3f}", flush=True)
    NL, pools = meta["layers"], meta["poolings"]; layers = {}
    for p in pools:
        for L in range(NL):
            layers[f"L{L}_{p}"] = evaluate(load(f"L{L}_{p}")); print(f"L{L}_{p} {layers[f'L{L}_{p}']['r2_beyond_family']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    res["layers"] = layers; best = max(layers, key=lambda k: layers[k]["r2_beyond_family"]); res["best_layer"] = best; bl, bp = re.match(r"L(\d+)_(\w+)", best).groups(); bl = int(bl)
    xb = load(best); extras = {"pca64": evaluate(xb, pca=64), "pca256": evaluate(xb, pca=256)}
    d = NL - 1; cat = [f"L{max(1, d // 2)}_{bp}", f"L{max(1, 3 * d // 4)}_{bp}", f"L{d}_{bp}"]; extras["concat_" + "+".join(cat)] = evaluate(np.concatenate([load(c) for c in cat], axis=1))
    extras["mlp_probe_" + best] = evaluate(xb, probe="mlp")
    if meta.get("logit_lens"):
        extras["logit_lens_last(65d)"] = evaluate(load("logit_lens_last"))
        st = np.load(ANALYSIS_ROOT / "features" / PANEL / "prompt" / "qwen7b_hidden" / "prompt_logprob_stats.npz")["stats"]; st = st[panel.keep] if panel.keep is not None and st.shape[0] == len(panel.keep) else st
        extras["prompt_surprise_scalars(5d)"] = evaluate(st.astype(np.float32)); extras["logit_lens+surprise(70d)"] = evaluate(np.concatenate([load("logit_lens_last"), st.astype(np.float32)], axis=1))
    res["extras"] = extras
    # within-family shuffle null for the best layer (split 0)
    tr, te, _ = splits[0]; sc = StandardScaler().fit(xb[tr]); xtr = np.column_stack([onehot[tr], sc.transform(xb[tr])]); xte = np.column_stack([onehot[te], sc.transform(xb[te])]); vals = []
    for pi in range(N_PERM):
        rng = np.random.default_rng(100 + pi); yp = y.copy()
        for g in range(F):
            rows_ = np.flatnonzero(fam == g); yp[rows_] = y[rows_][rng.permutation(len(rows_))]
        fl = np.array([yp[tr][fam[tr] == g].mean() for g in range(F)])[fam[te]]; vals.append(metrics(yp[te], RidgeCV(alphas=RIDGE_ALPHAS).fit(xtr, yp[tr]).predict(xte), fl, fam[te], F)["r2_beyond_family"])
    res["null_best_layer"] = {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "n": N_PERM}
    out = {**panel.provenance(), "experiment": EXP, "model": a.model, "model_meta": meta, "target": "n_solved", "seconds": time.time() - t0, **res}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    f = lambda k, v: f"{k} | {v['r2_beyond_family']:+.3f} ± {v['r2_beyond_family_sd']:.3f} | {v['equal_weight']:+.3f} | {v['spearman']:.3f} | [{v['ci_split0'][0]:+.3f}, {v['ci_split0'][1]:+.3f}]"  # noqa: E731
    md = [f"# {EXP} — {a.model} on {panel.name}", "", f"{meta['kind']}, {NL - 1} layers, hidden {meta['hidden_size']}, window {meta['max_len']}, poolings {pools}; sanity vs stored embedding: {meta.get('sanity_vs_stored')}", "",
          "Ridge on family one-hot + standardised representation, target = number of models solving the query; R²_beyond_family (mean ± sd over 5 splits), equal weight over families, Spearman, bootstrap CI on split 0.", "",
          "representation | R²_beyond_family | equal weight | Spearman | CI (split 0)", "---|---|---|---|---", f("gte (reference)", res["gte_reference"])]
    for k in layers: md.append(f(k, layers[k]))
    md += ["", f"Best layer: **{best}**; within-family shuffle null: mean {res['null_best_layer']['mean']:+.3f}, max {res['null_best_layer']['max']:+.3f}", "", "## Transformations of the best layer", "", "representation | R²_beyond_family | equal weight | Spearman | CI (split 0)", "---|---|---|---|---"]
    for k in extras: md.append(f(k, extras[k]))
    md += ["", "## Per family (best layer vs GTE)", "", "representation | " + " | ".join(names), "---|" + "---|" * F, "gte | " + " | ".join(f"{res['gte_reference']['per_family'][g]:+.3f}" for g in names), f"{best} | " + " | ".join(f"{layers[best]['per_family'][g]:+.3f}" for g in names)]
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[-12:]))


if __name__ == "__main__":
    main()
