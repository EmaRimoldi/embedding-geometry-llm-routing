"""exp45: the biggest space we can honestly build (~40k dimensions), used raw for kNN and ridge.

No dimensionality reduction anywhere.  Blocks are added in a fixed priority order until the target
width is reached; the order encodes the only "smartness" in the construction:
  1. one representative per frozen model first (source diversity: the 7B instruct reader, the GTE
     embedding, the 1.5B coder, the 0.6B contrastive decoder, e5, bge-m3, MiniLM, the hashed
     bag-of-words, the vocabulary projection, the prompt surprise scalars, the token counts),
     each at the depth the layer sweep found best for it;
  2. then depth diversity inside the strongest models (adjacent layers are nearly collinear, so the
     depths are spread across the network rather than consecutive);
  3. then pooling diversity (last token vs mean) and depth differences.

Two scalings of the same blocks, because in a wide concatenation the metric is decided by scale:
  big_raw       每 block standardised on the training rows only  -> the widest blocks dominate the
                distance, which is the literally ignorant version;
  big_balanced  standardised and then divided by sqrt(block width) -> every block contributes the
                same expected squared norm, so a 384-wide encoder counts as much as a 3,584-wide one.
Depth-difference blocks enter the kNN spaces only: for a linear ridge they are exact linear
combinations of the layers already present and would only worsen the conditioning.

Evaluated, 5 splits, 15-lambda grid, both aggregations, k selected on an inner 80/20 split of the
training rows: kNN on big_raw and big_balanced; the multi-output ridge on the big space predicting
the outcome-table proxies of exp42, its read-out R² against the 3,584-wide single-layer baseline,
kNN on the resulting predicted-proxy space, and the (family, predicted best model) lookup built from
it; references are the family lookup, the (family, topic) lookup, GTE kNN, the best single model and
the oracle.  Writes results[_root]/difficulty_llm/exp45_big_space/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, lengths_features, topic_str  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split, summarise  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from exp36_augmented_knn import boot, crossfit, topk, unit  # noqa: E402
from exp42_proxy_keys import build_targets, cell_choice, codes  # noqa: E402

EXP = "exp45_big_space"; KS = (10, 40, 100, 200); CLIP = 5.0; REPORT = (0.0, 1.0, 10.0, 100.0, 1000.0); AGGS = ("per_query", "per_family_equal_weight")

# (name, loader spec, in_ridge).  Tier 1 = one per model; tier 2 = depth; tier 3 = pooling and deltas.
BLOCKS = [
    ("qwen7b:L21_last", ("hidden", "qwen2.5-7b-instruct", "L21_last"), True),
    ("gte:output", ("rep", None, None), True),
    ("coder1.5b:L22_last", ("hidden", "qwen2.5-coder-1.5b-instruct", "L22_last"), True),
    ("qwen3emb0.6b:L19_last", ("hidden", "qwen3-embedding-0.6b", "L19_last"), True),
    ("e5:L20_mean", ("hidden", "e5-large-v2", "L20_mean"), True),
    ("bge-m3:L18_mean", ("hidden", "bge-m3", "L18_mean"), True),
    ("minilm:L7_mean", ("hidden", "minilm-l12", "L7_mean"), True),
    ("logit_lens", ("hidden", "qwen2.5-7b-instruct", "logit_lens_last"), True),
    ("surprise", ("surprise", None, None), True),
    ("lengths", ("lengths", None, None), True),
    ("qwen7b:L28_last", ("hidden", "qwen2.5-7b-instruct", "L28_last"), True),
    ("qwen7b:L14_last", ("hidden", "qwen2.5-7b-instruct", "L14_last"), True),
    ("gte_own:L21_last", ("hidden", "gte-qwen2-7b", "L21_last"), True),
    ("coder1.5b:L25_last", ("hidden", "qwen2.5-coder-1.5b-instruct", "L25_last"), True),
    ("coder1.5b:L14_last", ("hidden", "qwen2.5-coder-1.5b-instruct", "L14_last"), True),
    ("qwen3emb0.6b:L14_last", ("hidden", "qwen3-embedding-0.6b", "L14_last"), True),
    ("e5:L12_mean", ("hidden", "e5-large-v2", "L12_mean"), True),
    ("bge-m3:L12_mean", ("hidden", "bge-m3", "L12_mean"), True),
    ("minilm:L12_mean", ("hidden", "minilm-l12", "L12_mean"), True),
    ("qwen7b:L21_mean", ("hidden", "qwen2.5-7b-instruct", "L21_mean"), True),
    ("qwen7b:delta_L28_L21", ("delta", "qwen2.5-7b-instruct", ("L28_last", "L21_last")), False),
    ("qwen7b:delta_L21_L14", ("delta", "qwen2.5-7b-instruct", ("L21_last", "L14_last")), False),
    ("coder1.5b:delta_L25_L14", ("delta", "qwen2.5-coder-1.5b-instruct", ("L25_last", "L14_last")), False),
]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--target-dim", type=int, default=40000); ap.add_argument("--tag", default=""); a = ap.parse_args()
    OUT = RESULTS_ROOT / "difficulty_llm" / (EXP + (("_" + a.tag) if a.tag else ""))
    t0 = time.time(); panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_index(); names = [str(s) for s in panel.datasets]; F = len(names)
    onehot = np.eye(F, dtype=np.float32)[fam]
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    Y, pnames, M, _ = build_targets(r, c, panel); P = len(pnames)
    topics_all = np.array([str(t) for t in topic_str(panel)], dtype=object); _, topic_code = np.unique(topics_all, return_inverse=True)
    HROOT = ANALYSIS_ROOT / "features" / PANEL / "prompt"

    def take(arr):
        arr = arr[panel.keep] if panel.keep is not None and arr.shape[0] == len(panel.keep) else arr
        assert arr.shape[0] == n, arr.shape; return np.asarray(arr, dtype=np.float32)

    def load(spec):
        kind, mdl, lay = spec
        if kind == "hidden":
            f = HROOT / "hidden" / mdl / f"{lay}.npy"
            return take(np.load(f, mmap_mode="r")) if f.exists() else None
        if kind == "delta":
            fa, fb = (HROOT / "hidden" / mdl / f"{x}.npy" for x in lay)
            return take(np.load(fa, mmap_mode="r")) - take(np.load(fb, mmap_mode="r")) if fa.exists() and fb.exists() else None
        if kind == "rep":
            return load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)
        if kind == "bow":
            f = HROOT / "hash-bow.npy"
            return take(np.load(f, mmap_mode="r")) if f.exists() else None
        if kind == "surprise":
            f = HROOT / "qwen7b_hidden" / "prompt_logprob_stats.npz"
            return take(np.load(f)["stats"]) if f.exists() else None
        if kind == "lengths":
            return lengths_features(panel)[0].astype(np.float32)
        return None

    blocks = []; width = 0
    for name, spec, in_ridge in BLOCKS:
        if width >= a.target_dim:
            break
        x = load(spec)
        if x is None:
            print(f"  skip {name} (missing)", flush=True); continue
        blocks.append((name, x, in_ridge)); width += x.shape[1]
        print(f"  + {name}: {x.shape[1]} -> total {width}" + ("" if in_ridge else "  [kNN only]"), flush=True)
    manifest = [{"block": nm, "dim": int(x.shape[1]), "in_ridge": bool(ir)} for nm, x, ir in blocks]
    ridge_width = sum(x.shape[1] for _, x, ir in blocks if ir)
    print(f"panel {PANEL}: n={n} F={F} M={M}; big space {width} dims ({len(blocks)} blocks), ridge design {ridge_width} dims ({time.time() - t0:.0f}s)", flush=True)

    per = {}; pq = {}; readout = {}; sel_log = {}; w_block_split0 = {}; space_dims = {}
    base_arr = next((x for nm, x, _ in blocks if nm == "qwen7b:L21_last"), None)
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); ftr, fte = fam[tr], fam[te]
        dis = (r[te].max(1) - r[te].min(1)) > 1e-9
        scaled = []
        for nm, x, in_ridge in blocks:
            xs = np.clip(StandardScaler().fit(x[tr]).transform(x), -CLIP, CLIP).astype(np.float32)
            scaled.append((nm, xs, in_ridge))
        Xr = np.column_stack([xs for _, xs, ir in scaled if ir])
        print(f"seed {seed}: blocks standardised (clip ±{CLIP}), ridge design {Xr.shape} ({time.time() - t0:.0f}s)", flush=True)
        Ptr, Pte = crossfit(Xr, Y, onehot, tr, te, fam, seed); Ptr = np.asarray(Ptr); Pte = np.asarray(Pte)
        del Xr
        print(f"seed {seed}: big-space ridge done ({time.time() - t0:.0f}s)", flush=True)
        if seed == SEEDS[0]:
            Btr, Bte = (None, None) if base_arr is None else crossfit(base_arr, Y, onehot, tr, te, fam, seed)
            for j, nm in enumerate(pnames):
                floor = np.array([Y[tr][ftr == g, j].mean() for g in range(F)])[fte]; sse0 = ((Y[te][:, j] - floor) ** 2).sum()
                readout[nm] = {"big_space": float(1 - ((Y[te][:, j] - Pte[:, j]) ** 2).sum() / max(sse0, 1e-12))}
                if Bte is not None: readout[nm]["single_layer_3584"] = float(1 - ((Y[te][:, j] - np.asarray(Bte)[:, j]) ** 2).sum() / max(sse0, 1e-12))
            print("readout:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in readout.items()}), flush=True)
            del Btr, Bte
        inner_tr, inner_te = split(len(tr), ftr, 100 + seed); inner_tr, inner_te = np.sort(inner_tr), np.sort(inner_te)
        # supervised block weights: how much of the solve count each block alone explains on an inner
        # split of the TRAINING rows (no test row and no outcome of a test row is touched)
        y_diff = Y[:, pnames.index("n_solved")]
        itr, ite = tr[inner_tr], tr[inner_te]; w_block = {}
        for nm, xs, _ in scaled:
            m = Ridge(alpha=100.0).fit(xs[itr], y_diff[itr]); pred = m.predict(xs[ite])
            fl = np.array([y_diff[itr][ftr[inner_tr] == g].mean() for g in range(F)])[ftr[inner_te]]
            r2 = 1 - ((y_diff[ite] - pred) ** 2).sum() / max(((y_diff[ite] - fl) ** 2).sum(), 1e-9)
            w_block[nm] = float(np.sqrt(max(r2, 0.0)))
        if seed == SEEDS[0]:
            w_block_split0 = dict(w_block); print("block weights (sqrt R² on the solve count):", {k: round(v, 3) for k, v in w_block.items()}, flush=True)
        ps = StandardScaler().fit(Ptr); prox = (unit(ps.transform(Ptr)), unit(ps.transform(Pte)))
        gte_f = load_rep(find_rep(PANEL, REP), n, panel.keep).astype(np.float32)

        def build(kind):
            parts = []
            for nm, xs, _ in scaled:
                if kind == "nosem" and nm.startswith("gte"): continue
                d = xs.shape[1]
                if kind == "raw": parts.append(xs)
                elif kind == "rowl2": parts.append(xs / np.maximum(np.linalg.norm(xs, axis=1, keepdims=True), 1e-9))
                elif kind == "weighted": parts.append((xs * (w_block[nm] / np.sqrt(d))).astype(np.float32))
                else: parts.append((xs / np.sqrt(d)).astype(np.float32))
            return np.column_stack(parts)

        nb, nb_in, space_dims = {}, {}, {}
        for kind, label in [("raw", "big_raw"), ("balanced", "big_balanced"), ("rowl2", "big_rowl2"),
                            ("weighted", "big_weighted"), ("nosem", "big_nosemantic")]:
            X = build(kind); space_dims[label] = int(X.shape[1])
            u_tr, u_te = unit(X[tr]), unit(X[te]); del X
            nb[label] = topk(u_te, u_tr, max(KS)); nb_in[label] = topk(u_tr[inner_te], u_tr[inner_tr], max(KS))
            del u_tr, u_te
            print(f"seed {seed}: {label} ({space_dims[label]} dims) neighbours done ({time.time() - t0:.0f}s)", flush=True)
        spaces = {"proxy": prox, "gte": (unit(gte_f[tr]), unit(gte_f[te]))}
        for nm, v in spaces.items():
            nb[nm] = topk(v[1], v[0], max(KS)); nb_in[nm] = topk(v[0][inner_te], v[0][inner_tr], max(KS)); space_dims[nm] = int(v[0].shape[1])
        del scaled
        rhat = (Ptr[:, P:P + M], Pte[:, P:P + M]); chat = (np.exp(Ptr[:, P + M:]), np.exp(Pte[:, P + M:]))
        res = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]
            am_tr = (rhat[0] - lam * chat[0]).argmax(1); am_te = (rhat[1] - lam * chat[1]).argmax(1)
            keys = {"lookup::family": codes([(ftr, fte)]), "lookup::family+topic": codes([(ftr, fte), (topic_code[tr], topic_code[te])]),
                    "lookup::family+predicted_argmax": codes([(ftr, fte), (am_tr, am_te)])}
            choices = {nm: cell_choice(ct, ce, ftr, fte, Utr, F, nc) for nm, (ct, ce, nc) in keys.items()}
            choices["direct_argmax_router"] = am_te
            choices["best_single"] = np.full(len(te), int(Utr.mean(0).argmax())); choices["oracle"] = U[te].argmax(1)
            inner = {}
            for nm in nb:
                for k in KS:
                    ch = Utr[inner_tr][nb_in[nm][:, :k]].mean(1).argmax(1)
                    inner[f"knn::{nm}::{k}"] = float(U[tr[inner_te]][np.arange(len(inner_te)), ch].mean())
                    choices[f"knn::{nm}::{k}"] = Utr[nb[nm][:, :k]].mean(1).argmax(1)
            sel_log.setdefault(str(lam), []).append(max(inner, key=inner.get))
            for nm, ch in choices.items():
                s = summarise(ch, r[te], c[te], lam, fte, dis, F); u = s.pop("_per_query_utility")
                res.setdefault(nm, {})[str(lam)] = s["all"]
                if seed == SEEDS[0] and lam in REPORT: pq[(nm, lam)] = u
            if lam in REPORT: print(f"  seed {seed} lam={lam:g}: best inner {max(inner, key=inner.get)} ({time.time() - t0:.0f}s)", flush=True)
        per[seed] = res; del spaces, nb, nb_in, prox
    methods = list(per[SEEDS[0]])
    table = {str(lam): {nm: {m_: float(np.mean([per[s][nm][str(lam)][m_]["utility"] for s in SEEDS])) for m_ in AGGS} for nm in methods} for lam in LAMBDAS}
    rng = np.random.default_rng(0); _, te0 = split(n, fam, SEEDS[0]); te0 = np.sort(te0); fte0 = fam[te0]
    boots = {}
    for lam in REPORT:
        for nm in ("knn::big_raw::100", "knn::big_balanced::100", "knn::proxy::100", "lookup::family+predicted_argmax"):
            if (nm, lam) in pq:
                boots[f"{nm} - knn::gte::100 @ λ={lam:g}"] = boot(pq[(nm, lam)] - pq[("knn::gte::100", lam)], fte0, F, rng)
                boots[f"{nm} - lookup::family @ λ={lam:g}"] = boot(pq[(nm, lam)] - pq[("lookup::family", lam)], fte0, F, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    out = {**panel.provenance(), "experiment": EXP, "target_dim": a.target_dim, "achieved_dim": int(width), "ridge_dim": int(ridge_width),
           "blocks": manifest, "space_dims": space_dims, "block_weights_split0": w_block_split0, "clip": CLIP, "readout_split0": readout, "selected_per_lambda": sel_log, "by_lambda": table, "bootstrap_split0": boots, "seconds": time.time() - t0}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    md = [f"# {EXP} — a {width}-dimensional space built without any reduction ({panel.name})", "",
          f"{len(blocks)} blocks, {width} dimensions in total, {ridge_width} of them in the ridge design (depth-difference blocks are kNN-only because a linear model cannot use them). Blocks are standardised on the training rows; `big_raw` keeps them at that scale so the widest blocks dominate the distance, `big_balanced` divides each by the square root of its width so every source counts equally. 5 splits, k chosen on an inner split of the training rows.", "",
          "## Blocks", "", "block | dimensions | in ridge", "---|---|---"]
    md += [f"{b['block']} | {b['dim']} | {'yes' if b['in_ridge'] else 'no'}" for b in manifest]
    md += ["", "## Proxy read-out from the big space vs the single 3,584-wide layer (R² beyond the family mean, split 0)", "", "proxy | big space | single layer | Δ", "---|---|---|---"]
    for k, v in sorted(readout.items(), key=lambda kv: -kv[1]["big_space"]):
        s_ = v.get("single_layer_3584"); md.append(f"{k} | {v['big_space']:+.3f} | " + (f"{s_:+.3f} | {v['big_space'] - s_:+.3f}" if s_ is not None else "— | —"))
    for mode in AGGS:
        md += ["", f"## Utility, {mode}", "", "router | " + " | ".join(f"λ={l:g}" for l in REPORT), "---|" + "---|" * len(REPORT)]
        for nm in sorted(methods): md.append(f"{nm} | " + " | ".join(f"{table[str(l)][nm][mode]:.4f}" for l in REPORT))
    md += ["", "## Paired bootstraps, split 0 (per query / equal weight)", "", "comparison | per query | equal weight", "---|---|---"]
    for k_, v in boots.items():
        md.append(f"{k_} | {v['per_query']['point']:+.4f} [{v['per_query']['ci95'][0]:+.4f}, {v['per_query']['ci95'][1]:+.4f}] | {v['per_family_equal_weight']['point']:+.4f} [{v['per_family_equal_weight']['ci95'][0]:+.4f}, {v['per_family_equal_weight']['ci95'][1]:+.4f}]")
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
