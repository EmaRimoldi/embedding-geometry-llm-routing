"""exp16: attribute kNN's within-HellaSwag advantage — source (hidden sub-family), length, or per-query signal.

See experiments/exp16_hellaswag_check.yaml.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.probes.sweep import LAMBDAS, SEEDS, split  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

EXP = "exp16_hellaswag_check"
KS = (40, 100); DRAWS = 5; MIN_CELL = 50
REPORT_LAMS = (0.0, 10.0, 50.0, 100.0, 500.0, 1000.0)


def quartile_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    q = np.zeros(len(values), dtype=np.int64)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 8:
            edges = np.quantile(values[m], [0.25, 0.5, 0.75]); q[m] = np.searchsorted(edges, values[m], side="right")
    return q


def main() -> None:
    panel = load_panel("routerbench-0shot"); spec = find_rep("routerbench-0shot", "computed/gte-qwen2-7b@prompt")
    x = load_rep(spec, panel.n, panel.keep); u = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); c = np.nan_to_num(panel.costs, nan=0.0)
    fam = panel.dataset_index(); names = panel.datasets; F = len(names); n = panel.n; M = panel.m
    tokens = np.load(ANALYSIS_ROOT / "features/routerbench-0shot/prompt/lengths.npz")["qwen_tokens"]
    tokens = tokens[panel.keep] if panel.keep is not None else tokens        # lengths were computed on the full release
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str)
    rec = rec[panel.keep] if panel.keep is not None else rec
    subj = np.array([s_.split(".")[0] for s_ in rec])
    is_h = panel.dataset_ids == "hellaswag"
    src = np.array(["wikihow" if re.search(r"\[(header|title|step|substeps)\]", str(t)) else "activitynet" for t in panel.prompts])
    # cell definitions (integer codes over the whole panel; only HellaSwag / MMLU rows are scored)
    def codes(labels):
        u_, inv = np.unique(labels, return_inverse=True); return inv
    cells = {
        "family": codes(panel.dataset_ids),
        "source": codes(np.where(is_h, "hellaswag/" + src, panel.dataset_ids)),
        "family x length-q": None, "source x length-q": None,
        "subject": codes(subj), "subject x length-q": None,
    }
    lq_fam = quartile_within(tokens, cells["family"]); cells["family x length-q"] = codes(np.char.add(cells["family"].astype(str), np.char.add("/", lq_fam.astype(str))))
    lq_src = quartile_within(tokens, cells["source"]); cells["source x length-q"] = codes(np.char.add(cells["source"].astype(str), np.char.add("/", lq_src.astype(str))))
    lq_sub = quartile_within(tokens, cells["subject"]); cells["subject x length-q"] = codes(np.char.add(cells["subject"].astype(str), np.char.add("/", lq_sub.astype(str))))
    scopes = {"hellaswag": ("family", "source", "family x length-q", "source x length-q"), "mmlu": ("family", "subject", "family x length-q", "subject x length-q")}

    per_seed = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr, te = split(n, fam, seed); tr = np.sort(tr); te = np.sort(te)
        kmax = max(KS); nb = np.zeros((len(te), kmax), dtype=np.int64)
        for s0 in range(0, len(te), 2048):
            e0 = min(len(te), s0 + 2048); sims = u[te[s0:e0]] @ u[tr].T; top = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
            nb[s0:e0] = tr[np.take_along_axis(top, np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1), axis=1)]
        # random sets per cell definition (λ-independent), capped
        rand = {}
        for cname, cell in cells.items():
            ctr, cte = cell[tr], cell[te]; sets = np.zeros((DRAWS, len(te), kmax), dtype=np.int64); cnt = np.zeros(len(te), dtype=np.int64)
            for g in np.unique(cte):
                trg = tr[ctr == g]; teg = np.flatnonzero(cte == g)
                if len(trg) == 0: continue
                kk = min(kmax, len(trg)); cnt[teg] = kk
                for d in range(DRAWS):
                    keys = rng.random((len(teg), len(trg))); sets[d, teg, :kk] = trg[np.argpartition(keys, kk - 1, axis=1)[:, :kk]] if kk < len(trg) else trg[np.argsort(keys, axis=1)]
            rand[cname] = (sets, cnt)
        out = {}
        for lam in LAMBDAS:
            U = r - lam * c; Utr = U[tr]; Ute = U[te]
            fam_table = np.array([Utr[fam[tr] == g].mean(0).argmax() for g in range(F)])
            def lookup(cname):
                cell = cells[cname]; ctr, cte = cell[tr], cell[te]; ch = fam_table[fam[te]].copy()
                for g in np.unique(cte):
                    rows = ctr == g
                    if rows.sum() >= MIN_CELL: ch[cte == g] = Utr[rows].mean(0).argmax()
                return ch
            def rand_route(cname, k):
                sets, cnt = rand[cname]; kk = np.minimum(k, cnt); m = np.arange(k)[None, :] < kk[:, None]; outs = []
                for d in range(DRAWS):
                    mu = (U[sets[d, :, :k]] * m[:, :, None]).sum(1) / kk[:, None]; outs.append(mu.argmax(1))
                return outs
            choices = {f"knn-{k}": nb[:, :k] for k in KS}
            res_l = {}
            for scope, cnames in scopes.items():
                mask = (panel.dataset_ids[te] == scope)
                row = {}
                for k in KS:
                    ch = U[nb[:, :k]].mean(1).argmax(1); row[f"knn-{k}"] = float(Ute[mask][np.arange(mask.sum()), ch[mask]].mean())
                for cname in cnames:
                    ch = lookup(cname); row[f"lookup:{cname}"] = float(Ute[mask][np.arange(mask.sum()), ch[mask]].mean())
                    for k in KS:
                        vals = [Ute[mask][np.arange(mask.sum()), chd[mask]].mean() for chd in rand_route(cname, k)]; row[f"random-{k}:{cname}"] = float(np.mean(vals))
                row["oracle"] = float(Ute[mask].max(1).mean())
                if scope == "hellaswag" and lam == 100.0 and seed == SEEDS[0]:
                    ch = U[nb[:, :100]].mean(1).argmax(1); pick = {}
                    for s_ in ("wikihow", "activitynet"):
                        mm = mask & (src[te] == s_); share = np.bincount(ch[mm], minlength=M) / mm.sum()
                        pick[s_] = {"knn-100 top": {panel.models[j][:20]: round(float(share[j]), 2) for j in np.argsort(-share)[:3]},
                                    "lookup:source picks": panel.models[int(np.bincount(lookup("source")[mm], minlength=M).argmax())][:20],
                                    "lookup:family picks": panel.models[int(fam_table[names.index("hellaswag")])][:20]}
                    row["_picks_by_source_seed0"] = pick
                res_l[scope] = row
            out[str(lam)] = res_l
        per_seed[seed] = out; print(f"seed {seed} done", flush=True)

    def agg(scope, lam, key):
        v = [per_seed[s][str(lam)][scope][key] for s in SEEDS]; return float(np.mean(v)), float(min(v)), float(max(v))
    keys = {scope: [k for k in per_seed[SEEDS[0]]["0.0"][scope] if not k.startswith("_")] for scope in scopes}
    result = {**panel.provenance(), "experiment": EXP, "lambdas": LAMBDAS, "seeds": list(SEEDS), "min_cell": MIN_CELL,
              "source_counts": {"wikihow": int((src[is_h] == "wikihow").sum()), "activitynet": int((src[is_h] == "activitynet").sum())},
              "by_scope": {scope: {str(lam): {k: agg(scope, lam, k) for k in keys[scope]} for lam in LAMBDAS} for scope in scopes},
              "picks_by_source_lambda100_seed0": per_seed[SEEDS[0]]["100.0"]["hellaswag"].get("_picks_by_source_seed0")}
    md = [f"# {EXP}", "", f"HellaSwag test rows only (then MMLU as contrast). 5 stratified 70/30 splits; mean over splits (ranges in result.json). Source = WikiHow (markers) {result['source_counts']['wikihow']} rows / ActivityNet {result['source_counts']['activitynet']} rows. Length quartiles within the refined cell. Lookups use all training rows of the cell (family fallback below {MIN_CELL}).", ""]
    for scope in scopes:
        md += [f"## {scope}: realised utility per query", "", "λ | " + " | ".join(keys[scope]), "---|" + "---|" * len(keys[scope])]
        for lam in REPORT_LAMS:
            md.append(f"{lam:g} | " + " | ".join(f"{agg(scope, lam, k)[0]:.4f}" for k in keys[scope]))
        md.append("")
    md += ["## HellaSwag, λ = 100, split 0: which model each router picks, per source", "", "```", json.dumps(result["picks_by_source_lambda100_seed0"], indent=1), "```"]
    d = ANALYSIS_ROOT / "results" / "outcome_locality" / EXP; d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(result, indent=1)); (d / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
