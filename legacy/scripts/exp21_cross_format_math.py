"""exp21: is there a format-independent "math" direction? GSM8K vs MMLU-math vs MMLU-non-math in GTE (and bag-of-words).
See experiments/exp21_cross_format_math.yaml."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT  # noqa: E402

EXP = "exp21_cross_format_math"; OUT = RESULTS_ROOT / "geometry_tests" / EXP
K = 10; N_PERM = 200; N_PERM_T3 = 1000; N_BOOT = 1000
CORE = ["mmlu-elementary-mathematics", "mmlu-high-school-mathematics", "mmlu-college-mathematics", "mmlu-abstract-algebra", "mmlu-high-school-statistics"]
WIDE = CORE + ["mmlu-formal-logic", "mmlu-econometrics", "mmlu-high-school-physics", "mmlu-college-physics", "mmlu-conceptual-physics", "mmlu-high-school-chemistry",
               "mmlu-college-chemistry", "mmlu-electrical-engineering", "mmlu-machine-learning", "mmlu-high-school-computer-science", "mmlu-college-computer-science"]


def unit(a): return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def topk_sim(q, m, k, chunk=1024):
    """indices (into m) of the k most cosine-similar rows of m for every row of q; q, m unit."""
    out = np.zeros((len(q), k), dtype=np.int64)
    for s in range(0, len(q), chunk):
        S = q[s:s + chunk] @ m.T; out[s:s + chunk] = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return out


def auc_rows(q, m, pos, chunk=512):
    """per row of q: AUC of cosine similarity as a score for pos (bool over m). Mann-Whitney via ranks."""
    from scipy.stats import rankdata
    n1, n0 = pos.sum(), (~pos).sum(); out = np.zeros(len(q))
    for s in range(0, len(q), chunk):
        S = q[s:s + chunk] @ m.T
        for i, row in enumerate(S):
            rk = rankdata(row); out[s + i] = (rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def main() -> None:
    panel = load_panel(PANEL); n = panel.n
    reps = {"gte": unit(load_rep(find_rep(PANEL, REP), n, panel.keep)),
            "bow": unit(load_rep(find_rep(PANEL, "computed/hash-bow@prompt"), n, panel.keep))}
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str)
    rec = rec[panel.keep] if panel.keep is not None else rec
    subj = np.array([s_.split(".")[0] for s_ in rec]); ds = panel.dataset_ids
    is_m = ds == "mmlu"; mm_idx = np.flatnonzero(is_m); subj_m = subj[mm_idx]
    core = np.isin(subj_m, CORE); wide = np.isin(subj_m, WIDE)
    foreign = {"grade-school-math": ds == "grade-school-math", "mtbench-math": np.array([s_.startswith("mtbench-math.") for s_ in rec]),
               "mtbench-other": (ds == "mtbench") & ~np.array([s_.startswith("mtbench-math.") for s_ in rec]),
               "mbpp": ds == "mbpp", "arc-challenge": ds == "arc-challenge", "winogrande": ds == "winogrande", "hellaswag": ds == "hellaswag"}
    rng = np.random.default_rng(0); res = {"groups": {"mmlu_math_core": CORE, "mmlu_quant_wide": WIDE, "n_core": int(core.sum()), "n_wide": int(wide.sum()), "n_mmlu": int(is_m.sum()),
                                                     "base_rate_core": float(core.mean()), "base_rate_wide": float(wide.mean())}}
    md = [f"# {EXP}", "", f"MMLU-math (core) = {len(CORE)} subjects, {core.sum()} rows = {core.mean():.3f} of MMLU; MMLU-quant (wide) = {len(WIDE)} subjects, {wide.sum()} rows = {wide.mean():.3f}. k = {K}. Null for T1: subject labels permuted among MMLU rows ({N_PERM} perms). CIs: bootstrap over the foreign queries.", ""]
    for rname, X in reps.items():
        R = res.setdefault(rname, {}); Xm = X[mm_idx]
        # ---------------- T1: foreign -> MMLU neighbours
        md += [f"## [{rname}] T1 — nearest MMLU neighbours of foreign-family queries (k = {K})", "", "foreign family | n | share MMLU-math (core) [CI] | null 97.5th | share MMLU-quant (wide) [CI] | null 97.5th | top MMLU subjects among neighbours", "---|---|---|---|---|---|---"]
        for fname, fmask in foreign.items():
            q = X[fmask]; nb = topk_sim(q, Xm, K); sc = core[nb].mean(1); sw = wide[nb].mean(1)
            def ci(v): b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            null_c, null_w = [], []
            for _ in range(N_PERM):
                p_ = rng.permutation(len(subj_m)); null_c.append(core[p_][nb].mean()); null_w.append(wide[p_][nb].mean())
            top = collections.Counter(subj_m[nb].ravel()).most_common(8); tot = nb.size
            R[f"T1|{fname}"] = {"n": int(fmask.sum()), "share_core": float(sc.mean()), "ci_core": ci(sc), "null_core_p975": float(np.percentile(null_c, 97.5)), "p_core": float(np.mean(np.array(null_c) >= sc.mean())),
                                "share_wide": float(sw.mean()), "ci_wide": ci(sw), "null_wide_p975": float(np.percentile(null_w, 97.5)), "p_wide": float(np.mean(np.array(null_w) >= sw.mean())),
                                "top_subjects": [[s_.replace("mmlu-", ""), c_ / tot] for s_, c_ in top]}
            r_ = R[f"T1|{fname}"]
            md.append(f"{fname} | {r_['n']} | {r_['share_core']:.3f} [{r_['ci_core'][0]:.3f}, {r_['ci_core'][1]:.3f}] | {r_['null_core_p975']:.3f} | {r_['share_wide']:.3f} [{r_['ci_wide'][0]:.3f}, {r_['ci_wide'][1]:.3f}] | {r_['null_wide_p975']:.3f} | " + ", ".join(f"{s_} {f:.2f}" for s_, f in r_["top_subjects"][:6]))
        md.append(f"(base rates: core {core.mean():.3f}, wide {wide.mean():.3f})"); md.append("")
        # ---------------- T2: per-GSM8K-query AUC
        q = X[foreign["grade-school-math"]]; auc_c = auc_rows(q, Xm, core); auc_w = auc_rows(q, Xm, wide)
        def cib(v): b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        R["T2"] = {"auc_core_mean": float(auc_c.mean()), "auc_core_ci": cib(auc_c), "auc_core_frac_above_0.5": float((auc_c > 0.5).mean()), "auc_core_p10_p90": [float(np.percentile(auc_c, 10)), float(np.percentile(auc_c, 90))],
                   "auc_wide_mean": float(auc_w.mean()), "auc_wide_ci": cib(auc_w), "auc_wide_frac_above_0.5": float((auc_w > 0.5).mean())}
        md += [f"## [{rname}] T2 — per-GSM8K-query AUC: does similarity to the GSM8K query separate MMLU-math from the rest of MMLU?", "",
               f"core: mean AUC {auc_c.mean():.3f} [{R['T2']['auc_core_ci'][0]:.3f}, {R['T2']['auc_core_ci'][1]:.3f}], 10th–90th pct [{R['T2']['auc_core_p10_p90'][0]:.3f}, {R['T2']['auc_core_p10_p90'][1]:.3f}], {100 * R['T2']['auc_core_frac_above_0.5']:.1f}% of GSM8K queries > 0.5; wide: mean AUC {auc_w.mean():.3f} [{R['T2']['auc_wide_ci'][0]:.3f}, {R['T2']['auc_wide_ci'][1]:.3f}]", ""]
        # ---------------- T3: MMLU -> GSM8K
        g_cent = X[foreign["grade-school-math"]].mean(0); g_cent /= np.linalg.norm(g_cent); cos_g = Xm @ g_cent
        non_m = np.flatnonzero(~is_m); fam_non = ds[non_m]; Xn = X[non_m]
        # rank of the nearest GSM8K row among all non-MMLU rows, per MMLU query (1 = nearest non-MMLU row is a GSM8K row)
        S = np.zeros(len(mm_idx)); is_g_non = fam_non == "grade-school-math"
        nearest_is_gsm = np.zeros(len(mm_idx), dtype=bool); frac_gsm_top10 = np.zeros(len(mm_idx))
        for s in range(0, len(mm_idx), 1024):
            sim = Xm[s:s + 1024] @ Xn.T; top = np.argpartition(-sim, 9, axis=1)[:, :10]
            nearest_is_gsm[s:s + 1024] = is_g_non[np.argmax(sim, axis=1)]; frac_gsm_top10[s:s + 1024] = is_g_non[top].mean(1)
        def contrast(v, lab):
            a, b = v[core], v[~core]; d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2); obs = a.mean() - b.mean()
            null = [(lambda p_: v[p_][:core.sum()].mean() - v[p_][core.sum():].mean())(rng.permutation(len(v))) for _ in range(N_PERM_T3)]
            return {"math": float(a.mean()), "non_math": float(b.mean()), "diff": float(obs), "cohen_d": float(d), "p_perm": float(np.mean(np.abs(null) >= abs(obs)))}
        R["T3"] = {"cos_to_gsm8k_centroid": contrast(cos_g, "cos"), "nearest_non_mmlu_row_is_gsm8k": contrast(nearest_is_gsm.astype(float), "nearest"), "share_gsm8k_in_top10_non_mmlu": contrast(frac_gsm_top10, "top10"),
                   "base_rate_gsm8k_among_non_mmlu": float(is_g_non.mean()),
                   "per_subject_cos_to_gsm8k": {s_.replace("mmlu-", ""): float(cos_g[subj_m == s_].mean()) for s_ in np.unique(subj_m)}}
        md += [f"## [{rname}] T3 — MMLU → GSM8K: MMLU-math vs MMLU-non-math", "", "statistic | MMLU-math | MMLU-non-math | diff | Cohen d | permutation p", "---|---|---|---|---|---"]
        for key, lab in (("cos_to_gsm8k_centroid", "cosine to GSM8K centroid"), ("nearest_non_mmlu_row_is_gsm8k", "nearest non-MMLU row is GSM8K (share)"), ("share_gsm8k_in_top10_non_mmlu", "GSM8K share among 10 nearest non-MMLU rows")):
            t = R["T3"][key]; md.append(f"{lab} | {t['math']:.4f} | {t['non_math']:.4f} | {t['diff']:+.4f} | {t['cohen_d']:+.2f} | {t['p_perm']:.3f}")
        md.append(f"(GSM8K is {is_g_non.mean():.3f} of the non-MMLU rows)")
        ps = sorted(R["T3"]["per_subject_cos_to_gsm8k"].items(), key=lambda kv: -kv[1])
        md += ["", "MMLU subjects by cosine to the GSM8K centroid — top 10: " + ", ".join(f"{s_} {v:.3f}" for s_, v in ps[:10]) + "; bottom 5: " + ", ".join(f"{s_} {v:.3f}" for s_, v in ps[-5:]), ""]
        # ---------------- T4: centroid cosines
        groups = {"mmlu-math": mm_idx[core], "mmlu-non-math": mm_idx[~core], **{f: np.flatnonzero(m) for f, m in foreign.items()}}
        cents = {g: (lambda c_: c_ / np.linalg.norm(c_))(X[i].mean(0)) for g, i in groups.items()}; gl = list(groups)
        R["T4"] = {a: {b: float(cents[a] @ cents[b]) for b in gl} for a in gl}
        md += [f"## [{rname}] T4 — centroid cosines", "", "group | " + " | ".join(gl), "---|" + "---|" * len(gl)]
        md += [f"{a} | " + " | ".join(f"{R['T4'][a][b]:.3f}" for b in gl) for a in gl]; md.append("")
        print(f"{rname} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": EXP, "k": K, **res}, indent=1))
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
