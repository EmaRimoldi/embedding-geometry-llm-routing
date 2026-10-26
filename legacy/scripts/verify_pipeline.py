"""CPU-side verification of the whole pipeline. Prints PASS/FAIL per check and
writes results/verification/cpu.json. Independent of the probe jobs' outputs
except where it re-derives them for comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, PANELS, build_targets, load_panel, text_controls  # noqa: E402
from rb.probes import evaluate, fit_predict_oof, make_folds  # noqa: E402
from rb.report import baseline_rewards  # noqa: E402
from rb.reps import list_reps, load_rep, safe_name  # noqa: E402

OUT = ANALYSIS_ROOT / "results/verification"
report: dict = {"checks": []}


def check(name: str, ok: bool, detail: str = "") -> None:
    report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in PANELS:
        panel = load_panel(name)
        t = build_targets(panel)
        n, m = panel.n, panel.m
        # --- targets
        check(f"{name}: rewards finite where available",
              np.all(np.isfinite(panel.rewards[panel.available])))
        check(f"{name}: reward NaN exactly where unavailable",
              np.array_equal(np.isnan(t.reward), ~panel.available))
        check(f"{name}: every solvable row has a cheapest-sufficient model",
              np.all((t.cheapest_sufficient >= 0) == (t.n_success > 0)) or np.all(t.cheapest_sufficient >= 0),
              f"n_success==0: {int((t.n_success == 0).sum())}, cheapest==-1: {int((t.cheapest_sufficient < 0).sum())}")
        ok_cs = all(np.isclose(t.reward[i, t.cheapest_sufficient[i]], t.oracle_reward[i])
                    for i in np.flatnonzero(t.cheapest_sufficient >= 0)[:2000])
        check(f"{name}: cheapest-sufficient attains oracle reward (first 2000 rows)", ok_cs)
        check(f"{name}: disagreement == (max != min) over available",
              np.array_equal(t.disagreement, (np.nanmax(t.reward, 1) - np.nanmin(t.reward, 1)) > 1e-9))
        check(f"{name}: hard subset ⊂ disagreement and solvable",
              np.all(t.disagreement[t.hard]) and np.all(t.n_success[t.hard] > 0),
              f"hard={int(t.hard.sum())}")
        check(f"{name}: origin_query contained in prompt (whitespace-insensitive, sample 500)",
              np.mean([("".join(o.split()) in "".join(p.split())) for o, p in
                       zip(panel.origin_queries[:500], panel.prompts[:500])]) > 0.98)
        # --- folds
        for split in ("random", "lodo", "within"):
            folds = make_folds(split, t)
            cover = np.zeros(n, dtype=int)
            leak = 0
            for tr, te in folds:
                cover[te] += 1
                leak += len(np.intersect1d(tr, te))
            check(f"{name}/{split}: test folds partition rows, no train/test overlap",
                  cover.max() == 1 and leak == 0 and (cover.min() == 1 or split == "within"),
                  f"uncovered={int((cover == 0).sum())} folds={len(folds)}")
            if split == "lodo":
                check(f"{name}/lodo: held-out dataset absent from train",
                      all(len(set(t.dataset_index[tr]) & set(t.dataset_index[te])) == 0 for tr, te in folds))
            if split == "within":
                check(f"{name}/within: every fold inside one dataset",
                      all(len(set(t.dataset_index[np.concatenate([tr, te])])) == 1 for tr, te in folds))
        # --- features
        for spec in list_reps(name):
            try:
                x = load_rep(spec, n)
            except Exception as exc:  # noqa: BLE001
                check(f"{name}: load {spec.name}", False, str(exc))
                continue
            norms = np.linalg.norm(x, axis=1)
            texts = panel.text(spec.text_field)
            uniq_rows = len(np.unique(np.round(x[:, :64], 4), axis=0))
            uniq_text = len(set(map(str, texts)))
            meta = {}
            mp = spec.path.with_suffix(".npy.json")
            if mp.exists():
                meta = json.loads(mp.read_text())
            detail = (f"dim={x.shape[1]} norm={norms.mean():.3f}±{norms.std():.3f} zero_rows={int((norms < 1e-8).sum())} "
                      f"uniq_rows(first64dims)={uniq_rows}/{uniq_text} uniq_texts; "
                      f"truncated={meta.get('texts_truncated', 'n/a')} tok_max={meta.get('token_len_max', 'n/a')} "
                      f"trunc_side={meta.get('truncation_side', 'n/a')} normalized={meta.get('normalized', spec.normalized)}")
            ok = np.all(np.isfinite(x)) and (norms > 1e-8).all() and uniq_rows >= 0.95 * uniq_text
            check(f"{name}: {spec.name}", ok, detail)
            if spec.normalized:
                check(f"{name}: {spec.name} unit norm", np.allclose(norms, 1, atol=1e-3))
        # --- one-hot ridge reproduces domain-best per query
        res = fit_predict_oof(np.zeros((n, 1), np.float32), t, "onehot", "ridge", "random")
        choice = np.argmax(np.where(~np.isnan(t.reward), res["pred"], -np.inf), 1)
        agree = np.mean(choice == res["domain_best"])
        check(f"{name}: onehot+ridge choice == domain-best choice", agree > 0.99, f"agreement={agree:.4f}")
        # --- stored metric reproducibility (one stored unit)
        for rep_dir in sorted((ANALYSIS_ROOT / "results/probes" / name).glob("*")):
            f = rep_dir / "raw__ridge__random.json"
            z = rep_dir / "raw__ridge__random.npz"
            if f.exists() and z.exists():
                d = json.loads(f.read_text())
                q = np.load(z)
                base = baseline_rewards(t, "random")
                mk = t.disagreement & q["covered"]
                g = (q["router_reward"][mk].mean() - base["domain_best"][mk].mean()) / (t.oracle_reward[mk].mean() - base["domain_best"][mk].mean())
                check(f"{name}: stored gap_closed reproducible from per-query arrays ({rep_dir.name})",
                      abs(g - d["subsets"]["disagreement"]["gap_closed_vs_domain_best"]) < 1e-6,
                      f"recomputed={g:.4f} stored={d['subsets']['disagreement']['gap_closed_vs_domain_best']:.4f}")
                break
        # --- shuffle control (leakage test) on a cheap rep
        cheap = next((s for s in list_reps(name) if "minilm-l6-256" in s.name), None)
        if cheap is not None:
            x = load_rep(cheap, n)
            rng = np.random.default_rng(0)
            xs = x.copy()
            for g in np.unique(t.dataset_index):
                rows = np.flatnonzero(t.dataset_index == g)
                xs[rows] = x[rng.permutation(rows)]
            cost = np.where(panel.available, panel.costs, np.nan)
            controls, _ = text_controls(panel.text(cheap.text_field))
            for split in ("random", "within"):
                ev_real = evaluate(fit_predict_oof(x, t, "raw", "ridge", split, controls), t, cost)
                ev_shuf = evaluate(fit_predict_oof(xs, t, "raw", "ridge", split, controls), t, cost)
                gr = ev_real["subsets"]["disagreement"]["gap_closed_vs_domain_best"]
                gs = ev_shuf["subsets"]["disagreement"]["gap_closed_vs_domain_best"]
                check(f"{name}/{split}: within-dataset shuffled {cheap.name} gives G≈0 (leakage control)",
                      abs(gs) < 0.03, f"real G={gr:+.4f} shuffled G={gs:+.4f}")
        # --- controls features sanity
        c, names = text_controls(panel.origin_queries)
        check(f"{name}: surface controls finite, non-constant", np.all(np.isfinite(c)) and (c.std(0) > 0).all(),
              f"std={np.round(c.std(0), 3).tolist()}")
    (OUT / "cpu.json").write_text(json.dumps(report, indent=1))
    n_fail = sum(not c["ok"] for c in report["checks"])
    print(f"\n{len(report['checks'])} checks, {n_fail} failed")


if __name__ == "__main__":
    main()
