"""exp26 step 0: how to subtract the template's cost — per-model input price from the release's own accounting.

For each model m, OLS of recorded cost on prompt tokens (Qwen tokenizer) restricted to the four single-letter-answer families
(MMLU, ARC, HellaSwag, Winogrande), where the answer cost is a constant: slope b_m = input price per token, intercept a_m =
answer + any per-request fee.  Template tokens T_f = tokens of the stripped prefix + suffix (family constant).
Corrected cost c'(i, m) = c(i, m) - b_m * T_f(i).  Writes results/deformatting/panel/{cost_model.json, costs_corrected.npy}
and results/deformatting/cost_study.md.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from deformat import PANEL_DIR, deformat  # noqa: E402

MC = ("mmlu", "arc-challenge", "hellaswag", "winogrande")


def ols(xv, yv):
    A = np.column_stack([np.ones_like(xv), xv]); coef, *_ = np.linalg.lstsq(A, yv, rcond=None); pred = A @ coef
    r2 = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum(); return float(coef[0]), float(coef[1]), float(r2)


def main() -> None:
    panel = load_panel("routerbench-0shot"); n = panel.n; ds = panel.dataset_ids; fams = [str(f) for f in panel.datasets]; models = [str(m) for m in panel.models]
    c = np.nan_to_num(panel.costs.astype(np.float64), nan=0.0)
    tok = np.load(ANALYSIS_ROOT / "features/routerbench-0shot/prompt/lengths.npz")["qwen_tokens"]; tok = (tok[panel.keep] if panel.keep is not None else tok).astype(np.float64)
    prompts = [str(t) for t in panel.prompts]; core, rule = deformat(prompts, ds, fams)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(RUNTIME_ROOT / "artifacts/models/gte-Qwen2-7B-instruct"))
    T = {}
    for f in fams:
        T[f] = len(tk(rule[f]["prefix"], add_special_tokens=False)["input_ids"]) + len(tk(rule[f]["suffix"], add_special_tokens=False)["input_ids"])
    # empirical check of T_f: prompt tokens minus core tokens (sample of 300 per family)
    rng = np.random.default_rng(0); T_emp = {}
    for f in fams:
        idx = rng.choice(np.flatnonzero(ds == f), min(300, int((ds == f).sum())), replace=False)
        T_emp[f] = float(np.mean([len(tk(prompts[j], add_special_tokens=False)["input_ids"]) - len(tk(core[j], add_special_tokens=False)["input_ids"]) for j in idx]))
    Tf = np.array([T[str(f)] for f in ds], dtype=np.float64)
    mc = np.isin(ds, MC); out = {"template_tokens": T, "template_tokens_empirical_check": T_emp, "models": {}}
    md = ["# Cost study — subtracting the template's cost", "", f"Template tokens per family (Qwen tokenizer; prefix + suffix): " + ", ".join(f"{f} {T[f]}" for f in fams),
          f"Empirical check (prompt tokens − core tokens, 300-row samples): " + ", ".join(f"{f} {T_emp[f]:.1f}" for f in fams), "",
          "## Per-model regressions of recorded cost on prompt tokens", "", "Fit on the four single-letter-answer families (answer cost constant). Slope = input price per Qwen token; intercept = answer + fee. Also shown: slope on all families and on GSM8K alone (contaminated by answer length), and per-MC-family slopes (consistency).", "",
          "model | slope $/1M tokens (MC families) | intercept $ | R² | slope all families | slope GSM8K | per-MC-family slopes | min corrected cost (GSM8K) | share of cost removed: GSM8K / MMLU / ARC / HellaSwag / Winogrande", "---|---|---|---|---|---|---|---|---"]
    corrected = np.zeros_like(c)
    for mi, m in enumerate(models):
        a, b, r2 = ols(tok[mc], c[mc, mi]); _, b_all, _ = ols(tok, c[:, mi]); _, b_g, _ = ols(tok[ds == "grade-school-math"], c[ds == "grade-school-math", mi])
        per = {f: ols(tok[ds == f], c[ds == f, mi])[1] * 1e6 for f in MC}
        corrected[:, mi] = c[:, mi] - b * Tf
        share = {f: float(np.mean(b * Tf[ds == f] / np.maximum(c[ds == f, mi], 1e-12))) for f in ("grade-school-math", "mmlu", "arc-challenge", "hellaswag", "winogrande")}
        mn = {f: float(corrected[ds == f, mi].min()) for f in fams}
        out["models"][m] = {"slope_per_token": b, "intercept": a, "r2_mc": r2, "slope_all": b_all, "slope_gsm8k": b_g, "per_mc_family_slope_per_1M": per, "min_corrected_cost_by_family": mn, "share_removed": share,
                            "n_negative_cells": int((corrected[:, mi] < 0).sum())}
        md.append(f"{m.split('/')[-1]} | {b * 1e6:.3f} | {a:.2e} | {r2:.3f} | {b_all * 1e6:.3f} | {b_g * 1e6:.3f} | " + ", ".join(f"{f[:4]} {v:.2f}" for f, v in per.items()) +
                  f" | {mn['grade-school-math']:.2e} | " + " / ".join(f"{share[f]:.2f}" for f in ("grade-school-math", "mmlu", "arc-challenge", "hellaswag", "winogrande")))
    neg = int((corrected < 0).sum()); md += ["", f"Negative corrected cells: {neg} of {corrected.size}."]
    # published price table, if the project carries one (read-only lookup)
    found = []
    for pth in RUNTIME_ROOT.rglob("*.py"):
        if "venvs" in str(pth) or "experiments4" in str(pth) or "huggingface" in str(pth): continue
        try: s = pth.read_text(errors="ignore")
        except Exception: continue
        if "get_prompt_token_cost" in s or "prompt_token_cost" in s: found.append(str(pth))
    out["price_table_candidates"] = found[:10]; md += ["", "Project files mentioning a prompt-token price table: " + (", ".join(found[:10]) if found else "none found"), ""]
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PANEL_DIR / "costs_corrected.npy", corrected.astype(np.float64)); (PANEL_DIR / "cost_model.json").write_text(json.dumps({**panel.provenance(), "rule": "c_corr = c - slope_m(MC families) * T_f", **out}, indent=1))
    (HERE / "cost_study.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
