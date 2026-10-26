"""exp39: how much per-query difficulty is there to recover, independently of any embedding?
Difficulty = number of pool models solving the query (r >= 0.5).  Descriptive, whole panel, no fitting:
  * variance of the solve count; share explained by the family mean (between-family), by the
    (family, topic) cell mean, and what remains within cells (the part an embedding could add);
  * per family: rows, mean, sd, share solved by none / by all, sd within the family's topic cells,
    number of cells, share of the family's variance explained by its cells;
  * per-model version: mean within-family Bernoulli variance p(1-p) of each model's success, i.e.
    how unpredictable a model's outcome is once the family is known, and the same within cells;
  * routing-relevant version: reward of the family's best model vs the oracle within each family
    (the reward gap that per-query information could close), per query and equal weight.
Topic cells: RouterBench = MMLU subjects / HellaSwag sources (others = family); LLMRouterBench =
released topics and, separately, the LLM labels of exp33 (every family).  Writes
results[_root]/difficulty_llm/exp39_difficulty_variance/result.{json,md}."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, SUCCESS_THRESHOLD, load_panel  # noqa: E402
from rb.panelctx import IS_ROUTERBENCH, PANEL, RESULTS_ROOT, topic_str  # noqa: E402

OUT = RESULTS_ROOT / "difficulty_llm" / "exp39_difficulty_variance"


def var_within(y, groups):
    """pooled within-group variance (sum of squared deviations from the group mean / n)."""
    sse = 0.0
    for g in np.unique(groups):
        m = groups == g; sse += float(((y[m] - y[m].mean()) ** 2).sum())
    return sse / len(y)


def main() -> None:
    panel = load_panel(PANEL); n = panel.n; fam = panel.dataset_ids; names = [str(s) for s in panel.datasets]; M = panel.m
    r = np.nan_to_num(panel.rewards.astype(np.float64), nan=0.0); S = (r >= SUCCESS_THRESHOLD); y = S.sum(1).astype(np.float64)
    cells = {"released_topics": np.array([str(t) for t in topic_str(panel)], dtype=object)}
    if not IS_ROUTERBENCH:
        T = json.load(open(ANALYSIS_ROOT / "llmrouterbench" / "topics_llm" / f"{PANEL}.json")); llm = np.array(T["topic"], dtype=object)
        cells["llm_topics"] = llm[panel.keep] if panel.keep is not None and T["rows"] == len(panel.keep) else llm
    tot = float(y.var()); wf = var_within(y, fam)
    res = {"n": n, "models": M, "var_total": tot, "sd_total": float(np.sqrt(tot)), "var_within_family": wf, "share_between_family": 1 - wf / tot, "share_within_family": wf / tot, "cells": {}}
    for cname, cl in cells.items():
        wc = var_within(y, cl); res["cells"][cname] = {"n_cells": int(len(np.unique(cl))), "var_within_cell": wc, "share_within_cell": wc / tot, "share_explained_by_cells_beyond_family": (wf - wc) / tot}
    per = {}
    for f in names:
        m = fam == f; yf = y[m]; d = {"rows": int(m.sum()), "mean": float(yf.mean()), "sd": float(yf.std()), "solved_by_none": float((yf == 0).mean()), "solved_by_all": float((yf == M).mean()),
                                     "q10_q50_q90": [float(np.quantile(yf, q)) for q in (0.1, 0.5, 0.9)]}
        for cname, cl in cells.items():
            sub = cl[m]; k = len(np.unique(sub)); wv = var_within(yf, sub); d[cname] = {"n_cells": int(k), "sd_within_cells": float(np.sqrt(wv)), "share_of_family_variance_explained_by_cells": (1 - wv / yf.var()) if yf.var() > 0 else 0.0}
        # per-model unpredictability within the family and the routing-relevant gap
        p = S[m].mean(0); d["mean_model_bernoulli_var_within_family"] = float((p * (1 - p)).mean())
        best = int(r[m].mean(0).argmax()); d["best_model_reward"] = float(r[m, best].mean()); d["oracle_reward"] = float(r[m].max(1).mean()); d["reward_gap_oracle_minus_best"] = d["oracle_reward"] - d["best_model_reward"]
        d["share_queries_best_fails_some_model_solves"] = float(((~S[m, best]) & S[m].any(1)).mean())
        per[f] = d
    res["per_family"] = per
    gap_pq = float(np.mean([per[f]["reward_gap_oracle_minus_best"] * per[f]["rows"] for f in names]) / n * len(names)) if False else float(sum(per[f]["reward_gap_oracle_minus_best"] * per[f]["rows"] for f in names) / n)
    res["reward_gap_per_query"] = gap_pq; res["reward_gap_equal_weight"] = float(np.mean([per[f]["reward_gap_oracle_minus_best"] for f in names]))
    res["share_best_fails_some_solves_per_query"] = float(sum(per[f]["share_queries_best_fails_some_model_solves"] * per[f]["rows"] for f in names) / n)
    # per-model Bernoulli variance within family vs within cells, whole panel
    pm = {}
    for cname, cl in list(cells.items()) + [("family", fam)]:
        v = np.zeros(M)
        for g in np.unique(cl):
            m = cl == g; p = S[m].mean(0); v += (p * (1 - p)) * m.sum()
        pm[cname] = float((v / n).mean())
    res["mean_model_bernoulli_var"] = {"unconditional": float((S.mean(0) * (1 - S.mean(0))).mean()), **pm}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "result.json").write_text(json.dumps({**panel.provenance(), "experiment": "exp39_difficulty_variance", **res}, indent=1))
    md = [f"# exp39 — how much per-query difficulty exists, independently of any embedding ({panel.name})", "",
          f"Difficulty = number of the {M} pool models solving the query. Whole panel, descriptive, no fitting.", "",
          "## Variance decomposition of the solve count", "", "quantity | value", "---|---",
          f"total variance (sd) | {tot:.2f} ({np.sqrt(tot):.2f})", f"share explained by the family mean | {res['share_between_family']:.1%}", f"share within families | {res['share_within_family']:.1%}"]
    for cname, d in res["cells"].items():
        md.append(f"share within (family, topic) cells — {cname} ({d['n_cells']} cells) | {d['share_within_cell']:.1%} (cells add {d['share_explained_by_cells_beyond_family']:.1%} beyond the family)")
    md += ["", "## Per family", "", "family | rows | mean solved | sd | solved by none | by all | q10 / q50 / q90 | " + " | ".join(f"{c}: cells, sd within, % of family var explained" for c in cells) + " | mean p(1−p) of a model | best-model reward | oracle reward | gap | best fails & some model solves",
           "---|" + "---|" * (11 + len(cells))]
    for f in names:
        d = per[f]; cc = " | ".join(f"{d[c]['n_cells']}, {d[c]['sd_within_cells']:.2f}, {d[c]['share_of_family_variance_explained_by_cells']:.1%}" for c in cells)
        md.append(f"{f} | {d['rows']} | {d['mean']:.2f} | {d['sd']:.2f} | {d['solved_by_none']:.2f} | {d['solved_by_all']:.2f} | {' / '.join(f'{q:.0f}' for q in d['q10_q50_q90'])} | {cc} | {d['mean_model_bernoulli_var_within_family']:.3f} | {d['best_model_reward']:.3f} | {d['oracle_reward']:.3f} | {d['reward_gap_oracle_minus_best']:.3f} | {d['share_queries_best_fails_some_model_solves']:.2f}")
    md += ["", f"Reward gap oracle − family's best model: {res['reward_gap_per_query']:.3f} per query, {res['reward_gap_equal_weight']:.3f} equal weight over families; share of queries where the family's best model fails but some model solves: {res['share_best_fails_some_solves_per_query']:.2f} per query.", "",
           "## Per-model unpredictability (mean over models of the Bernoulli variance p(1−p) of success)", "", "conditioning | mean p(1−p)", "---|---"] + [f"{k} | {v:.3f}" for k, v in res["mean_model_bernoulli_var"].items()]
    (OUT / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
