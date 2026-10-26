#!/usr/bin/env python
"""Summarise exp76 decisions.jsonl: utility (per query / equal weight), accuracy, cost,
decision-source shares, accuracy by source, counterfactual lookup accuracy on reasoning
decisions, and the deterministic references.  Writes summary.json, result.json and result.md
next to it.

Rows are deduplicated by query_id keeping the LAST occurrence before anything is counted: a
query that errored and was retried on a later resume appears twice in decisions.jsonl and
must count once (as its final outcome), otherwise both the error count and every mean are
wrong."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import load_panel                                          # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT                             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); a = ap.parse_args()
    d = RESULTS_ROOT / "oracle_reasoning" / "exp76_lookup_reasoning" / a.tag
    raw = [json.loads(l) for l in open(d / "decisions.jsonl") if l.strip()]
    last = {}
    for row in raw:                                     # last row per query wins (retry after an error)
        last[str(row.get("query_id"))] = row
    rows = list(last.values()); n_dup = len(raw) - len(rows)
    df = pd.DataFrame([r for r in rows if "error" not in r]); n_err = sum("error" in r for r in rows)
    if n_dup:
        print(f"[exp76-analyze] {n_dup} superseded row(s) dropped ({len(raw)} lines -> {len(rows)} queries)")
    if df.empty:
        raise SystemExit("no decisions")
    panel = load_panel(PANEL); qfam = dict(zip(map(str, panel.query_ids), panel.dataset_ids)); df["family"] = df.query_id.map(qfam)
    def both(col):
        return {"per_query": float(df[col].mean()), "equal_weight": float(df.groupby("family")[col].mean().mean())}
    df["acc_chosen"] = df.reward_chosen >= 0.5
    s = {"n": int(len(df)), "errors": n_err, "utility": {k: both(k) for k in ("utility_chosen", "utility_knn", "utility_cell", "utility_oracle")},
         "accuracy_chosen": both("acc_chosen"), "cost_chosen": both("cost_chosen"),
         "source_share": df.decision_source.value_counts(normalize=True).to_dict(),
         "accuracy_by_source": df.groupby("decision_source").acc_chosen.mean().to_dict(),
         "utility_by_source": df.groupby("decision_source").utility_chosen.mean().to_dict()}
    rs = df[(df.decision_source == "reasoning") & df.utility_lookup_choice.notna()]
    s["reasoning_vs_its_lookup_choice"] = {"n": int(len(rs)), "utility_reasoning": float(rs.utility_chosen.mean()) if len(rs) else None, "utility_lookup_choice": float(rs.utility_lookup_choice.mean()) if len(rs) else None}
    s["tokens"] = {k: int(sum(r.get("usage", {}).get(k, 0) for r in rows)) for k in ("input_tokens", "output_tokens")}
    s["superseded_rows_dropped"] = int(n_dup)
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    cfg = json.loads((d / "settings.json").read_text()) if (d / "settings.json").exists() else {}
    rep = {**cfg, "experiment": "exp76_lookup_reasoning", "representations": ["computed/gte-qwen2-7b@prompt"],
           "settings": {k: cfg.get(k) for k in ("backend", "model", "mode", "lambda", "k", "max_tokens", "lofo", "n_test")},
           "metrics": s}
    (d / "result.json").write_text(json.dumps(rep, indent=2, default=float))
    md = [f"# exp76 — {a.tag} ({PANEL})", "", f"n={s['n']} errors={n_err}" + (f" (superseded rows dropped: {n_dup})" if n_dup else ""), "",
          "arm | utility per query | equal weight", "---|---|---"]
    for k, v in s["utility"].items(): md.append(f"{k} | {v['per_query']:.4f} | {v['equal_weight']:.4f}")
    md += ["", f"source share: {s['source_share']}", f"accuracy by source: {s['accuracy_by_source']}", f"reasoning vs its own lookup choice: {s['reasoning_vs_its_lookup_choice']}", f"tokens: {s['tokens']}", "",
           "`utility_knn` is the kNN-" + str(cfg.get("k", "?")) + " counterfactual over the rows actually retrieved into this "
           "prompt, NOT the kNN-100 router reported in the paper: it is what the lookup table in front of the model implies, "
           "not a routing baseline.  The reference arms (kNN-100, cell lookup, best single model) come from the exp44/exp36 "
           "tables, not from this file."]
    (d / "result.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
