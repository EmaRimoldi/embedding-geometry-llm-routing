#!/usr/bin/env python
"""Table 5 (tab:argmax) as a LaTeX fragment from exp44 result.json (both panels): every
value is a five-split mean; split-0 bootstrap intervals are written by hand in the
manuscript from result.md. Writes E4/reports/tab_argmax.tex and copies it to
OL/tables/tab_argmax.tex (destination overridable with $OL_TABLES).

Ruling 1 (controller): Table 3 (the main-text headline table) is already hand-derived from
the campaign's own lambda-rescaled table, so this script produces ONLY tab_argmax.tex
(Table 5). tab_headline_revised.tex is intentionally NOT produced."""
from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
# Manuscript tables directory, overridable like _style.OL_FIGURES's OL_FIGURES variable.
OL_TABLES = Path(os.environ.get("OL_TABLES", Path.home() / "paper/tables"))
SRC = {"RouterBench": ROOT / "results" / "difficulty_llm" / "exp44_argmax_key_null" / "result.json",
       "LLMRouterBench": ROOT / "llmrouterbench" / "results" / "performance-cost" / "difficulty_llm" / "exp44_argmax_key_null" / "result.json"}
ARG = [("family+topic", "Family/topic lookup"),
       ("family+predicted_argmax", "Family + predicted best model"),
       ("DEPLOY::predfamily+predicted_argmax", "\\quad with predicted family"),
       ("direct_argmax_router", "Direct argmax, no lookup"),
       ("CEILING::family+true_argmax", "Ceiling: family + true best model"),
       ("oracle", "Oracle")]
LAMS = ["0.0", "100.0"]


def table(rows, label, caption, agg_pairs):
    out = ["\\begin{table}[t]", "\\centering\\small", "\\begin{tabular}{ll" + "r" * (len(LAMS) * len(agg_pairs)) + "}", "\\toprule",
           "Panel & Router & " + " & ".join(f"{a} $\\lam={float(l):g}$" for a, _ in agg_pairs for l in LAMS) + " \\\\", "\\midrule"]
    for pn, p in SRC.items():
        if not p.exists():
            print(f"skipping {pn}: {p} not found")
            continue
        d = json.load(open(p))["by_lambda"]
        for i, (key, name) in enumerate(rows):
            cells = " & ".join(f"{d[l][key][agg]:.3f}" for _, agg in agg_pairs for l in LAMS)
            out.append(f"{pn if i == 0 else ''} & {name} & {cells} \\\\")
        out.append("\\midrule")
    out[-1] = "\\bottomrule"; out += ["\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out) + "\n"


def main() -> None:
    aggs = [("Per query", "per_query"), ("Equal family", "per_family_equal_weight")]
    caption = ("The (family, predicted best model) router: five-split mean utility. The key is the argmax of "
               "predicted utility from the frozen Qwen2.5-7B-Instruct layer-21 state, cross-fitted on training "
               "rows. Split-0 paired intervals are in the text. Source: exp44.")
    out_path = ROOT / "reports" / "tab_argmax.tex"
    out_path.write_text(table(ARG, "tab:argmax", caption, aggs))
    print("wrote", out_path)
    if OL_TABLES.parent.exists():
        OL_TABLES.mkdir(exist_ok=True)
        dest = OL_TABLES / "tab_argmax.tex"
        shutil.copy(out_path, dest)
        print("copied", dest)


if __name__ == "__main__":
    main()
