"""exp14 - does a better topic labelling make the (family, topic) cell lookup better on LLMRouterBench?

Label sets (read-only; nothing under llmrouterbench/ is modified):
  released   llmrouterbench/topics/<panel>.json      - the release's labels: MMLU-Pro 14 categories, ArenaHard 3,
                                                     every other family = its own name (25 cells)
  llm        llmrouterbench/topics_llm/<panel>.json  - Qwen2.5-32B-Instruct labels, 78 cells across all families
  audited    llm label where the independent full-text second pass (audits/2026-09-20_fulltext_second_pass/
             pass_B.jsonl) proposed the same label; otherwise the family name (the audit's own accepted set;
             built here on the fly, no file is written outside this folder)
  family     family only (fallback of every set)
Protocol: in-domain, the paper's five family-stratified 70/30 splits (seeds 0-4), lambda in {0, .5, 1, 2, 5, 10}
(the rescaled LRB grid, exp13), cell lookup = best mean training utility per cell with >= 50 training rows,
family fallback below. References: best single (train), kNN-100 on GTE (cosine), oracle. Reported query-weighted
and equal weight over families, per split min/max, per family at lambda 2 and 5, and the number of usable cells.
Writes results/exp14_lrb_topic_labels/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from rb.probes.sweep import split

EXP = "exp14_lrb_topic_labels"; PANEL = "llmrouterbench-performance-cost"; LAMS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0); MIN_CELL = 50; SEEDS = (0, 1, 2, 3, 4)
TOP = C.E4 / "llmrouterbench" / "topics" / f"{PANEL}.json"; TOPLLM = C.E4 / "llmrouterbench" / "topics_llm" / f"{PANEL}.json"
PASSB = C.E4 / "llmrouterbench" / "topics_llm" / "audits" / "2026-09-20_fulltext_second_pass" / "pass_B.jsonl"


def labels(D):
    p = D["panel"]; fam_name = np.array([str(s) for s in p.dataset_ids], dtype=object)
    rel = np.array(json.load(open(TOP))["topic"], dtype=object); llm = np.array(json.load(open(TOPLLM))["topic"], dtype=object)
    if p.keep is not None: rel, llm = rel[p.keep], llm[p.keep]
    passb = {}
    for line in open(PASSB):
        r = json.loads(line); pr = r.get("parsed"); passb[int(r["row"])] = (pr or {}).get("label")
    aud = np.array([llm[i] if (passb.get(i) is not None and str(llm[i]).endswith("/" + str(passb[i])) or passb.get(i) == llm[i]) else fam_name[i] for i in range(len(llm))], dtype=object)
    return {"family": fam_name, "released": rel, "llm": llm, "audited": aud}


def lookup(U, tr, te, fam, cell, F):
    fb = np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)]); ch = fb[fam[te]].copy(); used = 0
    for k in np.unique(cell[te]):
        m = cell[tr] == k
        if m.sum() >= MIN_CELL: ch[cell[te] == k] = U[tr][m].mean(0).argmax(); used += 1
    return ch, used


def main():
    D = C.load(PANEL); r, c, fam, fn = D["r"], D["c"], D["fam"], D["fnames"]; F, n = len(fn), len(r); xu = C.unit(D["x"])
    L = labels(D); cells = {k: np.unique(v, return_inverse=True)[1] for k, v in L.items()}
    ncells = {k: int(len(np.unique(v))) for k, v in L.items()}; print("cells:", ncells, flush=True)
    methods = ["best_single", "knn100"] + [f"lookup:{k}" for k in ("family", "released", "llm", "audited")] + ["oracle"]
    per = {m: {f"{l:g}": {"qw": [], "macro": [], "fam": []} for l in LAMS} for m in methods}; usable = {k: [] for k in cells}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); nb = tr[C.knn_mean(xu[te], xu[tr], 100)]; ii = np.arange(len(te))
        for lam in LAMS:
            U = r - lam * c; Ute = U[te]; ch = {"best_single": np.full(len(te), int(U[tr].mean(0).argmax())), "knn100": U[nb].mean(1).argmax(1), "oracle": Ute.argmax(1)}
            for k, cl in cells.items():
                ch[f"lookup:{k}"], used = lookup(U, tr, te, fam, cl, F)
                if lam == LAMS[0]: usable[k].append(used)
            for m in methods:
                u = Ute[ii, ch[m]]; perf = [float(u[fam[te] == g].mean()) for g in range(F)]; per[m][f"{lam:g}"]["qw"].append(float(u.mean())); per[m][f"{lam:g}"]["macro"].append(float(np.mean(perf))); per[m][f"{lam:g}"]["fam"].append(perf)
        print("seed", seed, "done", flush=True)
    def ms(v): return (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
    res = C.provenance(D); res.update({"experiment": EXP, "lambdas": list(LAMS), "n_cells": ncells, "usable_cells_per_split": {k: v for k, v in usable.items()},
                                       "results": {m: {L_: {"qw": ms(per[m][L_]["qw"]), "macro": ms(per[m][L_]["macro"]), "per_family_mean": dict(zip(fn, np.mean(per[m][L_]["fam"], 0).round(4).tolist()))} for L_ in per[m]} for m in methods},
                                       "paired_llm_minus_released_per_split": {L_: [a - b for a, b in zip(per["lookup:llm"][L_]["qw"], per["lookup:released"][L_]["qw"])] for L_ in per["lookup:llm"]}})
    md = [f"# {EXP} ({PANEL})", "", f"Cells per label set: {ncells}; cells with >= {MIN_CELL} training rows per split: " + ", ".join(f"{k} {int(np.mean(v))}" for k, v in usable.items()) + ". Utility mean over 5 splits [min, max]; qw = query-weighted, macro = equal weight over families. λ̃ = λ × 0.0613.", "",
          "method | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
    for m in methods: md.append(f"{m} | " + " | ".join(f"{res['results'][m][f'{l:g}']['qw'][0]:.3f} [{res['results'][m][f'{l:g}']['macro'][0]:.3f}]" for l in LAMS))
    md += ["", "## paired difference llm − released, query-weighted, per split", ""] + [f"- λ={L_}: " + ", ".join(f"{d:+.4f}" for d in v) for L_, v in res["paired_llm_minus_released_per_split"].items()]
    for lam in (2.0, 5.0):
        md += ["", f"## per family at λ = {lam:g} (mean over splits)", "", "family | " + " | ".join(methods), "---|" + "---|" * len(methods)]
        for g in fn: md.append(f"{g} | " + " | ".join(f"{res['results'][m][f'{lam:g}']['per_family_mean'][g]:.3f}" for m in methods))
    C.save(C.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
