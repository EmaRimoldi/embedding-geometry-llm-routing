"""minilm/ study: headline comparison MiniLM vs GTE, same rows, for the experiments of Sections 1-2.
Configurations: RouterBench (GTE = the study's results/; MiniLM = minilm/results/routerbench-0shot__minilm-l12)
and LLMRouterBench <= 512 tokens (GTE and MiniLM both under minilm/results/performance-cost-le512__*).
Writes minilm/results/COMPARISON.md.  Missing files are reported, not fatal."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

E4 = Path(__file__).resolve().parents[1]; R = E4 / "minilm" / "results"
CFG = {"RouterBench": {"GTE": E4 / "results", "MiniLM": R / "routerbench-0shot__minilm-l12"},
       "LLMRouterBench ≤512": {"GTE": R / "performance-cost-le512__gte-qwen2-7b", "MiniLM": R / "performance-cost-le512__minilm-l12"}}
LAMS = ["0.0", "1.0", "10.0", "100.0", "1000.0"]; AGG = {"per query": "per_query", "equal weight": "per_family_equal_weight"}


def J(p):
    try: return json.load(open(p))
    except Exception: return None


def main() -> None:
    md = ["# MiniLM vs GTE on the same rows — headline numbers of Sections 1–2", "", "Every cell: MiniLM value / GTE value. Full tables in each configuration's chapter folders (`result.md`, `SUMMARY.md`).", ""]
    for bench, roots in CFG.items():
        md += [f"# {bench}", ""]
        # --- geometry: exp01, exp05, exp04
        e1 = {k: J(v / "geometry_tests/exp01_kmeans7_vs_families/result.json") for k, v in roots.items()}
        e5 = {k: J(v / "geometry_tests/exp05_knn_family_neighbours/result.json") for k, v in roots.items()}
        e4 = {k: J(v / "geometry_tests/exp04_mmlu_topic/result.json") for k, v in roots.items()}
        md += ["## Claim 1 — the space is organised by family (exp01 K-means purity / NMI, exp05 share of same-family neighbours)", "", "measure | MiniLM | GTE", "---|---|---"]
        for key in ("purity", "nmi"):
            md.append(f"K-means (K = #families) {key}, mean over seeds | " + " | ".join(f"{np.mean([s[key] for s in e1[k]['per_seed'].values()]):.3f}" if e1[k] else "—" for k in ("MiniLM", "GTE")))
        for kk in ("10", "100"):
            md.append(f"same-family share among {kk} nearest neighbours | " + " | ".join(f"{e5[k]['by_k'][kk]['mean_same_family']:.3f} (chance {e5[k]['by_k'][kk]['chance_query_weighted']:.2f})" if e5[k] and kk in e5[k]["by_k"] else "—" for k in ("MiniLM", "GTE")))
        if all(e4.values()):
            md += ["", "Claim 2 — subjects within a format (exp04, MMLU / MMLU-Pro K-means against the subject label; [purity, NMI, ARI] as stored):", ""]
            for k in ("MiniLM", "GTE"):
                rep = [r for r in e4[k]["by_rep"] if r != "hash-bow"][0]; fmt = lambda v: json.dumps(v)[:160] if isinstance(v, dict) else str([round(x, 3) if isinstance(x, float) else json.dumps(x)[:80] for x in v])  # noqa: E731
                md.append(f"- {k}: k57_natural {fmt(e4[k]['by_rep'][rep]['k57_natural'])}; k57_balanced80 {fmt(e4[k]['by_rep'][rep]['k57_balanced80'])}")
        md.append("")
        # --- exp08 routers
        md += ["## Claim 6 / 13 — family lookup vs kNN vs MLP (exp08/exp09, utility, mean over 5 splits)", ""]
        meth = [("family lookup", "best_model_per_family/best_model_per_family.json"), ("classify→family lookup", "classify_per_family_best_model/classify_per_family_best_model.json"), ("kNN-10", "knn/knn10.json"), ("kNN-40", "knn/knn40.json"), ("kNN-100", "knn/knn100.json"), ("MLP (100)", "mlp/mlp100.json"), ("MLP (100,100)", "mlp/mlp100x100.json")]
        for aggname, agg in AGG.items():
            md += [f"### {aggname}", "", "method | " + " | ".join(f"λ={l}" for l in LAMS), "---|" + "---|" * len(LAMS)]
            for name, rel in meth:
                d = {k: J(v / "basic_model_tests" / rel) for k, v in roots.items()}
                cells = []
                for l in LAMS:
                    v = [f"{d[k]['by_lambda'][l]['all'][agg]['utility'][0]:.4f}" if d[k] and l in d[k]["by_lambda"] else "—" for k in ("MiniLM", "GTE")]; cells.append(" / ".join(v))
                md.append(f"{name} | " + " | ".join(cells))
            ref = {k: J(v / "basic_model_tests/references.json") for k, v in roots.items()}
            for name in ("best_single", "oracle"):
                cells = []
                for l in LAMS:
                    v = [f"{ref[k]['references'][name][l]['all'][agg]['utility'][0]:.4f}" if ref[k] and l in ref[k]["references"][name] else "—" for k in ("MiniLM", "GTE")]; cells.append(" / ".join(v))
                md.append(f"{name} | " + " | ".join(cells))
            md.append("")
        # --- exp11 outcome locality
        e11 = {k: J(v / "outcome_locality/exp11_outcome_locality_split/result.json") for k, v in roots.items()}
        md += ["## Claim 10 — outcome agreement of the nearest neighbours minus random same-family rows (exp11, k = 10, all rows; mean [min, max] over splits)", ""]
        md += ["| MiniLM | GTE", "---|---|---"] and ["config | knn − random", "---|---"]
        for k in ("MiniLM", "GTE"):
            v = e11[k]["by_k"]["10"]["all"]["knn_minus_random"] if e11[k] else None; md.append(f"{k} | " + (f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" if v else "—"))
        md.append("")
        # --- exp17 topic significance
        e17 = {k: J(v / "outcome_locality/exp17_topic_significance/result.json") for k, v in roots.items()}
        md += ["## Check 1 — (family, topic) cell lookup gain over the family lookup and its permutation p (exp17)", "", "cell | λ | MiniLM observed (p) | GTE observed (p)", "---|---|---|---"]
        if any(e17.values()):
            keys = sorted({kk for k in e17 if e17[k] for kk in e17[k]["table"] if kk.startswith("lookup|")}, key=lambda s: (s.split("|")[1], float(s.split("|")[2])))
            for kk in keys:
                _, fam, lam = kk.split("|"); cells = [f"{e17[k]['table'][kk]['observed']:+.4f} ({e17[k]['table'][kk]['p_value']:.3f})" if e17[k] and kk in e17[k]["table"] else "—" for k in ("MiniLM", "GTE")]
                md.append(f"{fam} | {lam} | " + " | ".join(cells))
        else:
            md.append("(exp17 not finished yet)")
        md.append("")
        # --- exp20 cell recoverability
        e20 = {k: J(v / "outcome_locality/exp20_topic_recovery/result.json") for k, v in roots.items()}
        md += ["## Check 2 — cell label recoverable from the embedding (exp20, split 0, stored `ci95_split0`)", ""]
        for k in ("MiniLM", "GTE"):
            md.append(f"- {k}: " + (json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in e20[k]['ci95_split0'].items()})[:600] if e20[k] and isinstance(e20[k].get("ci95_split0"), dict) else "(exp20 aggregate not finished yet)"))
        md.append("")
        # --- available chapters
        md += ["## Result folders", ""]
        for k, v in roots.items():
            found = sorted(str(p.relative_to(E4)) for p in v.rglob("result.md") if "difficulty_llm" not in str(p) and "exp3" not in p.parent.name)
            md.append(f"- {k}: {len(found)} result files under `{v.relative_to(E4)}`")
        md.append("")
    (R / "COMPARISON.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
