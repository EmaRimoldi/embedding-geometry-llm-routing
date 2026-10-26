"""exp37 summary across readers and both panels: best layer per reader, depth curves, transformations.
Reads results/difficulty_llm/exp37_layer_sweep/<model>/result.json (RouterBench) and the LLMRouterBench
counterpart; writes results/difficulty_llm/exp37_layer_sweep/SUMMARY.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT  # noqa: E402

ROOTS = {"RouterBench": ANALYSIS_ROOT / "results" / "difficulty_llm" / "exp37_layer_sweep", "LLMRouterBench": ANALYSIS_ROOT / "llmrouterbench" / "results" / "performance-cost" / "difficulty_llm" / "exp37_layer_sweep"}
PARAMS = {"qwen2.5-7b-instruct": "7.6B", "qwen2.5-coder-1.5b-instruct": "1.5B", "qwen3-embedding-0.6b": "0.6B", "gte-qwen2-7b": "7.6B", "e5-large-v2": "335M", "minilm-l12": "33M", "bge-m3": "568M", "qwen3-8b": "8.2B"}
KIND = {"qwen2.5-7b-instruct": "instruct decoder", "qwen2.5-coder-1.5b-instruct": "instruct decoder (code)", "qwen3-embedding-0.6b": "contrastive decoder", "gte-qwen2-7b": "contrastive decoder (the study's embedder)", "e5-large-v2": "contrastive encoder", "minilm-l12": "contrastive encoder", "bge-m3": "contrastive encoder", "qwen3-8b": "instruct decoder"}
ORDER = ["qwen2.5-7b-instruct", "qwen3-8b", "qwen2.5-coder-1.5b-instruct", "gte-qwen2-7b", "qwen3-embedding-0.6b", "bge-m3", "e5-large-v2", "minilm-l12"]


def main() -> None:
    R = {b: {m: json.load(open(root / m / "result.json")) for m in ORDER if (root / m / "result.json").exists()} for b, root in ROOTS.items()}
    md = ["# exp37 — which frozen reader, layer and pooling make per-query difficulty recoverable?", "",
          "Target: number of pool models solving the query; ridge on family one-hot + standardised representation; R²_beyond_family (mean over 5 splits; equal weight over families in brackets); CI = bootstrap on split 0. GTE reference is the study's embedding read at its own last layer through the normal pipeline.", ""]
    for b, res in R.items():
        if not res: continue
        anyres = next(iter(res.values())); g = anyres["gte_reference"]
        md += [f"## {b}", "", f"GTE reference: {g['r2_beyond_family']:+.3f} [{g['equal_weight']:+.3f}], CI [{g['ci_split0'][0]:+.3f}, {g['ci_split0'][1]:+.3f}]", "",
               "reader | kind | params | layers | best layer / pooling | R²_beyond_family [equal weight] | CI split 0 | Spearman | null (max) | vs GTE", "---|---|---|---|---|---|---|---|---|---"]
        for m, d in res.items():
            bl = d["best_layer"]; v = d["layers"][bl]; nl = d["model_meta"]["layers"] - 1
            md.append(f"{m} | {KIND.get(m, '')} | {PARAMS.get(m, '')} | {nl} | {bl} ({int(bl.split('_')[0][1:]) / nl:.2f} of depth) | {v['r2_beyond_family']:+.3f} [{v['equal_weight']:+.3f}] | [{v['ci_split0'][0]:+.3f}, {v['ci_split0'][1]:+.3f}] | {v['spearman']:.3f} | {d['null_best_layer']['max']:+.3f} | {v['r2_beyond_family'] - g['r2_beyond_family']:+.3f}")
        md += ["", "### Depth curves (best pooling of each reader; R²_beyond_family at 0, 1/4, 1/2, 3/4, full depth)", "", "reader | pooling | 0 | 1/4 | 1/2 | 3/4 | full | best", "---|---|---|---|---|---|---|---"]
        for m, d in res.items():
            nl = d["model_meta"]["layers"] - 1; pool = d["best_layer"].split("_")[1]; vals = [d["layers"][f"L{int(round(f * nl))}_{pool}"]["r2_beyond_family"] for f in (0, .25, .5, .75, 1)]
            md.append(f"{m} | {pool} | " + " | ".join(f"{x:+.3f}" for x in vals) + f" | {d['layers'][d['best_layer']]['r2_beyond_family']:+.3f}")
        md += ["", "### Poolings (best layer of each pooling)", "", "reader | " + " | ".join(["last", "mean", "last8", "cls"]), "---|---|---|---|---"]
        for m, d in res.items():
            cells = []
            for p in ("last", "mean", "last8", "cls"):
                ks = [k for k in d["layers"] if k.endswith("_" + p)]
                cells.append("—" if not ks else max(f"{d['layers'][k]['r2_beyond_family']:+.3f} ({k.split('_')[0]})" for k in ks) if False else (lambda kb: f"{d['layers'][kb]['r2_beyond_family']:+.3f} ({kb.split('_')[0]})")(max(ks, key=lambda k: d["layers"][k]["r2_beyond_family"])))
            md.append(f"{m} | " + " | ".join(cells))
        md += ["", "### Transformations of the best layer", "", "reader | " + " | ".join(next(iter(res.values()))["extras"].keys() if False else ["pca64", "pca256", "concat(1/2+3/4+full)", "MLP probe", "vocab projection (65d)", "prompt surprise (5d)", "vocab+surprise (70d)"]), "---|" + "---|" * 7]
        for m, d in res.items():
            e = d["extras"]; keys = ["pca64", "pca256"] + [k for k in e if k.startswith("concat")] + [k for k in e if k.startswith("mlp")] + ["logit_lens_last(65d)", "prompt_surprise_scalars(5d)", "logit_lens+surprise(70d)"]
            md.append(f"{m} | " + " | ".join(f"{e[k]['r2_beyond_family']:+.3f} [{e[k]['equal_weight']:+.3f}]" if k in e else "—" for k in keys))
        md.append("")
    out = ROOTS["RouterBench"] / "SUMMARY.md"; out.write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
