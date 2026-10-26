"""External SWE-bench Verified human time labels joined to frozen routing prompts.

Source: princeton-nlp/SWE-bench_Verified at c104f840cc67f8b6eec6f759ebc8b2693d585d4a,
data/test-00000-of-00001.parquet, SHA256 a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd.
Uses exact release record index and verifies all 500 origin_query strings.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

SOURCE = ROOT / "results/difficulty_neighborhoods/source_labels/swe_verified_c104.parquet"
SHA = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
LABELS = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    t = pd.read_parquet(SOURCE, columns=["instance_id", "difficulty", "repo"])
    assert len(t) == len(set(t.instance_id)) == 500
    by_id = t.set_index("instance_id")
    p = load_panel("llmrouterbench-performance-cost")
    ix = np.flatnonzero(p.dataset_ids == "swe-bench")
    assert len(ix) == 500 and len(set(p.record_indices[ix])) == 500
    release_path = glob.glob(str(RUNTIME_ROOT / "artifacts/llmrouterbench/bench-release/swe-bench/verified/*/*.json"))[0]
    rec = {str(z["index"]): z for z in json.load(open(release_path))["records"]}
    assert all(p.origin_queries[i] == rec[str(p.record_indices[i])]["origin_query"] for i in ix)
    instances = [rec[str(p.record_indices[i])]["instance_id"] for i in ix]
    assert set(instances) == set(by_id.index)
    labs = np.asarray([LABELS.index(by_id.loc[z, "difficulty"]) for z in instances])
    repo = np.asarray([str(by_id.loc[z, "repo"]) for z in instances])
    out = ROOT / "llmrouterbench/results/performance-cost/difficulty_neighborhoods/exp53_swe_labels"
    out.mkdir(parents=True, exist_ok=True)
    result = {"source_url": "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
              "source_sha256": SHA, "join": "500/500 verified instance IDs and exact origin_query strings",
              "labels": {name: int((labs == k).sum()) for k, name in enumerate(LABELS)}, "representations": {}}
    issue_path = out.parent / "exp54_swe_issues/gte-qwen2-7b.npy"
    fields = ("prompt", "origin_query") + (("issue_statement",) if issue_path.exists() else ())
    for field in fields:
        if field == "issue_statement":
            x = np.load(issue_path).astype(np.float32)
        else:
            rep = f"computed/gte-qwen2-7b@{field}"
            x = load_rep(find_rep(p.name, rep), p.n, p.keep)[ix].copy()
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        sim = x @ x.T
        np.fill_diagonal(sim, -np.inf)
        ret = {}
        rng = np.random.default_rng(531)
        peers = [np.flatnonzero((repo == repo[i]) & (np.arange(500) != i)) for i in range(500)]
        valid = np.array([len(z) > 0 for z in peers])
        for k in (5, 10, 40):
            nb = [z[np.argsort(-sim[i, z])[:k]] for i, z in enumerate(peers)]
            dist = np.asarray([np.abs(labs[i] - labs[z]).mean() if len(z) else np.nan for i, z in enumerate(nb)])
            ref = np.asarray([np.abs(labs[i] - labs[np.arange(500) != i]).mean() for i in range(500)])
            ref_repo = np.asarray([np.abs(labs[i] - labs[z]).mean() if len(z) else np.nan for i, z in enumerate(peers)])
            boot = np.array([1 - dist[g].mean() / ref_repo[g].mean() for g in
                             (rng.choice(np.flatnonzero(valid), valid.sum(), replace=True) for _ in range(2000))])
            ret[str(k)] = {"neighbor_ordinal_gap": float(np.nanmean(dist)), "random_all_gap": float(ref.mean()),
                           "random_same_repo_gap": float(np.nanmean(ref_repo)), "same_repo_valid": int(valid.sum()),
                           "queries_with_fewer_than_k_peers": int(sum(0 < len(z) < k for z in peers)),
                           "relative_reduction_vs_same_repo": float(1 - dist[valid].mean() / ref_repo[valid].mean()),
                           "ci95": np.quantile(boot, [.025, .975]).tolist(),
                           "same_label_share": float(np.mean([np.mean(labs[i] == labs[z]) for i, z in enumerate(nb) if len(z)]))}
        result["representations"][field] = ret
        if field == "prompt":
            xy = PCA(n_components=2, random_state=0).fit_transform(x)
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.1, 3.4),
                                          gridspec_kw={"width_ratios": [1.8, 1]}, constrained_layout=True)
            colors = ["#228b8d", "#ed9c28", "#af508c", "#5a389b"]
            for j in range(4):
                q = labs == j
                ax.scatter(xy[q, 0], xy[q, 1], s=13, alpha=.6, linewidths=0,
                           color=colors[j], label=f"{LABELS[j]} ({q.sum()})", rasterized=True)
            ax.set(xlabel="GTE principal component 1", ylabel="Principal component 2")
            ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.22),
                      ncol=2, columnspacing=.8)
            ax.spines[["top", "right"]].set_visible(False)
            for y, kk in enumerate((5, 10, 40)):
                d = ret[str(kk)]
                v = 100 * d["relative_reduction_vs_same_repo"]
                lo, hi = 100 * np.array(d["ci95"])
                ax2.errorbar(v, y, xerr=np.array([[v-lo], [hi-v]]), fmt="o", color="#1b6377", capsize=4)
            ax2.axvline(0, color="0.35", lw=1, ls="--")
            ax2.set(yticks=[0, 1, 2], yticklabels=["5 neighbors", "10 neighbors", "40 neighbors"],
                    xlabel="Difficulty-gap reduction vs.\nsame-repository random (%)",
                    xlim=(-5, 9), ylim=(2.5, -.5))
            ax2.spines[["top", "right"]].set_visible(False)
            fig.savefig(out / "swe_difficulty_pca.pdf", dpi=220)
            fig.savefig(out / "swe_difficulty_pca.png", dpi=220)
            plt.close(fig)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
