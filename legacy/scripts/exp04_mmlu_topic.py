"""exp04: inside MMLU (one format), does K-means on the embedding cluster by subject (topic)?

GTE vs a lexical control (hashed bag-of-words) vs chance (permuted labels).
See experiments/exp04_mmlu_topic.yaml.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402
from rb.panelctx import PANEL, REP, RESULTS_ROOT, TOPIC_FAMILIES, topic_str  # noqa: E402

EXP = "exp04_mmlu_topic"
# MMLU's official grouping (hendrycks/test categories.py): subject -> super-category.
STEM = {"abstract-algebra", "anatomy", "astronomy", "college-biology", "college-chemistry", "college-computer-science",
        "college-mathematics", "college-physics", "computer-security", "conceptual-physics", "electrical-engineering",
        "elementary-mathematics", "high-school-biology", "high-school-chemistry", "high-school-computer-science",
        "high-school-mathematics", "high-school-physics", "high-school-statistics", "machine-learning"}
HUMANITIES = {"formal-logic", "high-school-european-history", "high-school-us-history", "high-school-world-history",
              "international-law", "jurisprudence", "logical-fallacies", "moral-disputes", "moral-scenarios", "philosophy",
              "prehistory", "professional-law", "world-religions"}
SOCIAL = {"econometrics", "high-school-geography", "high-school-government-and-politics", "high-school-macroeconomics",
          "high-school-microeconomics", "high-school-psychology", "human-sexuality", "professional-psychology",
          "public-relations", "security-studies", "sociology", "us-foreign-policy"}
# everything else -> "other" (business, health, misc)


def supercat(subject: str) -> str:
    return "stem" if subject in STEM else "humanities" if subject in HUMANITIES else "social" if subject in SOCIAL else "other"


def scores(labels: np.ndarray, truth: np.ndarray) -> dict:
    k = labels.max() + 1
    cont = np.zeros((k, truth.max() + 1), dtype=int)
    for c in range(k):
        cont[c] = np.bincount(truth[labels == c], minlength=truth.max() + 1)
    return {"purity": float(cont.max(1).sum() / cont.sum()),
            "nmi": float(normalized_mutual_info_score(truth, labels)),
            "ari": float(adjusted_rand_score(truth, labels)), "_cont": cont}


def chance(labels: np.ndarray, truth: np.ndarray, n: int = 5) -> dict:
    rng = np.random.default_rng(123)
    vals = [scores(labels, rng.permutation(truth)) for _ in range(n)]
    return {m: float(np.mean([v[m] for v in vals])) for m in ("purity", "nmi", "ari")}


def main() -> None:
    panel = load_panel(PANEL)
    tf = TOPIC_FAMILIES[0]; is_rb = PANEL.startswith("routerbench")
    mm = np.flatnonzero(panel.dataset_ids == tf)
    # Subject labels: the release's record index is "mmlu-<subject>.<split>.<n>" (read-only npz, same rows, keep mask applied).
    ts = topic_str(panel, "record_prefix")[mm]
    subjects = np.array([str(t).replace("mmlu-", "", 1).replace("mmlupro/", "", 1) for t in ts])
    assert all(str(t).startswith(("mmlu-", "mmlupro/")) for t in ts), "subject join failed"
    subj_names = sorted(set(subjects)); s_idx = {s: i for i, s in enumerate(subj_names)}
    subj = np.array([s_idx[s] for s in subjects])
    sc_names = ["stem", "humanities", "social", "other"]; sc = np.array([sc_names.index(supercat(s)) for s in subjects]) if is_rb else None   # MMLU-Pro has no super-categories

    reps = {"gte": find_rep(PANEL, REP),
            "hash-bow": find_rep(PANEL, "computed/hash-bow@prompt")}
    # the key below is historical ("gte"); record which representation it actually was, so the file is self-describing
    _rep_used = reps["gte"].name
    result = {**panel.provenance(), "experiment": EXP, "topic_family": tf, "n_mmlu": int(len(mm)), "n_subjects": len(subj_names),
              "subject_sizes": {s: int((subjects == s).sum()) for s in subj_names},
              "supercategory_sizes": {n: int((sc == i).sum()) for i, n in enumerate(sc_names)} if is_rb else None, "representation": _rep_used, "by_rep": {}}
    md = [f"# {EXP}", "", f"{tf} rows: {len(mm)}; subjects: {len(subj_names)} (sizes {min(result['subject_sizes'].values())}–{max(result['subject_sizes'].values())}); super-categories: {result['supercategory_sizes']}", ""]
    for name, spec in reps.items():
        x = load_rep(spec, panel.n, panel.keep)[mm]
        u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        out = {}
        # (a) K = 57, natural sizes
        nat = []
        for seed in (0, 1, 2):
            lab = KMeans(n_clusters=len(subj_names), n_init=3, random_state=seed).fit_predict(u)
            s = scores(lab, subj); s.pop("_cont"); s["chance"] = chance(lab, subj); nat.append(s)
        out["k57_natural"] = nat
        # (b) K = 57, balanced 80 per subject
        bal = []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            rows = np.concatenate([rng.choice(np.flatnonzero(subj == j), size=min(80, int((subj == j).sum())), replace=False) for j in range(len(subj_names))])
            lab = KMeans(n_clusters=len(subj_names), n_init=3, random_state=0).fit_predict(u[rows])
            s = scores(lab, subj[rows]); s.pop("_cont"); s["chance"] = chance(lab, subj[rows]); s["n_rows"] = int(len(rows)); bal.append(s)
        out["k57_balanced80"] = bal
        # (c) K = 4 vs super-categories, (d) K = 3 vs super-categories (RouterBench MMLU only)
        for k in ((4, 3) if is_rb else ()):
            runs = []
            for seed in (0, 1, 2):
                lab = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(u)
                s = scores(lab, sc); cont = s.pop("_cont"); s["chance"] = chance(lab, sc)
                s["contingency_clusters_x_supercat"] = cont.tolist(); runs.append(s)
            out[f"k{k}_vs_supercategory"] = runs
        result["by_rep"][name] = out
        md += [f"## {name}", "", "setting | purity | NMI | ARI | chance purity / NMI / ARI", "---|---|---|---|---"]
        for key, runs in out.items():
            for i, s in enumerate(runs):
                md.append(f"{key} seed{i} | {s['purity']:.3f} | {s['nmi']:.3f} | {s['ari']:.3f} | {s['chance']['purity']:.3f} / {s['chance']['nmi']:.3f} / {s['chance']['ari']:.3f}")
        if is_rb:
            c = np.asarray(out["k4_vs_supercategory"][0]["contingency_clusters_x_supercat"])
            md += ["", f"K = 4 contingency (seed 0), rows = clusters, cols = {sc_names}:", ""] + [f"cluster {i}: {row.tolist()}" for i, row in enumerate(c)] + [""]
            c = np.asarray(out["k3_vs_supercategory"][0]["contingency_clusters_x_supercat"])
            md += [f"K = 3 contingency (seed 0), cols = {sc_names}:", ""] + [f"cluster {i}: {row.tolist()}" for i, row in enumerate(c)] + [""]
        print("\n".join(md[-40:]), flush=True)
    out_dir = RESULTS_ROOT / "geometry_tests" / EXP; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    (out_dir / "result.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
