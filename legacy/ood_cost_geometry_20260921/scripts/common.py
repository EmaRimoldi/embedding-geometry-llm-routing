"""Shared helpers for the ood_cost_geometry_20260921 study (held-out family; cost transfer).

Everything here is read-only on the rest of experiments4: frozen panels, canonical GTE
prompt embeddings, hashed bag-of-words, prompt token counts. Nothing outside this folder
is written. Utility is the paper convention reward - lambda * cost.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ISO = HERE.parent
E4 = ISO.parent
sys.path.insert(0, str(E4 / "src"))
from rb.data import load_panel  # noqa: E402
from rb.reps import find_rep, load_rep  # noqa: E402

RESULTS = ISO / "results"
LAMS = tuple(float(v) for v in os.environ["E4_OCG_LAMS"].replace(";", " ").replace(",", " ").split()) if os.environ.get("E4_OCG_LAMS") else (0.0, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0)   # separator: ; or space (Slurm --export splits on commas)
SEEDS = (42, 999, 2024, 2025, 3407)
EPS = 1e-7


def load(panel_name: str, with_bow: bool = False):
    p = load_panel(panel_name)
    x = load_rep(find_rep(panel_name, "computed/gte-qwen2-7b@prompt"), p.n, p.keep).astype(np.float32)
    r = np.nan_to_num(p.rewards.astype(np.float64), nan=0.0)
    c = np.nan_to_num(p.costs.astype(np.float64), nan=0.0)
    fam = p.dataset_index().astype(np.int64)
    fnames = [str(s) for s in p.datasets]
    names = [str(m).split("/")[-1] for m in p.models]
    z = np.load(E4 / "features" / panel_name / "prompt" / "lengths.npz")
    key = "qwen_tokens" if "qwen_tokens" in z.files else z.files[0]
    L = z[key].astype(np.float64)
    if L.shape[0] != p.n:
        L = L[p.keep]
    assert L.shape[0] == p.n
    out = {"panel": p, "x": x, "r": r, "c": c, "fam": fam, "fnames": fnames, "names": names, "n_in": L, "name": panel_name}
    if with_bow:
        b = np.load(E4 / "features" / panel_name / "prompt" / "hash-bow.npy", mmap_mode="r")
        b = np.asarray(b[p.keep] if b.shape[0] != p.n else b, dtype=np.float32)
        out["bow"] = b
    return out


def unit(a):
    return (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def fit_posterior(x, fam, sources, target_rows, balanced=True, seed=42):
    """Soft posterior over source families from GTE, as in the held-out study (raw GTE, lbfgs, C=1)."""
    rows = np.flatnonzero(np.isin(fam, sources))
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, class_weight="balanced" if balanced else None, random_state=seed).fit(x[rows], fam[rows])
    raw = clf.predict_proba(x[target_rows])
    order = {int(g): i for i, g in enumerate(clf.classes_)}
    return raw[:, [order[int(g)] for g in sources]]


def family_tables(r, c, fam, sources):
    rt = np.stack([r[fam == g].mean(0) for g in sources])
    ct = np.stack([c[fam == g].mean(0) for g in sources])
    return rt, ct


def knn_mean(xq, xs, k, chunk=2000):
    """Indices of the k nearest source rows (cosine on unit vectors) for each query row."""
    out = np.zeros((len(xq), k), dtype=np.int64)
    for s in range(0, len(xq), chunk):
        sim = xq[s:s + chunk] @ xs.T
        out[s:s + chunk] = np.argpartition(-sim, k, axis=1)[:, :k]
    return out


def per_family(u, fam_te, F):
    return {int(g): float(u[fam_te == g].mean()) for g in np.unique(fam_te)}


def aggregate(fold_values: dict, fold_n: dict):
    """fold_values: family -> utility. Returns query-weighted and macro means."""
    fams = sorted(fold_values)
    w = np.array([fold_n[g] for g in fams], float); v = np.array([fold_values[g] for g in fams], float)
    return float((w * v).sum() / w.sum()), float(v.mean())


def save(d: Path, result: dict, md_lines: list[str]):
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(result, indent=1, default=float))
    (d / "result.md").write_text("\n".join(md_lines) + "\n")
    print("wrote", d, flush=True)


def provenance(data):
    p = data["panel"]
    return {"panel": data["name"], "rows": int(p.n), "models": data["names"], "families": data["fnames"],
            "lambdas": list(LAMS), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
