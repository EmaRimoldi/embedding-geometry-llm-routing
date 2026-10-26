"""exp24 addendum: for every version of every question, how many panel rows (and rows of its own family) are closer than its twin in another format?"""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import load_panel
from rb.reps import find_rep, load_rep
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME
HERE = Path(__file__).resolve().parent
p = load_panel("routerbench-0shot")
x = load_rep(find_rep(PANEL, REP), p.n, p.keep).astype(np.float32); u = x / np.linalg.norm(x, axis=1, keepdims=True)
ds = p.dataset_ids
E = {f: np.load(HERE / "embeddings" / f"{REP_NAME}__{f}.npy").astype(np.float32) for f in ("raw", "mmlu", "gsm")}
E = {f: e / np.linalg.norm(e, axis=1, keepdims=True) for f, e in E.items()}
famof = {"mmlu": "mmlu", "gsm": "grade-school-math"}
N = 24; out = {}; lines = []
lines.append("anchor -> twin | mean cos(twin) | mean cos to nearest panel row | panel rows closer than twin: median [min, max] | rows of the anchor family closer than twin: median [min, max] | questions with >= 100 panel rows closer | among the other 23 same-format texts: closer than twin (mean)")
lines.append("---|---|---|---|---|---|---")
for a in ("raw", "mmlu", "gsm"):
    S = E[a] @ u.T; Saa = E[a] @ E[a].T; np.fill_diagonal(Saa, -2)
    for b in ("raw", "mmlu", "gsm"):
        if a == b: continue
        tw = np.sum(E[a] * E[b], axis=1); closer = (S > tw[:, None]).sum(1)
        fam_closer = ((S > tw[:, None]) & (ds[None, :] == famof[a])).sum(1) if a in famof else None
        same24 = (Saa > tw[:, None]).sum(1)
        out[f"{a}->{b}"] = {"cos_twin": tw.tolist(), "panel_closer": closer.tolist(), "family_closer": None if fam_closer is None else fam_closer.tolist(), "same24_closer": same24.tolist(), "nearest_panel_cos": S.max(1).tolist()}
        fam_str = "n/a (raw has no family)" if fam_closer is None else f"{int(np.median(fam_closer))} [{fam_closer.min()}, {fam_closer.max()}]"
        lines.append(f"{a} -> {b} | {tw.mean():.3f} | {S.max(1).mean():.3f} | {int(np.median(closer))} [{closer.min()}, {closer.max()}] | {fam_str} | {int((closer >= 100).sum())}/24 | {same24.mean():.1f}")
lines.append("")
for a in ("mmlu", "gsm"):
    S = E[a] @ u.T; b = "gsm" if a == "mmlu" else "mmlu"; tw = np.sum(E[a] * E[b], axis=1); fams = {}
    for i in range(N):
        sel = ds[S[i] > tw[i]]
        for f_ in np.unique(sel): fams[f_] = fams.get(f_, 0) + int((sel == f_).sum())
    lines.append(f"{a} version: panel rows closer than the {b} twin, by family (summed over the 24 questions): " + ", ".join(f"{k} {v}" for k, v in sorted(fams.items(), key=lambda kv: -kv[1])))
json.dump(out, open(HERE / "results" / "twin_vs_panel.json", "w"), indent=1)
((OUT_OVERRIDE or HERE / "results") / "twin_vs_panel.md").write_text("# Twin vs panel: how many panel rows are closer than the same question in another format?\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
