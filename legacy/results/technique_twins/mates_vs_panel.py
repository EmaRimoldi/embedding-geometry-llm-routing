"""exp25 addendum: with ALL 34,661 panel rows as competing candidates, do a probe's technique-mates (3) and story-mates (7)
still appear among its k nearest neighbours?  Also prints the top-10 panel neighbour texts of selected probes for reading."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import load_panel
from rb.reps import find_rep, load_rep
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME
HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "collection.json").read_text())["items"]; N = len(items)
tech = np.array([i["technique"] for i in items]); story = np.array([i["story"] for i in items])
p = load_panel("routerbench-0shot")
x = load_rep(find_rep(PANEL, REP), p.n, p.keep).astype(np.float32); u = x / np.linalg.norm(x, axis=1, keepdims=True)
E = np.load(HERE / "embeddings" / f"{REP_NAME}__raw.npy").astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
Sp = E @ u.T; Se = E @ E.T; np.fill_diagonal(Se, -2)
rows = []; out = {}
for i in range(N):
    def rank(j): return 1 + int((Sp[i] > Se[i, j]).sum()) + int((Se[i] > Se[i, j]).sum())   # among panel + other probes
    tm = np.flatnonzero((tech == tech[i]) & (np.arange(N) != i)); sm = np.flatnonzero((story == story[i]) & (np.arange(N) != i))
    rt = sorted(rank(j) for j in tm); rs = sorted(rank(j) for j in sm)
    out[i] = {"technique_mate_ranks": rt, "story_mate_ranks": rs}
    rows.append(f"{tech[i]} | {story[i]} | {rt} | {sum(r <= 10 for r in rt)} / {sum(r <= 100 for r in rt)} | {rs[:3]}… | {sum(r <= 10 for r in rs)} / {sum(r <= 100 for r in rs)} | {Sp[i].max():.2f}")
allt = [r for v in out.values() for r in v["technique_mate_ranks"]]; alls = [r for v in out.values() for r in v["story_mate_ranks"]]
summary = [f"Technique-mates (96 pairs): rank 1 = {np.mean(np.array(allt) == 1):.2f}, in top 10 = {np.mean(np.array(allt) <= 10):.2f}, in top 100 = {np.mean(np.array(allt) <= 100):.2f}, median rank {int(np.median(allt))}",
           f"Story-mates (224 pairs):   rank 1 = {np.mean(np.array(alls) == 1):.2f}, in top 10 = {np.mean(np.array(alls) <= 10):.2f}, in top 100 = {np.mean(np.array(alls) <= 100):.2f}, median rank {int(np.median(alls))}",
           f"Probes whose nearest text overall (panel + probes) is a technique-mate: {np.mean([out[i]['technique_mate_ranks'][0] == 1 for i in range(N)]):.2f}; a story-mate: {np.mean([out[i]['story_mate_ranks'][0] == 1 for i in range(N)]):.2f}"]
md = ["# Technique-mates vs the whole panel", "", "For each probe, every one of the 34,661 panel rows and the 31 other probes compete as candidates. Rank of each technique-mate / story-mate among them (1 = nearest of all).", ""] + summary + ["",
      "technique | story | ranks of the 3 technique-mates | in top 10 / top 100 | ranks of nearest 3 story-mates | story-mates in top 10 / top 100 | cos to nearest panel row", "---|---|---|---|---|---|---"] + rows
md += ["", "## Top-10 panel neighbours of selected probes (read them: are they the same technique?)", ""]
for key in [("work_rate", "school"), ("work_rate", "farm"), ("parity_invariant", "school"), ("parity_invariant", "farm"), ("pigeonhole", "shop"), ("missing_average", "travel"), ("linear_system", "farm")]:
    i = int(np.flatnonzero((tech == key[0]) & (story == key[1]))[0]); top = np.argsort(-Sp[i])[:10]
    md += [f"### {key[0]} / {key[1]}: {items[i]['text']}", ""]
    for j in top:
        t = str(p.prompts[j]); t = t[t.rfind("Answer: 9") + len("Answer: 9"):].strip() if p.dataset_ids[j] == "grade-school-math" else t
        md.append(f"* {Sp[i, j]:.2f} [{p.dataset_ids[j]}] {t.replace(chr(10), ' ')[:220]}")
    md.append("")
((OUT_OVERRIDE or HERE / "results") / "mates_vs_panel.md").write_text("\n".join(md) + "\n"); json.dump({str(k): v for k, v in out.items()}, open((OUT_OVERRIDE or HERE / "results") / "mates_vs_panel.json", "w"), indent=1)
print("\n".join(md))
