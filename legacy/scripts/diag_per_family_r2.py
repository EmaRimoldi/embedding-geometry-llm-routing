"""Diagnostic: what makes per-family R2 beyond the family mean go large-negative?

Hypothesis under test: it is NOT small family size but small within-family
variance of the target. R2 beyond the family mean divides by SSE(family mean),
which is tiny for a family that is uniformly easy or uniformly hard, so a
homogeneous family produces a large negative R2 at any size.

Reads only results that already exist. No refitting.
"""
import json, glob, os

SIZES = {
    "routerbench-0shot": {"mmlu": 13925, "hellaswag": 10042, "grade-school-math": 7450,
                          "arc-challenge": 1470, "winogrande": 1267, "mbpp": 427, "mtbench": 80},
    "llmrouterbench-performance-cost": {"aime": 60, "arenahard": 750, "gpqa": 198, "hle": 2158,
                                        "livecodebench": 1055, "livemathbench": 121, "mmlupro": 3000,
                                        "simpleqa": 4326, "swe-bench": 500, "tau2": 278},
}
SHOW = ("ridge:gte", "ridge:family+gte", "ridge:bow", "ridge:surface", "floor:cell_mean")

def hunt(o, p="", acc=None):
    if acc is None:
        acc = []
    if isinstance(o, dict):
        if "per_family" in o and isinstance(o["per_family"], dict):
            acc.append((p, o["per_family"]))
        for k, v in o.items():
            hunt(v, p + "/" + str(k), acc)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            hunt(v, p + "[%d]" % i, acc)
    return acc

paths = []
for r in ("results", "llmrouterbench/results"):
    paths += glob.glob(os.path.join(r, "**", "*exp18*", "result.json"), recursive=True)
paths = sorted(set(p for p in paths if ".bak" not in p))

for path in paths:
    d = json.load(open(path))
    panel = d.get("panel", "?")
    sizes = SIZES.get(panel, {})
    print("#" * 78)
    print(panel, " <-", path)

    # 1. the target's own spread per family (n / mean / sd), saved by exp18
    spread = {}
    for lbl, pf in hunt(d):
        got = {f: rec for f, rec in pf.items()
               if isinstance(rec, dict) and "sd" in rec and "mean" in rec}
        if got:
            spread = got
            break
    if spread:
        print("\n  target (solve count) spread per family, sorted by sd:")
        print("       %-20s %7s %8s %8s %12s" % ("family", "n", "mean", "sd", "var=sd^2"))
        for f, rec in sorted(spread.items(), key=lambda kv: kv[1]["sd"]):
            print("       %-20s %7d %8.3f %8.3f %12.4f"
                  % (f, rec.get("n", sizes.get(f, -1)), rec["mean"], rec["sd"], rec["sd"] ** 2))

    # 2. per-family R2 for the methods we care about
    r2 = {}
    for lbl, pf in hunt(d):
        vals = {f: rec["r2_beyond_family"] for f, rec in pf.items()
                if isinstance(rec, dict) and "r2_beyond_family" in rec
                and rec["r2_beyond_family"] == rec["r2_beyond_family"]}
        if vals:
            name = lbl.strip("/").split("/")[-1]
            r2[name] = vals

    keep = [m for m in SHOW if m in r2] or list(r2)[:4]
    if keep:
        fams = sorted(spread or r2[keep[0]], key=lambda f: (spread[f]["sd"] if spread else sizes.get(f, 0)))
        print("\n  per-family R2 beyond the family mean (families ordered by target sd, lowest first):")
        print("       %-20s %7s %6s | %s" % ("family", "n", "sd", " | ".join("%-16s" % m for m in keep)))
        for f in fams:
            sd = spread[f]["sd"] if spread else float("nan")
            cells = " | ".join("%+16.3f" % r2[m][f] if f in r2[m] else "%16s" % "-" for m in keep)
            print("       %-20s %7d %6.3f | %s" % (f, sizes.get(f, -1), sd, cells))
        print()
        for m in keep:
            v = r2[m]
            allm = sum(v.values()) / len(v)
            big = [x for f, x in v.items() if sizes.get(f, 0) >= 400]
            lowvar = [x for f, x in v.items() if spread and spread[f]["sd"] >= 1.0]
            line = "       %-18s unweighted mean all = %+.4f" % (m, allm)
            if big and len(big) < len(v):
                line += " | n>=400 only = %+.4f" % (sum(big) / len(big))
            if lowvar and len(lowvar) < len(v):
                line += " | sd>=1.0 only = %+.4f" % (sum(lowvar) / len(lowvar))
            print(line)
    print()
