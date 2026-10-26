"""exp20 - SPROUT sensitivities for the routing comparison (best single, family lookup, cell lookup, kNN-100, oracle;
5 splits; lambda grid 0, 0.5, 5, 25, 50, 250; per query [equal weight]):
  baseline            : the appendix setting (13 models, duplicates removed, 74 long rows excluded)
  no_titan            : Titan Text Premier removed from the pool (430 zero-output rows, a quarter of its rows <= 3 tokens)
  titan_price_x0.5 / x2: Titan's cost column scaled
  duplicates_kept     : the 280 duplicate rows kept (a prompt can sit on both sides of a split)
  long_rows_kept      : the 74 rows above the GTE context added back for the lookup methods (no embedding: kNN n/a)
Writes results/exp20_sprout_sensitivity/."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import train_test_split
import sprout_common as S
from exp19_sprout_routing import lookup

EXP = "exp20_sprout_sensitivity"; LAMS = (0.0, 0.5, 5.0, 25.0, 50.0, 250.0, 500.0, 1000.0); SEEDS = (0, 1, 2, 3, 4)


def run(D, knn=True, titan_factor=1.0):
    r, c, fam, fn = D["r"], D["c"].copy(), D["fam"], D["fnames"]; F, n = len(fn), D["n"]
    if titan_factor != 1.0 and "aws-titan-text-premier-v1" in D["names"]: c[:, D["names"].index("aws-titan-text-premier-v1")] *= titan_factor
    _, cell = np.unique(D["cell_str"], return_inverse=True); xu = S.unit(D["x"]) if (knn and D["x"] is not None) else None
    out = {m: {f"{l:g}": {"qw": [], "eq": []} for l in LAMS} for m in ("best_single", "family_lookup", "cell_lookup", "knn100", "oracle")}
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=fam); ii = np.arange(len(te))
        nb = tr[S.knn_idx(xu[te], xu[tr], 100)] if xu is not None else None
        for lam in LAMS:
            U = r - lam * c; Ute = U[te]; L = f"{lam:g}"
            ch = {"best_single": np.full(len(te), int(U[tr].mean(0).argmax())), "oracle": Ute.argmax(1), "family_lookup": np.array([U[tr][fam[tr] == g].mean(0).argmax() for g in range(F)])[fam[te]], "cell_lookup": lookup(U, tr, te, fam, cell, F)}
            if nb is not None: ch["knn100"] = U[nb].mean(1).argmax(1)
            for m, sel in ch.items():
                u = Ute[ii, sel]; out[m][L]["qw"].append(float(u.mean())); out[m][L]["eq"].append(float(np.mean([u[fam[te] == g].mean() for g in range(F)])))
    return {m: {L: [float(np.mean(v["qw"])), float(np.mean(v["eq"]))] if v["qw"] else None for L, v in d.items()} for m, d in out.items()}


def load_with_long():
    """all rows inside the GTE context plus the 74 excluded ones (no embedding), duplicates removed."""
    D = S.load(dedup=True, with_x=False); rows = [json.loads(l) for l in open(S.ROOT / "panel" / "rows.jsonl")]; z = np.load(S.ROOT / "panel" / "sprout.npz")
    km = np.load(S.ROOT / "features" / "gte-qwen2-7b_rows.npz"); keep = km["keep"].astype(bool); extra = np.where(~keep)[0]
    fmap = {f: g for g, f in enumerate(D["fnames"])}; r = np.vstack([D["r"], z["rewards"][extra]]); c = np.vstack([D["c"], z["costs"][extra]])
    fam = np.concatenate([D["fam"], [fmap[rows[i]["family"]] for i in extra]]); cell_str = np.concatenate([D["cell_str"], [rows[i]["family"] + "/" + rows[i]["family"] for i in extra]])
    return {"r": r, "c": c, "fam": fam, "fnames": D["fnames"], "n": len(r), "cell_str": np.asarray(cell_str, dtype=object), "x": None, "names": D["names"]}


def main():
    res = {"experiment": EXP, "lambdas": list(LAMS), "variants": {}}
    base = S.load(); res["variants"]["baseline"] = run(base); print("baseline done", flush=True)
    res["variants"]["no_titan"] = run(S.load(drop_models=("aws-titan-text-premier-v1",))); print("no_titan done", flush=True)
    res["variants"]["titan_price_x0.5"] = run(base, titan_factor=0.5); res["variants"]["titan_price_x2"] = run(base, titan_factor=2.0); print("titan price done", flush=True)
    res["variants"]["duplicates_kept"] = run(S.load(dedup=False)); print("dups done", flush=True)
    res["variants"]["long_rows_kept"] = run(load_with_long(), knn=False); print("long done", flush=True)
    md = [f"# {EXP} (SPROUT)", "", "Per query [equal weight], mean over 5 splits; lambda grid 0 / 0.5 / 5 / 25 / 50 / 250.", ""]
    for m in ("best_single", "family_lookup", "cell_lookup", "knn100", "oracle"):
        md += [f"## {m}", "", "variant | " + " | ".join(f"λ={l:g}" for l in LAMS), "---|" + "---|" * len(LAMS)]
        for v, d in res["variants"].items(): md.append(f"{v} | " + " | ".join((f"{d[m][f'{l:g}'][0]:.3f} [{d[m][f'{l:g}'][1]:.3f}]" if d[m][f"{l:g}"] else "n/a") for l in LAMS))
        md.append("")
    S.save(S.RESULTS / EXP, res, md)


if __name__ == "__main__":
    main()
