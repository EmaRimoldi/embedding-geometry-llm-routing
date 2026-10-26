"""exp09 - what should an unseen query retrieve? Oracle-keyed lookups under strict family holdout.

For every held-out family g: lookups are built from SOURCE queries only, keyed by a property of each
query; the test query is routed by its TRUE key (diagnostic: the key is taken from the test query's own
outcomes/labels, so these are ceilings for "if we could estimate this key"). Keys:
  family        the source family (test rows use their own family -> the in-domain family ceiling)
  topic         broad topic label shared across families (unmatched -> global)
  regime        the query's verbosity coordinate v(x) = LS coordinate of its log-cost vector on the rank-1
                axis fitted on source families; binned into NB quantile bins of the source distribution
  difficulty    n_solved (number of panel models with reward >= .5), binned into NB quantile bins
  regime x difficulty
  regime x tier the tier of the cheapest solver (rank of the best-utility model at lambda among solvers)
  global        no key (family-balanced best single)
Each bin stores per-model mean reward and mean cost over the source queries in the bin, family-balanced
(each source family weighted equally inside the bin); bins with < MIN rows fall back to the parent
(regime alone, then global). Utility on the held-out family at the lambda grid, query-weighted [macro].
PROVES which query property is the right retrieval target OOD, and how far each can take a router.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common as C
from exp05_verbosity_axis_router import axes, coords
from rb.panelctx import topic_str

EXP = "exp09_oracle_keys"; NB = 4; MIN = 100


def broad_topic(t):
    t = str(t).lower()
    for k, ws in (("math", ("math", "algebra", "calculus", "arith", "geometry", "statistic", "probab", "aime", "gsm")), ("coding", ("code", "program", "python", "swe", "mbpp")),
                  ("science", ("physic", "chem", "bio", "science", "medic", "anatom", "astron", "gpqa", "engineer", "computer")), ("humanities", ("history", "philos", "law", "moral", "religion", "polit", "econom", "psych", "sociol", "human")),
                  ("language", ("language", "arts", "writing", "hellaswag", "winogrande", "commonsense")), ("dialogue", ("tau", "mtbench", "arena", "chat", "dialog"))):
        if any(w in t for w in ws): return k
    return "other"


def table(rows, key_rows, r, c, fam, keyvals):
    """per-key family-balanced mean reward/cost over source rows."""
    out = {}
    for k in keyvals:
        m = rows[key_rows == k]
        if len(m) < MIN: continue
        fams = np.unique(fam[m]); rt = np.mean([r[m][fam[m] == f].mean(0) for f in fams], 0); ct = np.mean([c[m][fam[m] == f].mean(0) for f in fams], 0)
        out[k] = (rt, ct)
    return out


def main():
    panel_name = sys.argv[1]; D = C.load(panel_name); r, c, fam, fn, names = D["r"], D["c"], D["fam"], D["fnames"], D["names"]; F, M, n = len(fn), len(names), len(r)
    Y = np.log(c + C.EPS); S_ = (r >= 0.5); nsolved = S_.sum(1)
    topics = np.array([broad_topic(t) for t in topic_str(D["panel"])], dtype=object)
    keys = ["global", "family", "topic", "regime", "difficulty", "regime_x_difficulty", "regime_x_tier", "soft_prior"]
    per = {k: {str(l): {} for l in C.LAMS} for k in keys + ["oracle"]}; n_by = {}
    for g in range(F):
        te = np.flatnonzero(fam == g); tr = np.flatnonzero(fam != g); S = [s for s in range(F) if s != g]; n_by[fn[g]] = len(te)
        rt_f, ct_f = C.family_tables(r, c, fam, S); logct = np.log(ct_f + C.EPS); alpha, B, G = axes(logct, 1)
        v = coords(Y, alpha, B)[:, 0]                                    # regime coordinate for every query (test rows: their own = oracle key)
        qb = np.quantile(v[tr], np.linspace(0, 1, NB + 1)[1:-1]); rbin = np.digitize(v, qb)
        db = np.quantile(nsolved[tr], np.linspace(0, 1, NB + 1)[1:-1]); dbin = np.digitize(nsolved, db)
        P = C.fit_posterior(D["x"], fam, S, te, True, 42); Pr, Pc = P @ rt_f, P @ ct_f
        glob = (np.mean([r[fam == s].mean(0) for s in S], 0), np.mean([c[fam == s].mean(0) for s in S], 0))
        for lam in C.LAMS:
            L = str(lam); U = r - lam * c; Ute = U[te]; ii = np.arange(len(te))
            # tier key: rank (0 = cheapest) of the utility-best model among the query's solvers, at this lambda; queries nobody solves -> tier -1
            cost_rank = np.argsort(np.argsort(glob[1]))
            best = U.argmax(1); tier = np.where(S_[np.arange(n), best], cost_rank[best], -1)
            def route(keyfun):
                ktr = keyfun(tr); kte = keyfun(te); tab = table(tr, ktr, r, c, fam, np.unique(ktr)); ch = np.zeros(len(te), int)
                for i, k in enumerate(kte):
                    rt_, ct_ = tab.get(k, glob); ch[i] = int((rt_ - lam * ct_).argmax())
                return ch
            chs = {"global": np.full(len(te), int((glob[0] - lam * glob[1]).argmax())),
                   "family": np.full(len(te), int((r[te].mean(0) - lam * c[te].mean(0)).argmax())),
                   "topic": route(lambda idx: topics[idx]),
                   "regime": route(lambda idx: rbin[idx]),
                   "difficulty": route(lambda idx: dbin[idx]),
                   "regime_x_difficulty": route(lambda idx: rbin[idx] * 10 + dbin[idx]),
                   "regime_x_tier": route(lambda idx: rbin[idx] * 100 + tier[idx] + 1),
                   "soft_prior": (Pr - lam * Pc).argmax(1)}
            for k, ch in chs.items(): per[k][L][fn[g]] = float(Ute[ii, ch].mean())
            per["oracle"][L][fn[g]] = float(Ute.max(1).mean())
        print(fn[g], "lam100:", {k: round(per[k]["100.0"][fn[g]], 3) for k in keys}, flush=True)
    agg = {k: {L: C.aggregate(per[k][L], n_by) for L in per[k]} for k in per}
    res = C.provenance(D); res.update({"experiment": EXP, "bins": NB, "min_rows": MIN, "per_family": per, "aggregate_qw_macro": agg})
    cols = keys + ["oracle"]
    md = [f"# {EXP} ({panel_name})", "", f"Held-out family; lookups from source queries keyed by the TRUE property of the test query ({NB} quantile bins, min {MIN} rows, family-balanced bin means, fallback global). Query-weighted [macro].", "",
          "lambda | " + " | ".join(cols), "---|" + "---|" * len(cols)]
    for lam in C.LAMS:
        L = str(lam); md.append(f"{lam:g} | " + " | ".join(f"{agg[k][L][0]:.3f} [{agg[k][L][1]:.3f}]" for k in cols))
    for lam in (10.0, 50.0, 100.0):
        L = str(lam); md += ["", f"## per family, lambda = {lam:g}", "", "family | " + " | ".join(cols), "---|" + "---|" * len(cols)]
        for g in range(F): md.append(f"{fn[g]} | " + " | ".join(f"{per[k][L][fn[g]]:.3f}" for k in cols))
    C.save(C.RESULTS / EXP / panel_name, res, md)


if __name__ == "__main__":
    main()
