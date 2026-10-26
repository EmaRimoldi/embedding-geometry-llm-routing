"""exp26 addendum: which families get confused once the template is gone? Family classifier (exp08 recipe) on prompt vs core, split 0; confusion matrices;
and the difficulty read-out per family with ridge fit WITHIN each family (to see whether the GSM8K drop is a cross-family fitting effect or a loss of signal)."""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from rb.data import SUCCESS_THRESHOLD, load_panel
from rb.probes.constants import RIDGE_ALPHAS
from rb.probes.sweep import split
from rb.reps import find_rep, load_rep
from deformat import PANEL_DIR
from rb.panelctx import OUT_OVERRIDE, PANEL, REP, REP_NAME
p = load_panel("routerbench-0shot"); n = p.n; fam = p.dataset_index(); names = [str(s) for s in p.datasets]; F = len(names)
r = np.nan_to_num(p.rewards.astype(np.float64), nan=0.0); ns = (r >= SUCCESS_THRESHOLD).sum(1).astype(np.float64)
X = {"prompt": load_rep(find_rep(PANEL, REP), n, p.keep).astype(np.float32), "core": np.load(PANEL_DIR / f"{REP_NAME}.npy").astype(np.float32)}
tr, te = split(n, fam, 0); out = {}; lines = []
for k, x in X.items():
    sc = StandardScaler().fit(x[tr]); pred = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(x[tr]), fam[tr]).predict(sc.transform(x[te]))
    cm = np.zeros((F, F), int)
    for a, b in zip(fam[te], pred): cm[a, b] += 1
    out[k] = {"family_acc": float(np.mean(pred == fam[te])), "confusion_rows_true_cols_pred": cm.tolist()}
    lines += [f"## {k}: family accuracy {out[k]['family_acc']:.4f} (split 0)", "", "true \\ pred | " + " | ".join(names), "---|" + "---|" * F] + [f"{names[i]} | " + " | ".join(str(v) for v in cm[i]) for i in range(F)] + [""]
    # within-family ridge on n_solved (fit and test inside each family, split 0)
    wf = {}
    for g in range(F):
        trg, teg = tr[fam[tr] == g], te[fam[te] == g]
        if len(trg) < 100: continue
        s2 = StandardScaler().fit(x[trg]); pn = RidgeCV(alphas=RIDGE_ALPHAS).fit(s2.transform(x[trg]), ns[trg]).predict(s2.transform(x[teg]))
        wf[names[g]] = float(1 - ((ns[teg] - pn) ** 2).sum() / ((ns[teg] - ns[trg].mean()) ** 2).sum())
    out[k]["within_family_ridge_r2"] = wf; lines += ["within-family ridge R² (fit inside the family): " + ", ".join(f"{a} {b:+.3f}" for a, b in wf.items()), ""]
((OUT_OVERRIDE or HERE) / "family_confusion.md").write_text("# Family confusion and within-family difficulty, prompt vs core (split 0)\n\n" + "\n".join(lines)); json.dump(out, open((OUT_OVERRIDE or HERE) / "family_confusion.json", "w"), indent=1); print("\n".join(lines))
