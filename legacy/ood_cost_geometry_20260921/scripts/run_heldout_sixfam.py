"""Run one of the held-out study's method scripts (heldout_generalization_20260920/scripts/<name>.py, unchanged, read-only)
under the SIX-FAMILY SPROUT holdout instead of its raw20 source holdout. The script's panel loader is replaced by a
six-family panel built from this folder's sprout_common (duplicates removed, 43,887 rows) and its output root is
redirected to results/heldout_sixfam/ in this folder. Usage: run_heldout_sixfam.py <script_stem> [script args...]
e.g. run_heldout_sixfam.py heldout_shrink --panel sprout-sixfam --family-index 0 --seed-index 0"""
from __future__ import annotations
import importlib, os, sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import sprout_common as S
E4 = S.E4; ISO = E4 / "heldout_generalization_20260920"; sys.path[:0] = [str(E4 / "src"), str(ISO / "scripts")]
PANEL = "sprout-sixfam"; os.environ["E4_PANEL"] = PANEL


class Panel(SimpleNamespace):
    def dataset_index(self): return self._fam
    def provenance(self): return dict(self._provenance)


def load_sixfam(name=None):
    D = S.load(); fn = D["fnames"]; ids = np.asarray([fn[g] for g in D["fam"]], dtype=object)
    prov = {"panel": PANEL, "source_revision": "d54e3cda2189ca1ed13a877a44ed34736c4b6920", "rows_total": D["n_total"], "rows_kept": D["n"], "rows_excluded_context": D["n_total"] - D["n_kept"], "duplicates_removed": D["n_kept"] - D["n"], "models": D["names"], "families": fn, "family_definition": "six conceptual SPROUT families (gpqa, math, mmlupro, musr, openhermes, ragbench)", "reward_sha256": D["reward_sha256"], "cost_sha256": D["cost_sha256"], "runner": "ood_cost_geometry_20260921/scripts/run_heldout_sixfam.py"}
    p = Panel(name=PANEL, n=D["n"], m=len(D["names"]), datasets=fn, models=D["names"], dataset_ids=ids, prompts=D["prompts"], rewards=D["r"], costs=D["c"], _fam=D["fam"], _provenance=prov)
    return p, np.asarray(D["x"]), D["r"], D["c"], D["fam"]


def main():
    stem = sys.argv[1]; mod = importlib.import_module(stem)
    mod.load = load_sixfam
    if hasattr(mod, "load_sprout_raw20"): mod.load_sprout_raw20 = load_sixfam
    out_root = HERE.parent / "results" / "heldout_sixfam"; out_root.mkdir(parents=True, exist_ok=True)
    mod.ISO = out_root   # their scripts write to ISO/"results"/<panel>/family_<method>/
    sys.argv = [stem + ".py"] + sys.argv[2:]; mod.main()


if __name__ == "__main__":
    main()
