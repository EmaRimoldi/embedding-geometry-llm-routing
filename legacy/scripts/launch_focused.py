"""Focused relaunch: the units that carry the pre-declared comparisons first.

Stage 1 (this launcher, default): transforms raw/whiten/unit/resid/onehot+raw/surf+raw/desurf x
decoders ridge/logisticcv/knn20/kmeans64 x all splits.  Stage 2 (--stage 2): mlp, knn5/50/100,
kmeans7/256 on raw/unit.  Existing units are skipped by run_probes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT  # noqa: E402
from rb.reps import list_reps  # noqa: E402

STAGES = {
    1: "--transforms raw whiten unit resid onehot+raw surf+raw desurf --decoders ridge logisticcv knn20 kmeans64",
    # RouterBench's published router configurations on the corrected inputs (fidelity rows) + our remaining grid.
    2: "--transforms raw unit --decoders knn5 knn10 knn40 knn50 knn100 mlp100 mlp100x100 mlp kmeans7 kmeans256",
}


def wanted(panel: str, name: str) -> bool:
    # Hidden-state comparators are out of scope in this phase.
    return "hidden" not in name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--panels", nargs="*", default=["routerbench-0shot"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    n = 0
    for panel in args.panels:
        names = [s.name for s in list_reps(panel)] + ["onehot", "controls"]
        for name in names:
            if not wanted(panel, name):
                continue
            extra = STAGES[args.stage]
            if name in ("onehot", "controls"):
                extra = "--decoders ridge logisticcv knn20" if args.stage == 1 else "--decoders mlp knn100"
            cmd = ["sbatch", "--parsable", f"--export=ALL,PANEL={panel},REP={name},EXTRA={extra}", "slurm/probes.sbatch"]
            if args.dry_run:
                print(panel, name)
                continue
            job = subprocess.run(cmd, capture_output=True, text=True, cwd=ANALYSIS_ROOT, check=True).stdout.strip()
            n += 1
            print(f"{panel} {name} -> {job}")
    print(f"launched {n}")


if __name__ == "__main__":
    main()
