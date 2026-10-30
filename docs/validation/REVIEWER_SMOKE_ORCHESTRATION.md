# Reviewer smoke: fresh-start orchestration

Commands for the current checkout run from the [scientific workspace](../../experiments/README.md). From the repository root, enter it with `cd experiments`.

## Scope

Executed `python3.11 -B scripts/smoke_orchestration.py` in the submission copy.
**All 17 check categories passed.** The harness uses tiny, explicitly synthetic
metadata/data placeholders and inert Python/Slurm command stubs. No original
research folder is accessed by the harness, and no real dataset download,
embedding extraction, model fit, Slurm submission or scheduler query occurs.

The complete machine-readable run is written to
`.cache/reviewer_smoke_orchestration/report.json`. Temporary fixtures are removed
when the harness exits. Re-running it regenerates the report.

## What was exercised

| Area | Checks |
|---|---|
| Initialization | All nine stages; required directory creation; relative internal links; repeated calls preserve exact config bytes and hashes. |
| Prerequisites | Missing baseline preparation, missing reduced provenance and stale NVIDIA sources fail before changing configs. |
| Frozen inputs | Altered utility data is rejected, not accepted by refreshing its checksum. |
| Partial progress | Baseline checkpoints, PCA caches and utility fit checkpoints block provenance changes even before a final result exists. |
| Failure preservation | Missing utility source files and existing incompatible source manifests leave the previous config and manifest intact. |
| Third-party sources | Modified vendor code is rejected against the published hashes. |
| Path safety | An external runtime symlink is rejected; stale inherited project/allocation directories do not redirect launchers. |
| Launchers | 59 actual Slurm shell files executed against inert Python/sbatch commands, including their argument handling and nested submission paths. |
| Accounting | Missing or empty study job metadata produces explicit unavailable scheduler accounting; the scheduler stub is forbidden from being called. |
| Coverage | Documented entrypoint files exist; 207 geometry, 46 baseline, 18 descriptive-clustering and 90 additional-clustering tasks match configurations; all geometry tasks occur exactly once. |

This is an orchestration smoke, not a statistical or numerical reproduction
certificate. A successful inert launcher does not prove GPU availability,
resource sufficiency or completion of the real worker.

## Concrete release fixes from the review

1. The initializer formerly checked only `result.json` before changing metadata.
   It now also catches incomplete fit/PCA output files. Existing source/log
   metadata is treated separately so an untouched new run can initialize.
2. NVIDIA initialization now verifies its baseline/reduced provenance
   prerequisites. Utility initialization verifies the NVIDIA source manifest.
3. Utility initialization formerly could change its config before discovering a
   missing source or incompatible manifest. It now validates dependencies and the
   prospective config hash first, then writes consistent metadata.
4. Layer-selection and controlled-format audit instructions now explicitly wait
   for their respective report jobs.
5. The historical common-layer NVIDIA compute report formerly required a
   `submission.json` absent from a fresh copy. It now preserves measured phase
   timing and explicitly marks unavailable scheduler accounting in JSON and
   Markdown. It never calls `sacct` with an empty study-job list.
6. The lookup report submission explicitly invokes Bash because Slurm's wrapped
   command may otherwise run under a POSIX shell without `source`.
7. Launchers source the environment from the documented working directory.
   The environment still derives its root from its own location. Inherited
   `PAPER_ROOT` and `SLURM_SUBMIT_DIR` cannot redirect a launcher executed from
   this copy's root.

These changes affect release initialization, scheduling or reporting behavior.
No grids, model architectures, objectives, selection rules, seeds, datasets,
splits or recorded scientific outcomes changed.

## Methods and provenance consistency

The fresh-run order is: asset preparation; canonical embeddings; geometry;
baseline preparation and reduced provenance; reduced MLP/Ridge checks;
candidate-layer extraction and source-fold PCA; final per-target utility probe,
training and audit. The older common-layer NVIDIA control and its CPU inference
benchmark remain separately labeled. The latter is not a benchmark of the final
per-target implementation.

Input checksums in the final utility configuration remain strict. Only the
copied NVIDIA-config link and source-provenance metadata are refreshed for the
new release. Original evidence remains under `results/evidence/`; runtime
manifests describe a new run, not a replacement assertion about old results.

## Archive integrity

At the start of this review, the current initializer and the then-frozen ZIP
entry were byte-identical (SHA256
`ac393a629333d8adeece45f179c20979b7df202cb47b1838df50360a3e9c0092`).
The fixes above subsequently changed release files; regenerate the allowlisted
archive and its integrity manifest before delivery. The release verifier and
fresh-extraction checks provide the final archive integrity evidence.
