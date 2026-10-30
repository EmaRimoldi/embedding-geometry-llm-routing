PYTHON ?= python3
COMPUTE_PYTHON ?= experiments/.venv/bin/python
DOWNLOAD_PYTHON ?= experiments/.venv-download/bin/python

.PHONY: help check-docs manifest verify compare-baselines test smoke download data controls diagnostics tables package

help:
	@echo "check-docs: offline documentation links, anchors and versioned command paths"
	@echo "manifest: refresh current file fingerprints without changing a release ZIP"
	@echo "verify: saved-evidence and paper checks (Python standard library)"
	@echo "compare-baselines: compare new main-baseline results; incomplete coverage exits 2"
	@echo "test / smoke: CPU tests and synthetic execution checks (compute environment)"
	@echo "download: all pinned public data AND model weights (tens of GB)"
	@echo "data: prepare both benchmarks after downloading inputs"
	@echo "controls / diagnostics: fresh complete CPU analyses of canonical public outcomes"
	@echo "tables: regenerate seven numerical tables from frozen reports"
	@echo "package: refresh the release manifest and deterministic ZIP"

check-docs:
	"$(PYTHON)" -B experiments/scripts/check_docs.py

manifest:
	"$(PYTHON)" -B experiments/scripts/build_submission.py --manifest-only

verify: check-docs
	"$(PYTHON)" -B experiments/scripts/verify_release.py
	"$(PYTHON)" -B experiments/scripts/reproduce_results.py
	"$(PYTHON)" -B experiments/scripts/check_paper_results.py

compare-baselines:
	"$(PYTHON)" -B experiments/scripts/compare_baseline_results.py --fresh-results experiments/results

test:
	"$(COMPUTE_PYTHON)" -B -m pytest -q experiments/tests

smoke:
	"$(COMPUTE_PYTHON)" -B experiments/scripts/smoke_baselines.py
	"$(COMPUTE_PYTHON)" -B experiments/scripts/smoke_nvidia.py
	"$(COMPUTE_PYTHON)" -B experiments/scripts/smoke_pca.py
	"$(COMPUTE_PYTHON)" -B experiments/scripts/smoke_pca.py --rows 160 --source-rows 120 --features 128
	"$(COMPUTE_PYTHON)" -B experiments/scripts/smoke_orchestration.py

download:
	"$(DOWNLOAD_PYTHON)" -B experiments/src/download_assets.py

data:
	"$(COMPUTE_PYTHON)" -B experiments/src/prepare_data.py routerbench
	"$(COMPUTE_PYTHON)" -B experiments/src/prepare_data.py sprout

controls:
	"$(COMPUTE_PYTHON)" -B experiments/scripts/reproduce_outcome_controls.py

diagnostics:
	"$(COMPUTE_PYTHON)" -B experiments/scripts/reproduce_cpu_diagnostics.py

tables:
	"$(COMPUTE_PYTHON)" -B experiments/scripts/render_saved_tables.py

package:
	"$(PYTHON)" -B experiments/scripts/build_submission.py
	"$(PYTHON)" -B experiments/scripts/verify_release.py
