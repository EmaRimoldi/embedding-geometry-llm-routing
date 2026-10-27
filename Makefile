PYTHON ?= python3
COMPUTE_PYTHON ?= .venv/bin/python
DOWNLOAD_PYTHON ?= .venv-download/bin/python

.PHONY: help verify test smoke download data controls diagnostics tables package

help:
	@echo "verify: saved-evidence and paper checks (Python standard library)"
	@echo "test / smoke: CPU tests and synthetic execution checks (compute environment)"
	@echo "download: all pinned public data AND model weights (tens of GB)"
	@echo "data: prepare both benchmarks after downloading inputs"
	@echo "controls / diagnostics: fresh complete CPU analyses of canonical public outcomes"
	@echo "tables: regenerate seven numerical tables from frozen reports"
	@echo "package: refresh the release manifest and deterministic ZIP"

verify:
	"$(PYTHON)" -B scripts/verify_release.py
	"$(PYTHON)" -B scripts/reproduce_results.py
	"$(PYTHON)" -B scripts/check_paper_results.py

test:
	"$(COMPUTE_PYTHON)" -B -m pytest -q tests

smoke:
	"$(COMPUTE_PYTHON)" -B scripts/smoke_baselines.py
	"$(COMPUTE_PYTHON)" -B scripts/smoke_nvidia.py
	"$(COMPUTE_PYTHON)" -B scripts/smoke_pca.py
	"$(COMPUTE_PYTHON)" -B scripts/smoke_pca.py --rows 160 --source-rows 120 --features 128
	"$(COMPUTE_PYTHON)" -B scripts/smoke_orchestration.py

download:
	"$(DOWNLOAD_PYTHON)" -B src/download_assets.py

data:
	"$(COMPUTE_PYTHON)" -B src/prepare_data.py routerbench
	"$(COMPUTE_PYTHON)" -B src/prepare_data.py sprout

controls:
	"$(COMPUTE_PYTHON)" -B scripts/reproduce_outcome_controls.py

diagnostics:
	"$(COMPUTE_PYTHON)" -B scripts/reproduce_cpu_diagnostics.py

tables:
	"$(COMPUTE_PYTHON)" -B scripts/render_saved_tables.py

package:
	"$(PYTHON)" -B scripts/build_submission.py
	"$(PYTHON)" -B scripts/verify_release.py
