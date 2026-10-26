"""Representation registry.

A representation is an ``N x D`` float32 matrix aligned with a panel's rows.
Two sources exist:

* ``cached``: arrays already produced by the project (all L2-normalised, so the
  norm is unavailable and the truncation policy is the project's historical one).
* ``computed``: arrays produced by ``scripts/embed_gpu.py`` for this analysis
  (unnormalised, tail-preserving truncation, both text fields).

Family labels drive the report: ``contrastive`` (sentence encoders trained with
InfoNCE-style objectives) and ``control`` (surface features / hashing / domain
identity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import ANALYSIS_ROOT, RUNTIME_ROOT

FEATURES_ROOT = ANALYSIS_ROOT / "features"


@dataclass(frozen=True)
class RepSpec:
    name: str
    path: Path
    family: str        # contrastive | hidden | control
    source: str        # cached | computed
    text_field: str    # prompt | origin_query
    normalized: bool
    note: str = ""


# Historical caches (normalised).  Keyed by panel.
_CACHED: dict[str, list[RepSpec]] = {
    "routerbench-0shot": [
        RepSpec("historical/gte-qwen2-7b@prompt", RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot-gte.npy",
                "contrastive", "cached", "prompt", True, "project's earlier run: normalised, right-truncated at 8192 (all prompts fit)"),
    ],
}

# Computed representations (unnormalised) live under
# features/<panel>/<text_field>/<rep>.npy with a sidecar .json.
_FAMILY_BY_PREFIX = {
    "minilm": "contrastive", "bge": "contrastive", "gte": "contrastive", "e5": "contrastive",
    "qwen3emb": "contrastive", "hash": "control",
}


def _computed(panel: str) -> list[RepSpec]:
    specs = []
    root = FEATURES_ROOT / panel
    if not root.exists():
        return specs
    for field_dir in sorted(root.iterdir()):
        if not field_dir.is_dir():
            continue
        for npy in sorted(field_dir.glob("*.npy")):
            rep = npy.stem
            family = next((fam for prefix, fam in _FAMILY_BY_PREFIX.items() if rep.startswith(prefix)), "contrastive")
            note = ""
            meta = npy.with_suffix(".npy.json")
            if meta.exists():
                try:
                    note = json.loads(meta.read_text()).get("note", "")
                except json.JSONDecodeError:
                    note = ""
            specs.append(RepSpec(f"computed/{rep}@{field_dir.name}", npy, family, "computed",
                                 field_dir.name, False, note))
    return specs


def list_reps(panel: str, include_cached: bool = True) -> list[RepSpec]:
    specs = list(_CACHED.get(panel, [])) if include_cached else []
    specs.extend(_computed(panel))
    return [s for s in specs if s.path.exists()]


def find_rep(panel: str, name: str) -> RepSpec:
    for spec in list_reps(panel):
        if spec.name == name:
            return spec
    raise KeyError(f"{name} not available for {panel}")


def load_rep(spec: RepSpec, n_expected: int, keep: np.ndarray | None = None) -> np.ndarray:
    """Load a feature matrix; matrices computed on the full pinned panel are
    filtered with the panel's keep mask so rows stay aligned."""
    matrix = np.load(spec.path, mmap_mode="r")
    if keep is not None and matrix.shape[0] == len(keep):
        matrix = matrix[keep]
    if matrix.shape[0] != n_expected:
        raise ValueError(f"{spec.name}: {matrix.shape[0]} rows, panel has {n_expected}")
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{spec.name}: non-finite values")
    return matrix


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace("@", "_AT_")
