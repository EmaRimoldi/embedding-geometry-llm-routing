"""Routing panels, outcome targets and cheap text controls.

Everything here is read-only over the hash-pinned ``.npz`` panels already used
by the project.  No outcome ever enters a representation; targets are derived
once and reused by every probe so that all representations are scored against
byte-identical labels.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNTIME_ROOT = Path(os.environ.get("RB_RUNTIME_ROOT", Path.home() / "cluster/scratch/project"))
ANALYSIS_ROOT = Path(os.environ.get("E4_ROOT", RUNTIME_ROOT / "experiments4"))

# Success threshold: rewards are {0, 0.5, 1} (0.5 only on ArenaHard ties). A tie
# with the reference answer is counted as acceptable; the strict variant is
# reported as a sensitivity check in the report.
SUCCESS_THRESHOLD = 0.5

PANELS = {
    # name: (npz path, short description).  Read-only, hash-pinned by the project.
    "routerbench-0shot": (RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz",
                          "RouterBench 0-shot: 34,778 queries, 11 models, 7 families (question only, no few-shot prefix)"),
    "routerbench-5shot": (RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-5shot.npz",
                          "RouterBench 5-shot: 34,792 prompts with the upstream few-shot prefix (kept supported; GTE reads them whole)"),
    # LLMRouterBench (Li et al., ACL 2026 Findings; release NPULH/LLMRouterBench@0e5af1b, repo c77cb050).
    # Frozen by the project from the raw bench-release archive; verified cell by cell in
    # experiments4/llmrouterbench/verification/.  The performance panel keeps the archive's
    # nominal per-call costs (the benchmark's performance setting itself ignores cost).
    "llmrouterbench-performance-cost": (RUNTIME_ROOT / "artifacts/performance-cost.npz",
                                        "LLMRouterBench performance-cost: 12,446 queries, 13 flagship models with API prices, 10 datasets"),
    "llmrouterbench-performance": (RUNTIME_ROOT / "artifacts/performance-paper.npz",
                                   "LLMRouterBench performance: 11,480 queries, 20 lightweight (~7-9B) models, 15 datasets"),
}


def _decode(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    raw = data.tobytes()
    return np.asarray(
        [raw[offsets[i]:offsets[i + 1]].decode("utf-8") for i in range(len(offsets) - 1)],
        dtype=object,
    )


@dataclass
class Panel:
    name: str
    query_ids: np.ndarray
    dataset_ids: np.ndarray
    prompts: np.ndarray
    origin_queries: np.ndarray
    models: np.ndarray
    rewards: np.ndarray      # N x M float32, 0 where unavailable
    costs: np.ndarray        # N x M float64, inf where unavailable
    available: np.ndarray    # N x M bool
    keep: np.ndarray | None = None   # row filter applied to the pinned panel (None = all rows)
    split_ids: np.ndarray | None = None       # LLMRouterBench: release split folder per row
    release_missing: np.ndarray | None = None  # LLMRouterBench: cells with no record in the release (filled 0/0)
    record_indices: np.ndarray | None = None  # LLMRouterBench: record index inside the split

    def provenance(self) -> dict:
        """Identity of the exact table this object holds; written into every result file."""
        import hashlib
        return {
            "panel": self.name,
            "filtered": self.keep is not None,
            "rows": int(self.n),
            "rows_in_release": int(len(self.keep)) if self.keep is not None else int(self.n),
            "keep_mask_sha256": hashlib.sha256(self.keep.tobytes()).hexdigest() if self.keep is not None else None,
            "families": {d: int((self.dataset_ids == d).sum()) for d in self.datasets},
        }

    @property
    def n(self) -> int:
        return len(self.query_ids)

    @property
    def m(self) -> int:
        return len(self.models)

    @property
    def datasets(self) -> list[str]:
        return sorted(set(self.dataset_ids.tolist()))

    def dataset_index(self) -> np.ndarray:
        lookup = {d: i for i, d in enumerate(self.datasets)}
        return np.asarray([lookup[d] for d in self.dataset_ids], dtype=np.int64)

    def text(self, field: str) -> np.ndarray:
        if field == "prompt":
            return self.prompts
        if field == "origin_query":
            return self.origin_queries
        raise ValueError(field)


def keep_mask(name: str) -> np.ndarray | None:
    """Row filter guaranteeing that no encoder in this phase truncates any query.

    Built once by scripts/build_keep_mask.py from per-query token counts under
    every encoder tokenizer used: a query is kept iff it fits the smallest
    architectural window (512 for the BERT-family encoders, prefix included).
    Stored as features/<panel>/keep_mask.npz; absent => no filtering.
    """
    path = ANALYSIS_ROOT / "features" / name / "keep_mask.npz"
    if not path.exists():
        return None
    return np.load(path)["keep"].astype(bool)


def load_panel(name: str, filtered: bool = True, fill_missing: bool = True) -> Panel:
    """``fill_missing`` applies, to the LLMRouterBench panels only, the convention of
    the official loader (``baselines/adaptors/common.py::fill_missing_models_scores``
    and ``data_loader.py``: a failed request or a model with no record for a query
    is scored 0).  Such cells become available with reward 0 and cost 0; the release
    mask is kept in ``panel.release_missing``.  RouterBench panels are unaffected."""
    path, _ = PANELS[name]
    z = np.load(path, allow_pickle=False)
    keep = keep_mask(name) if filtered else None
    sel = (lambda a: a[keep]) if keep is not None else (lambda a: a)
    panel = Panel(
        name=name,
        query_ids=sel(z["query_ids"]),
        dataset_ids=sel(z["dataset_ids"].astype(str)),
        prompts=sel(_decode(z["prompts_utf8"], z["prompts_offsets"])),
        origin_queries=sel(_decode(z["origin_queries_utf8"], z["origin_queries_offsets"])),
        models=z["models"].astype(str),
        rewards=sel(z["rewards"].astype(np.float32)),
        costs=sel(z["costs"].astype(np.float64)),
        available=sel(z["available"].astype(bool)),
    )
    panel.keep = keep
    if fill_missing and name.startswith("llmrouterbench"):
        missing = ~panel.available
        panel.release_missing = missing
        panel.rewards = np.where(missing, 0.0, panel.rewards).astype(np.float32)
        panel.costs = np.where(missing, 0.0, panel.costs).astype(np.float64)
        panel.available = np.ones_like(panel.available)
    if "split_ids" in z.files:
        panel.split_ids = sel(z["split_ids"].astype(str))
        panel.record_indices = sel(z["record_indices"].astype(str))
    assert panel.rewards.shape == panel.available.shape == panel.costs.shape
    assert len(panel.prompts) == panel.n == len(panel.origin_queries)
    return panel


@dataclass
class Targets:
    """Outcome-derived quantities. All arrays are aligned with the panel rows."""

    reward: np.ndarray            # N x M, NaN where unavailable
    success: np.ndarray           # N x M, NaN where unavailable, else 0/1
    n_avail: np.ndarray           # N
    difficulty: np.ndarray        # N: 1 - mean available reward
    n_success: np.ndarray         # N: number of available models with success
    oracle_reward: np.ndarray     # N: max available reward
    disagreement: np.ndarray      # N bool: not all available rewards equal
    hard: np.ndarray              # N bool: solvable but by <= 25% of available models
    unsolvable: np.ndarray        # N bool: no available model succeeds
    cheapest_sufficient: np.ndarray  # N int: cheapest model attaining oracle reward (-1 if none)
    dataset_index: np.ndarray     # N int
    model_cost_scale: np.ndarray  # M: median observed cost per model (for cost-aware decisions)

    def subset_masks(self) -> dict[str, np.ndarray]:
        return {
            "all": np.ones_like(self.disagreement),
            "disagreement": self.disagreement,
            "hard": self.hard,
        }


def build_targets(panel: Panel, success_threshold: float = SUCCESS_THRESHOLD) -> Targets:
    avail = panel.available
    reward = np.where(avail, panel.rewards, np.nan).astype(np.float64)
    success = np.where(avail, (panel.rewards >= success_threshold).astype(np.float64), np.nan)
    n_avail = avail.sum(1)
    with np.errstate(invalid="ignore"):
        mean_reward = np.nanmean(reward, axis=1)
        oracle = np.nanmax(reward, axis=1)
        n_success = np.nansum(success, axis=1)
    difficulty = 1.0 - mean_reward
    rmin = np.nanmin(reward, axis=1)
    disagreement = (oracle - rmin) > 1e-9
    unsolvable = n_success == 0
    hard = (~unsolvable) & (n_success <= np.maximum(1, np.floor(0.25 * n_avail)))

    costs = np.where(avail, panel.costs, np.nan)
    with np.errstate(invalid="ignore"):
        cost_scale = np.nanmedian(costs, axis=0)
    cost_scale = np.where(np.isfinite(cost_scale) & (cost_scale > 0), cost_scale, np.nan)
    # Panels without cost information (all zeros) get a unit scale so that
    # cost-aware decisions degrade to pure-quality decisions.
    if np.all(~np.isfinite(cost_scale)):
        cost_scale = np.ones(panel.m)
    else:
        cost_scale = np.where(np.isfinite(cost_scale), cost_scale, np.nanmax(cost_scale))

    cheapest = np.full(panel.n, -1, dtype=np.int64)
    for i in range(panel.n):
        best = np.isclose(reward[i], oracle[i]) & avail[i]
        if not best.any():
            continue
        c = np.where(best, costs[i], np.inf)
        c = np.where(np.isfinite(c), c, np.inf)
        if np.all(~np.isfinite(c)):
            cheapest[i] = int(np.flatnonzero(best)[0])
        else:
            cheapest[i] = int(np.argmin(c))

    return Targets(
        reward=reward, success=success, n_avail=n_avail, difficulty=difficulty,
        n_success=n_success, oracle_reward=oracle, disagreement=disagreement, hard=hard,
        unsolvable=unsolvable, cheapest_sufficient=cheapest,
        dataset_index=panel.dataset_index(), model_cost_scale=cost_scale,
    )


_CODE_RE = re.compile(r"[{}\[\]();=<>+\-*/%&|^~`#]|\bdef\b|\bclass\b|\breturn\b|\bimport\b")
_MATH_RE = re.compile(r"\\[a-zA-Z]+|\$|\^|\\frac|\\sqrt|\d+\.\d+|[=+\-*/]")
_OPTION_RE = re.compile(r"(?m)^\s*(?:\(?[A-J]\)|[A-J][.:)])\s")


def text_controls(texts: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Cheap, outcome-free surface features used as a control representation.

    They are deliberately crude: the point is to know how much of any routing
    signal an embedder carries is already available from length and format.
    """

    rows = []
    for t in texts:
        n_char = len(t)
        n_ws = len(t.split())
        n_lines = t.count("\n") + 1
        n_digit = sum(ch.isdigit() for ch in t)
        n_code = len(_CODE_RE.findall(t))
        n_math = len(_MATH_RE.findall(t))
        n_opt = len(_OPTION_RE.findall(t))
        n_upper = sum(ch.isupper() for ch in t)
        n_q = t.count("?")
        rows.append([
            np.log1p(n_char), np.log1p(n_ws), np.log1p(n_lines),
            n_digit / max(n_char, 1), n_code / max(n_ws, 1), n_math / max(n_ws, 1),
            np.log1p(n_opt), n_upper / max(n_char, 1), np.log1p(n_q),
            np.log1p(n_char) / max(np.log1p(n_ws), 1e-6),  # mean word length proxy
        ])
    names = ["log_chars", "log_words", "log_lines", "digit_ratio", "code_ratio", "math_ratio",
             "log_options", "upper_ratio", "log_questions", "char_per_word"]
    return np.asarray(rows, dtype=np.float64), names
