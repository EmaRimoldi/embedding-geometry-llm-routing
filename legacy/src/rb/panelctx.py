"""Panel context: which panel the experiment scripts run on, and the few pieces
that are panel-specific (results root, number of families, topic cells, the
cheap model whose role Mistral-7B plays on RouterBench, surface length features,
the "core" representation).

Selected by the environment variable ``E4_PANEL`` (default ``routerbench-0shot``,
so every script behaves exactly as before when the variable is unset).
LLMRouterBench results go to ``experiments4/llmrouterbench/results/<panel>/``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .data import ANALYSIS_ROOT, RUNTIME_ROOT, Panel
from .reps import find_rep, load_rep

PANEL = os.environ.get("E4_PANEL", "routerbench-0shot")
IS_ROUTERBENCH = PANEL.startswith("routerbench")
# The representation every experiment reads (E4_REP), default the harness-prompt embedding; when
# another representation is selected, results go to a sibling root named after it (E4_RESULTS_SUFFIX
# overrides the name), so the default outputs are never overwritten.
REP = os.environ.get("E4_REP", "computed/gte-qwen2-7b@prompt")
_SUFFIX = os.environ.get("E4_RESULTS_SUFFIX", "" if REP == "computed/gte-qwen2-7b@prompt" else "_" + REP.split("@")[-1])
RESULTS_ROOT = ((ANALYSIS_ROOT / ("results" + _SUFFIX)) if IS_ROUTERBENCH
                else (ANALYSIS_ROOT / "llmrouterbench" / ("results" + _SUFFIX) / PANEL.replace("llmrouterbench-", "")))
# minilm/ study: an explicit (absolute) results root overrides the rule above; REP_NAME is the file stem of the
# representation (the probe embeddings are stored as <REP_NAME>__<variant>.npy); E4_OUT overrides the output
# folder of the probe scripts that live under results/<probe>/.  Unset => behaviour unchanged.
if os.environ.get("E4_RESULTS_ROOT"):
    RESULTS_ROOT = Path(os.environ["E4_RESULTS_ROOT"])
REP_NAME = REP.split("/")[-1].split("@")[0]
OUT_OVERRIDE = Path(os.environ["E4_OUT"]) if os.environ.get("E4_OUT") else None

# Number of task families (drives K in the K = #families clusterings).
N_FAM = {"routerbench-0shot": 7, "routerbench-5shot": 7, "llmrouterbench-performance-cost": 10, "llmrouterbench-performance-cost-le512": 9, "llmrouterbench-performance": 15}[PANEL]

# Families with a finer released label: RouterBench MMLU subjects / HellaSwag sources;
# LLMRouterBench MMLU-Pro categories / ArenaHard categories (llmrouterbench/topics/).
TOPIC_FAMILIES = ("mmlu", "hellaswag") if IS_ROUTERBENCH else ("mmlupro", "arenahard")

# The cheap model the oracle over-uses (exp29-31): Mistral-7B on RouterBench; on the
# LLMRouterBench performance-cost panel the model with the lowest median cost per query.
CHEAP_MODEL = "mistral-7b-chat" if IS_ROUTERBENCH else {"llmrouterbench-performance-cost": "qwen3-235b-a22b-2507", "llmrouterbench-performance-cost-le512": "qwen3-235b-a22b-2507", "llmrouterbench-performance": "Qwen3-8B"}[PANEL]

_HS = re.compile(r"\[(header|title|step|substeps)\]")


def _rb_record_prefix(panel: Panel) -> np.ndarray:
    rec = np.load(RUNTIME_ROOT / "artifacts/external/routerbench/routerbench-0shot.npz")["record_indices"].astype(str)
    rec = rec[panel.keep] if panel.keep is not None else rec
    return np.array([s_.split(".")[0] for s_ in rec])


def topic_str(panel: Panel, convention: str = "cell") -> np.ndarray:
    """Per-row topic label (string).  ``cell``: family x (MMLU subject | HellaSwag source),
    every other row = its family (the exp17-exp32 convention).  ``record_prefix``: the
    release's record-index prefix (MMLU subject, MT-Bench category, HellaSwag unsplit),
    the exp12/exp14/exp15 convention.  On LLMRouterBench both give the released topics."""
    if IS_ROUTERBENCH:
        pre = _rb_record_prefix(panel)
        if convention == "record_prefix":
            return pre
        src = np.array(["wikihow" if _HS.search(str(t)) else "activitynet" for t in panel.prompts])
        return np.where(panel.dataset_ids == "mmlu", pre, np.where(panel.dataset_ids == "hellaswag", "hellaswag/" + src, panel.dataset_ids))
    t = json.load(open(ANALYSIS_ROOT / "llmrouterbench" / "topics" / f"{panel.name}.json"))
    assert t["rows"] == (len(panel.keep) if panel.keep is not None else panel.n)
    arr = np.array(t["topic"], dtype=object)
    return arr[panel.keep] if panel.keep is not None else arr


def small_families(panel: Panel, per_family: int) -> tuple[str, ...]:
    """Families with fewer than ``per_family`` rows (excluded from balanced subsamples)."""
    return tuple(d for d in panel.datasets if int((panel.dataset_ids == d).sum()) < per_family)


def lengths_features(panel: Panel):
    """Log token counts as extra surface features (exp18).  RouterBench: Qwen2 + BERT
    tokenizers; LLMRouterBench: the embedder's own tokenizer."""
    L = np.load(ANALYSIS_ROOT / "features" / panel.name / "prompt" / "lengths.npz")
    if "qwen_tokens" in L.files:
        cols, names = [L["qwen_tokens"], L["bert_tokens"]], ["log_qwen_tokens", "log_bert_tokens"]
    else:
        cols, names = [L["gte_tokens"]], ["log_gte_tokens"]
    cols = [c[panel.keep] if panel.keep is not None else c for c in cols]
    return np.column_stack([np.log1p(c) for c in cols]).astype(np.float64), names


def core_rep(panel: Panel) -> np.ndarray:
    """The 'deformatted core' representation of exp32: on RouterBench the embedding of
    the template-stripped text (results/deformatting); on LLMRouterBench the embedding
    of the release's ``origin_query`` (the question without harness instructions)."""
    if IS_ROUTERBENCH:
        return np.load(ANALYSIS_ROOT / "results" / "deformatting" / "panel" / "gte-qwen2-7b.npy").astype(np.float32)
    return load_rep(find_rep(panel.name, "computed/gte-qwen2-7b@origin_query"), panel.n, panel.keep).astype(np.float32)
