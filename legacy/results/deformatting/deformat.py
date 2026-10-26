"""Shared pieces of the deformatting chapter: the stripping rule and the deformatted-panel paths."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np

PANEL_DIR = Path(__file__).resolve().parent / "panel"      # results/deformatting/panel/


def strip_rule(texts: list[str]) -> tuple[str, str]:
    """prefix = longest string shared by all texts at the start, trimmed to end at a line break;
    suffix = longest string shared by all texts at the end, trimmed to start at a line break."""
    pre = os.path.commonprefix(texts)
    if pre:
        k = pre.rfind("\n"); pre = pre[:k + 1] if k >= 0 else ""
    suf = os.path.commonprefix([t[::-1] for t in texts])[::-1]
    if suf:
        k = suf.find("\n"); suf = suf[k:] if k >= 0 else ""
    return pre, suf


def deformat(prompts: list[str], dataset_ids: np.ndarray, families) -> tuple[list[str], dict]:
    core = [None] * len(prompts); rule = {}
    for fam in families:
        idx = np.flatnonzero(dataset_ids == fam); T = [prompts[j] for j in idx]; pre, suf = strip_rule(T)
        for j, t in zip(idx, T):
            assert t.startswith(pre) and (not suf or t.endswith(suf))
            core[j] = t[len(pre):len(t) - len(suf)] if suf else t[len(pre):]
        rule[str(fam)] = {"n": int(len(idx)), "prefix": pre, "suffix": suf, "prefix_chars": len(pre), "suffix_chars": len(suf),
                          "prompt_chars_median": float(np.median([len(t) for t in T])), "core_chars_median": float(np.median([len(core[j]) for j in idx])),
                          "example_prompt": T[0][:400], "example_core": core[idx[0]][:400]}
    return core, rule


def load_core_texts():
    z = np.load(PANEL_DIR / "texts.npz"); b = z["utf8"].tobytes(); off = z["offsets"]
    return [b[off[i]:off[i + 1]].decode("utf-8") for i in range(len(off) - 1)]
