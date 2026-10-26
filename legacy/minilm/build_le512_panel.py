"""minilm/ study, step 0: the LLMRouterBench performance-cost panel restricted to the rows that fit
MiniLM's 512-token window (PI, 2026-09-16: "take out queries > 512 tokens").  Token count = the
MiniLM (BERT WordPiece) tokenizer with special tokens, the same count the RouterBench keep mask
was built from (that panel already satisfies the rule: max 510 tokens among its 34,661 rows).
Writes features/llmrouterbench-performance-cost-le512/keep_mask.npz, links the representations,
lengths and topic files of the parent panel into the new panel's folders (row filtering happens
at load time through the keep mask), and minilm/panel_le512.md."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SRC, DST = "llmrouterbench-performance-cost", "llmrouterbench-performance-cost-le512"; WINDOW = 512

def link(target: Path, name: Path) -> None:
    name.parent.mkdir(parents=True, exist_ok=True)
    if name.is_symlink() or name.exists():
        return
    os.symlink(os.path.relpath(target, name.parent), name)

def main() -> None:
    p = load_panel(SRC); tok = AutoTokenizer.from_pretrained(str(RUNTIME_ROOT / "artifacts/models/all-MiniLM-L12-v2__3392ab41"))
    lb = np.array([len(tok(str(t), add_special_tokens=True, truncation=False)["input_ids"]) for t in p.prompts]); keep = lb <= WINDOW
    fdir = ANALYSIS_ROOT / "features" / DST; fdir.mkdir(parents=True, exist_ok=True)
    np.savez(fdir / "keep_mask.npz", keep=keep, bert_tokens=lb, rule=np.array(f"MiniLM tokenizer, special tokens included, <= {WINDOW}"))
    for f in ["gte-qwen2-7b.npy", "gte-qwen2-7b.npy.json", "hash-bow.npy", "hash-bow.npy.json", "lengths.npz", "minilm-l12.npy", "minilm-l12.npy.json"]:
        link(ANALYSIS_ROOT / "features" / SRC / "prompt" / f, fdir / "prompt" / f)
    for sub in ["topics", "topics_llm"]:
        link(ANALYSIS_ROOT / "llmrouterbench" / sub / f"{SRC}.json", ANALYSIS_ROOT / "llmrouterbench" / sub / f"{DST}.json")
    fam = p.dataset_ids; rows = []
    for f in sorted(set(fam)):
        m = fam == f; rows.append((f, int(m.sum()), int((~keep[m]).sum()), float((~keep[m]).mean() * 100), float(np.median(lb[m]))))
    md = ["# LLMRouterBench performance-cost, rows <= 512 MiniLM tokens (`llmrouterbench-performance-cost-le512`)", "",
          f"Rule: MiniLM WordPiece token count with special tokens <= {WINDOW}. Kept {int(keep.sum())} of {len(keep)} rows ({(~keep).mean()*100:.1f}% removed); "
          f"families left: {sum(1 for r in rows if r[1] > r[2])} of {len(rows)}.", "",
          "family | rows | removed | removed % | median tokens", "---|---|---|---|---"] + [f"{r[0]} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]:.0f}" for r in rows] + [
          "", "RouterBench 0-shot: the standard filtered panel already satisfies the rule (34,661 rows, max 510 BERT tokens); no further row is removed."]
    (ROOT / "minilm" / "panel_le512.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    q = load_panel(DST); assert q.n == int(keep.sum()) and set(q.datasets) == {r[0] for r in rows if r[1] > r[2]}; print("reload ok:", q.n, "rows,", len(q.datasets), "families")

if __name__ == "__main__":
    main()
