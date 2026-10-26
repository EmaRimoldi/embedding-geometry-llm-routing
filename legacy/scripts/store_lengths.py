"""Store per-query token lengths (BERT-family and Qwen2 tokenizers) so the report
can evaluate on window-controlled subsets. Writes features/<panel>/<field>/lengths.npz."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

bert = AutoTokenizer.from_pretrained(str(RUNTIME_ROOT / "artifacts/models/all-MiniLM-L12-v2__3392ab41"))
qwen = AutoTokenizer.from_pretrained(str(RUNTIME_ROOT / "artifacts/models/Qwen2.5-7B-Instruct__a09a3545"))
for panel_name, fields in [("routerbench-0shot", ["prompt"])]:
    panel = load_panel(panel_name)
    for field in fields:
        texts = [str(t) for t in panel.text(field)]
        lb = np.array([len(bert(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts])
        lq = np.array([len(qwen(t, add_special_tokens=False, truncation=False)["input_ids"]) for t in texts])
        out = ANALYSIS_ROOT / "features" / panel_name / field / "lengths.npz"
        np.savez(out, bert_tokens=lb, qwen_tokens=lq)
        print(panel_name, field, "fits256(bert) %.3f fits512 %.3f fits4096(qwen) %.3f" % ((lb <= 256).mean(), (lb <= 512).mean(), (lq <= 4096).mean()))
