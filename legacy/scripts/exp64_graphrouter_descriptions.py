"""Outcome-free RouterBench model and task descriptions for a GraphRouter transfer.
The original adaptor supplies descriptions only for LLMRouterBench. These texts
use model identifiers and task definitions, fixed before looking at routed
outcomes. The same frozen GTE pipeline embeds all descriptions.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

E4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E4 / "src"))
sys.path.insert(0, str(E4 / "scripts"))
from rb.data import load_panel
from embed_gpu import ST_MODELS, _register_optional_flash_attention_stub

TASKS = {
    "arc-challenge": "ARC Challenge: multiple-choice science questions.",
    "grade-school-math": "GSM8K: grade-school mathematics word problems.",
    "hellaswag": "HellaSwag: choose a plausible continuation of a situation.",
    "mbpp": "MBPP: write Python programs for short programming tasks.",
    "mmlu": "MMLU: multiple-choice questions across academic subjects.",
    "mtbench": "MT-Bench: multi-turn conversational requests.",
    "winogrande": "WinoGrande: resolve the referent in a sentence.",
}
OUT = E4 / "results" / "routerbench_baselines_exp64" / "graphrouter_inputs"

def main():
    panel = load_panel("routerbench-0shot")
    models = [str(m) for m in panel.models]
    tasks = list(panel.datasets)
    assert set(tasks) == set(TASKS)
    model_texts = ["Language model: " + m.replace("/", " ").replace("-", " ") + "." for m in models]
    task_texts = [TASKS[t] for t in tasks]
    _register_optional_flash_attention_stub()
    import torch
    from sentence_transformers import SentenceTransformer
    path, max_seq, prompt, dtype, batch = ST_MODELS["gte-qwen2-7b"]
    model = SentenceTransformer(path, trust_remote_code=True, device="cuda",
        model_kwargs={"torch_dtype": getattr(torch, dtype), "attn_implementation": "sdpa"})
    model.max_seq_length = max_seq
    model.tokenizer.truncation_side = "left"
    emb_model = model.encode(model_texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=False)
    emb_task = model.encode(task_texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=False)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "descriptions.npz", models=np.array(models), tasks=np.array(tasks),
             llm_embedding=emb_model.astype(np.float32), task_embedding=emb_task.astype(np.float32))
    (OUT / "descriptions.json").write_text(json.dumps(
        {"models": models, "tasks": tasks, "model_texts": model_texts, "task_texts": task_texts,
         "embedder": path, "description_source": "fixed outcome-free identifiers and task definitions"}, indent=1))
    print(OUT / "descriptions.npz", emb_model.shape, emb_task.shape, flush=True)

if __name__ == "__main__":
    main()
