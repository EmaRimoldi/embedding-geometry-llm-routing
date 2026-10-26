"""Frozen-LLM reader representations (GPU job): hidden states of Qwen2.5-7B-Instruct on the routed
prompt text of a panel, plus probe-free surprise scalars from the same forward pass.

For every prompt (the text the benchmark sends, as the embedder saw it; left-truncated at 8,192
tokens like every representation of the study; no chat template, no generation):
  * hidden states at layers 0 (token embeddings, the lexical control), 7, 14, 21 and 28 (final)
    of the 28-layer model, pooled two ways: mean over the prompt's tokens and the last token;
    saved as float16, shape (n, 3584), features/<panel>/prompt/qwen7b_hidden/L<layer>_<pool>.npy;
  * scalars: mean log-probability the model assigns to the prompt's own tokens, mean entropy of its
    next-token distribution, and both restricted to the last 32 tokens, plus the token count;
    saved in features/<panel>/prompt/qwen7b_hidden/prompt_logprob_stats.npz.
No outcome is used.  Usage: extract_llm_hidden.py --panel routerbench-0shot"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402

MODEL = RUNTIME_ROOT / "artifacts/models/Qwen2.5-7B-Instruct__a09a3545"
LAYERS = (0, 7, 14, 21, 28)
MAX_LEN = 8192
LAST_K = 32


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--panel", required=True); ap.add_argument("--field", default="prompt"); a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    panel = load_panel(a.panel); texts = [str(t) for t in panel.text(a.field)]; n = len(texts)
    out_dir = ANALYSIS_ROOT / "features" / a.panel / a.field / "qwen7b_hidden"; out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL)); tok.padding_side = "left"; tok.truncation_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", trust_remote_code=False); model.eval()
    H = model.config.hidden_size
    hid = {(L, p): np.zeros((n, H), dtype=np.float16) for L in LAYERS for p in ("mean", "last")}
    stats = np.zeros((n, 5), dtype=np.float32)          # mean_logprob, mean_entropy, last32_logprob, last32_entropy, n_tokens
    lengths = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]); order = np.argsort(lengths)
    t0 = time.time(); s = 0; done = 0
    while s < n:
        L = int(min(lengths[order[min(s + 15, n - 1)]], MAX_LEN)); bs = 16 if L <= 512 else 8 if L <= 1500 else 2 if L <= 4096 else 1
        idx = [int(i) for i in order[s:s + bs]]; s += bs
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to("cuda")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)          # (b, T, 1); left padding -> last position is a real token
        for Lr in LAYERS:
            h = out.hidden_states[Lr].to(torch.float32)
            mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1); last = h[:, -1, :]
            for j, i in enumerate(idx):
                hid[(Lr, "mean")][i] = mean[j].cpu().numpy().astype(np.float16); hid[(Lr, "last")][i] = last[j].cpu().numpy().astype(np.float16)
        # surprise scalars: next-token distribution at position t predicts token t+1
        logits = out.logits; ids = enc["input_ids"]; am = enc["attention_mask"]
        for j, i in enumerate(idx):
            valid = am[j].bool(); pos = torch.nonzero(valid).squeeze(-1)
            if len(pos) < 2:
                stats[i] = [0, 0, 0, 0, len(pos)]; continue
            lp_list, ent_list = [], []; src = pos[:-1]                      # positions that have a next token
            for c in range(0, len(src), 1024):
                sl = src[c:c + 1024]; lg = logits[j, sl, :].to(torch.float32); logp = torch.log_softmax(lg, dim=-1)
                nxt = ids[j, sl + 1]; lp_list.append(logp.gather(1, nxt.unsqueeze(1)).squeeze(1)); ent_list.append(-(logp.exp() * logp).sum(-1))
            lp = torch.cat(lp_list); ent = torch.cat(ent_list)
            stats[i] = [float(lp.mean()), float(ent.mean()), float(lp[-LAST_K:].mean()), float(ent[-LAST_K:].mean()), float(len(pos))]
        done += len(idx)
        if (done // bs) % 100 == 0 or done == n:
            print(f"{done}/{n} ({time.time() - t0:.0f}s, batch {bs}, len {L})", flush=True)
        del out, logits
    for (Lr, p), arr in hid.items():
        np.save(out_dir / f"L{Lr}_{p}.npy", arr, allow_pickle=False)
    np.savez(out_dir / "prompt_logprob_stats.npz", stats=stats, columns=np.array(["mean_logprob", "mean_entropy", "last32_logprob", "last32_entropy", "n_tokens"]))
    json.dump({**panel.provenance(), "model": str(MODEL), "field": a.field, "layers": list(LAYERS), "poolings": ["mean", "last"], "max_len": MAX_LEN, "truncation": "left",
               "chat_template": False, "dtype": "bfloat16 forward, float16 on disk", "hidden_size": H, "texts_truncated": int((lengths > MAX_LEN).sum()),
               "seconds": time.time() - t0}, open(out_dir / "meta.json", "w"), indent=1)
    print("done", out_dir, flush=True)


if __name__ == "__main__":
    main()
