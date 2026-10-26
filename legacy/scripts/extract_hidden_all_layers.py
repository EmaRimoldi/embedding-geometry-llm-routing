"""Frozen-reader layer sweep (GPU job): hidden states of a frozen model at EVERY layer on a panel's
prompts, several poolings, saved as float16 memmaps under
features/<panel>/prompt/hidden/<model>/L<layer>_<pool>.npy (+ meta.json).  No outcome is used.

Models (all already on the cluster, read in place; nothing downloaded):
  decoders  qwen2.5-7b-instruct, qwen2-7b-base, qwen3-8b, qwen2.5-coder-1.5b-instruct
  contrastive decoders (their own intermediate layers)  gte-qwen2-7b (custom bidirectional code, as
            sentence-transformers runs it), qwen3-embedding-0.6b
  contrastive encoders  e5-large-v2, minilm-l12, bge-m3
Poolings: decoders -> last token, mean over tokens, mean of the last 8 tokens; encoders -> CLS, mean.
Text = the routed prompt, left-truncated at the model's window (8,192; 512 for the BERT encoders),
no chat template.  For qwen2.5-7b-instruct the vocabulary projection at the last token is also
stored (logit_lens_last.npy: entropy + top-64 sorted log-probs), the KEEN-style feature.
Sanity check: for the contrastive models the last layer's own pooling is compared with the stored
embedding of the same model (cosine), printed at the end.
Usage: extract_hidden_all_layers.py --model qwen3-8b --panel routerbench-0shot"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, RUNTIME_ROOT, load_panel  # noqa: E402

HF = RUNTIME_ROOT / "huggingface-cache/hub"; HOME_HF = Path.home() / ".cache/huggingface/hub"
def snap(root: Path, name: str) -> str:
    return sorted(glob.glob(str(root / name / "snapshots" / "*")))[-1]
MODELS = {
    "qwen2.5-7b-instruct": dict(path=str(RUNTIME_ROOT / "artifacts/models/Qwen2.5-7B-Instruct__a09a3545"), kind="decoder", max_len=8192, lm_head=True),
    # qwen2-7b-base: the cached snapshot has no weight shards (incomplete download) -> not usable without downloading; dropped.
    # the venv's tokenizers library cannot parse the Qwen3 tokenizer.json; Qwen3 shares Qwen2.5's vocabulary and merges
    # (verified at submission: identical token ids on the panel prompts), so the Qwen2.5 tokenizer is used for the Qwen3 models.
    "qwen3-8b": dict(path=snap(HF, "models--Qwen--Qwen3-8B"), kind="decoder", max_len=8192, tokenizer=str(RUNTIME_ROOT / "artifacts/models/Qwen2.5-7B-Instruct__a09a3545")),
    "qwen2.5-coder-1.5b-instruct": dict(path=snap(HOME_HF, "models--Qwen--Qwen2.5-Coder-1.5B-Instruct"), kind="decoder", max_len=8192),
    "qwen3-embedding-0.6b": dict(path=snap(HF, "models--Qwen--Qwen3-Embedding-0.6B"), kind="decoder", max_len=8192, append_eos=True, ref=None, tokenizer=str(RUNTIME_ROOT / "artifacts/models/Qwen2.5-7B-Instruct__a09a3545")),
    "gte-qwen2-7b": dict(path=str(RUNTIME_ROOT / "artifacts/models/gte-Qwen2-7B-instruct"), kind="decoder", max_len=8192, trust=True, append_eos=True, ref="gte-qwen2-7b", ref_pool="last"),
    "e5-large-v2": dict(path=snap(HF, "models--intfloat--e5-large-v2"), kind="encoder", max_len=512, prefix="query: "),
    "minilm-l12": dict(path=str(RUNTIME_ROOT / "artifacts/models/all-MiniLM-L12-v2__3392ab41"), kind="encoder", max_len=512, ref="minilm-l12", ref_pool="mean"),
    "bge-m3": dict(path=str(RUNTIME_ROOT / "artifacts/models/bge-m3"), kind="encoder", max_len=8192, ref_pool="cls"),
}
TOPK = 64


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS)); ap.add_argument("--panel", required=True); a = ap.parse_args()
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    cfg = MODELS[a.model]; kind = cfg["kind"]; trust = cfg.get("trust", False)
    if trust:   # GTE's custom modeling file imports flash_attn unconditionally; the project's stub (scripts/embed_gpu.py) makes it optional
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from embed_gpu import _register_optional_flash_attention_stub; _register_optional_flash_attention_stub()
    panel = load_panel(a.panel); texts = [cfg.get("prefix", "") + str(t) for t in panel.text("prompt")]; n = len(texts)
    out_dir = ANALYSIS_ROOT / "features" / a.panel / "prompt" / "hidden" / a.model; out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["path"]), trust_remote_code=trust); tok.truncation_side = "left"
    if kind == "decoder":
        tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
        cls_ = AutoModelForCausalLM if cfg.get("lm_head") else AutoModel
        model = cls_.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=trust, **({} if trust else {"attn_implementation": "sdpa"}))
        pools = ("last", "mean", "last8")
    else:
        tok.padding_side = "right"; model = AutoModel.from_pretrained(cfg["path"], torch_dtype=torch.float32, device_map="cuda"); pools = ("cls", "mean")
    model.eval(); H = model.config.hidden_size; NL = model.config.num_hidden_layers + 1; max_len = cfg["max_len"]
    eos = tok.eos_token_id if cfg.get("append_eos") else None
    def encode(batch):
        enc = tok(batch, padding=False, truncation=True, max_length=max_len - (1 if eos is not None else 0), add_special_tokens=(kind == "encoder"))
        ids = [x + ([eos] if eos is not None and (not x or x[-1] != eos) else []) for x in enc["input_ids"]]
        T = max(len(x) for x in ids); pad = tok.pad_token_id
        if kind == "decoder":
            inp = torch.tensor([[pad] * (T - len(x)) + x for x in ids]); am = torch.tensor([[0] * (T - len(x)) + [1] * len(x) for x in ids])
        else:
            inp = torch.tensor([x + [pad] * (T - len(x)) for x in ids]); am = torch.tensor([[1] * len(x) + [0] * (T - len(x)) for x in ids])
        return inp.cuda(), am.cuda()
    lengths = np.array([min(len(tok(t, add_special_tokens=(kind == "encoder"))["input_ids"]), max_len) for t in texts]); order = np.argsort(lengths)
    mm = {(L, p): np.lib.format.open_memmap(out_dir / f"L{L}_{p}.npy", mode="w+", dtype=np.float16, shape=(n, H)) for L in range(NL) for p in pools}
    lens = None
    if cfg.get("lm_head"):
        lens = np.lib.format.open_memmap(out_dir / "logit_lens_last.npy", mode="w+", dtype=np.float32, shape=(n, TOPK + 1))
    big = H >= 3584; t0 = time.time(); s = 0; done = 0; nb = 0
    while s < n:
        L = int(lengths[order[min(s + 31, n - 1)]])
        bs = (32 if L <= 256 else 16 if L <= 1024 else 4 if L <= 4096 else 1) if big else (64 if L <= 256 else 32 if L <= 1024 else 8 if L <= 4096 else 2)
        idx = [int(i) for i in order[s:s + bs]]; s += bs
        inp, am = encode([texts[i] for i in idx])
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=am, output_hidden_states=True)
        hs = out.hidden_states; assert len(hs) == NL, (len(hs), NL)
        m = am.unsqueeze(-1).to(torch.float32); cnt = m.sum(1).clamp(min=1)
        if kind == "decoder":
            m8 = m.clone(); m8[:, :-8, :] = 0; cnt8 = m8.sum(1).clamp(min=1)
        for Lr in range(NL):
            h = hs[Lr].to(torch.float32)
            if kind == "decoder":
                got = {"last": h[:, -1, :], "mean": (h * m).sum(1) / cnt, "last8": (h * m8).sum(1) / cnt8}
            else:
                got = {"cls": h[:, 0, :], "mean": (h * m).sum(1) / cnt}
            for p, v in got.items():
                mm[(Lr, p)][idx] = v.cpu().numpy().astype(np.float16)
        if lens is not None:
            lp = torch.log_softmax(out.logits[:, -1, :].to(torch.float32), dim=-1); ent = -(lp.exp() * lp).sum(-1, keepdim=True); top = lp.topk(TOPK, dim=-1).values
            lens[idx] = torch.cat([ent, top], dim=1).cpu().numpy()
        done += len(idx); nb += 1
        if nb % 50 == 0 or done == n:
            print(f"{done}/{n} ({time.time() - t0:.0f}s, batch {bs}, len {L})", flush=True)
        del out, hs
    for arr in mm.values():
        arr.flush()
    if lens is not None:
        lens.flush()
    check = None
    if cfg.get("ref"):
        ref = np.load(ANALYSIS_ROOT / "features" / a.panel / "prompt" / f"{cfg['ref']}.npy", mmap_mode="r")
        ref = ref[panel.keep] if panel.keep is not None and ref.shape[0] == len(panel.keep) else ref
        mine = np.asarray(mm[(NL - 1, cfg["ref_pool"])][:2000], dtype=np.float32); r = np.asarray(ref[:2000], dtype=np.float32)
        cos = (mine * r).sum(1) / (np.linalg.norm(mine, axis=1) * np.linalg.norm(r, axis=1) + 1e-9); check = {"rows": 2000, "cosine_mean": float(cos.mean()), "cosine_min": float(cos.min())}
        print("sanity vs stored embedding:", check, flush=True)
    json.dump({**panel.provenance(), "model": a.model, "path": cfg["path"], "kind": kind, "layers": NL, "hidden_size": H, "poolings": list(pools), "max_len": max_len,
               "prefix": cfg.get("prefix", ""), "append_eos": bool(eos is not None), "texts_truncated": int((np.array([len(tok(t, add_special_tokens=(kind == 'encoder'))['input_ids']) for t in texts]) > max_len).sum()),
               "logit_lens": bool(lens is not None), "sanity_vs_stored": check, "seconds": time.time() - t0}, open(out_dir / "meta.json", "w"), indent=1)
    print("done", out_dir, flush=True)


if __name__ == "__main__":
    main()
