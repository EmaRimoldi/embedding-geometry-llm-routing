"""Prefill hidden states of Qwen2.5-7B-Instruct for every SPROUT prompt (all 44,241 release rows, in rows.jsonl order),
replicating the existing extraction (raw prompt text, no prefix, no EOS appended, truncation at 8,192 tokens, last-token
and mean pooling). Layers saved: 12, 21, 28 (of 29 hidden states). Output: <this folder>/features/sprout/hidden/
qwen2.5-7b-instruct/L{k}_{last,mean}.npy as float16 plus meta.json. Reads the model from the shared artifacts path used
by the existing features; writes only inside this folder."""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; E4 = ROOT.parent
MODEL = Path("__POOL__/Qwen2.5-7B-Instruct"); LAYERS = (12, 21, 28); MAXLEN = 8192; TOKEN_BUDGET = 24000
OUT = ROOT / "features" / "sprout" / "hidden" / "qwen2.5-7b-instruct"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time(); rows = [json.loads(l) for l in open(E4 / "sprout" / "panel" / "rows.jsonl")]; texts = [r["prompt"] for r in rows]; n = len(texts)
    tok = AutoTokenizer.from_pretrained(str(MODEL)); tok.padding_side = "right"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, device_map="cuda").eval(); H = model.config.hidden_size
    enc = [tok(t, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"] for t in texts]; lens = np.array([len(e) for e in enc]); order = np.argsort(lens); trunc = int((lens >= MAXLEN).sum())
    out = {(L, p): np.zeros((n, H), dtype=np.float16) for L in LAYERS for p in ("last", "mean")}; pos = 0; done = 0
    while pos < n:
        L = lens[order[pos]]; bs = max(1, min(64, TOKEN_BUDGET // max(int(lens[order[min(pos + 63, n - 1)]]), 1))); idx = order[pos:pos + bs]; mx = int(lens[idx].max())
        ids = torch.full((len(idx), mx), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(idx), mx), dtype=torch.long)
        for j, i in enumerate(idx): e = enc[i]; ids[j, :len(e)] = torch.tensor(e); att[j, :len(e)] = 1
        with torch.no_grad(): hs = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True, use_cache=False).hidden_states
        for Lk in LAYERS:
            h = hs[Lk].float(); m = att.cuda().unsqueeze(-1).float(); last = h[torch.arange(len(idx)), (att.sum(1) - 1).cuda()]; mean = (h * m).sum(1) / m.sum(1)
            out[(Lk, "last")][idx] = last.cpu().numpy().astype(np.float16); out[(Lk, "mean")][idx] = mean.cpu().numpy().astype(np.float16)
        pos += len(idx); done += len(idx)
        if done % 2000 < len(idx): print(done, "/", n, "max len", mx, round(time.time() - t0), flush=True)
    for (Lk, p), a in out.items(): np.save(OUT / f"L{Lk}_{p}.npy", a)
    json.dump({"panel": "sprout (all 44,241 release rows, rows.jsonl order)", "model": "qwen2.5-7b-instruct", "path": str(MODEL), "layers_saved": list(LAYERS), "hidden_size": H, "poolings": ["last", "mean"], "max_len": MAXLEN, "prefix": "", "append_eos": False, "texts_truncated": trunc, "dtype": "float16", "seconds": time.time() - t0}, open(OUT / "meta.json", "w"), indent=1); print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
