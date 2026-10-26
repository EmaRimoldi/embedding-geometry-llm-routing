"""Behavioural probe (GPU): let a frozen small instruct model START answering every query and record
what it does. No outcome is used. Representation = the model's trajectory, not the text.

For each panel row: chat-templated prompt (user = the routed prompt text, left-truncated to 4,096
tokens), greedy decoding of at most MAX_NEW tokens with Qwen2.5-7B-Instruct (the reader model already
used for hidden states; nothing is downloaded). Saved per row:
  draft_len      tokens generated before EOS (== MAX_NEW if not finished)   -> verbosity proxy
  finished       EOS reached within MAX_NEW
  first_entropy  entropy of the next-token distribution at the first generated position
  first_margin   top-1 minus top-2 log-prob at the first generated position
  mean_logprob   mean log-prob of the generated tokens
  mean_entropy   mean next-token entropy over generated positions
  draft          the generated text (jsonl)
Outputs: results/drafts/<panel>/stats.npz, drafts.jsonl, meta.json. Usage: gen_drafts.py --panel <name>
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; ISO = HERE.parent; E4 = ISO.parent
sys.path.insert(0, str(E4 / "src"))
from rb.data import RUNTIME_ROOT, load_panel  # noqa: E402

MODEL = RUNTIME_ROOT / "artifacts/models/Qwen2.5-7B-Instruct__a09a3545"
MAX_NEW = 64; MAX_IN = 4096; BATCH = 48


def main():
    global MAX_NEW
    ap = argparse.ArgumentParser(); ap.add_argument("--panel", required=True); ap.add_argument("--field", default="prompt"); ap.add_argument("--max_new", type=int, default=MAX_NEW); ap.add_argument("--out", default="drafts"); a = ap.parse_args()
    MAX_NEW = a.max_new
    from transformers import AutoModelForCausalLM, AutoTokenizer
    panel = load_panel(a.panel); texts = [str(t) for t in panel.text(a.field)]; n = len(texts)
    out = ISO / "results" / a.out / a.panel; out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(MODEL)); tok.padding_side = "left"; tok.truncation_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"); model.eval()
    eos_ids = set([tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else []))
    ntok = np.array([len(tok(t, truncation=True, max_length=MAX_IN)["input_ids"]) for t in texts]); order = np.argsort(ntok)
    TOKEN_BUDGET = 12000   # batch x prompt tokens; prefill logits are batch x tokens x vocab in float32
    L = np.zeros(n, np.int32); fin = np.zeros(n, bool); fe = np.zeros(n, np.float32); fm = np.zeros(n, np.float32); mlp = np.zeros(n, np.float32); men = np.zeros(n, np.float32)
    t0 = time.time()
    with open(out / "drafts.jsonl", "w") as fh, torch.inference_mode():
        b = 0
        while b < n:
            # length-adaptive batch: keep tokens x vocab logits under budget (long prompts -> small batches)
            longest = int(ntok[order[min(b + BATCH, n) - 1]]) + 40
            bs = int(max(1, min(BATCH, TOKEN_BUDGET // max(longest, 1)))); idx = order[b:b + bs]; b += bs
            chats = [tok.apply_chat_template([{"role": "user", "content": texts[i]}], tokenize=False, add_generation_prompt=True) for i in idx]
            enc = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=MAX_IN).to("cuda")
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
            gen = g.sequences[:, enc["input_ids"].shape[1]:]
            logps = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)
            ent = -(logps.exp() * logps).sum(-1); top2 = logps.topk(2, -1).values; margin = top2[..., 0] - top2[..., 1]
            tokp = logps.gather(-1, gen.unsqueeze(-1)).squeeze(-1)
            for k, i in enumerate(idx):
                seq = gen[k].tolist(); T = len(seq)
                for t, tid in enumerate(seq):
                    if tid in eos_ids: T = t; break
                L[i] = T; fin[i] = T < len(seq); fe[i] = ent[k, 0].item(); fm[i] = margin[k, 0].item()
                mlp[i] = tokp[k, :max(T, 1)].mean().item(); men[i] = ent[k, :max(T, 1)].mean().item()
                fh.write(json.dumps({"row": int(i), "draft_len": int(T), "finished": bool(fin[i]), "draft": tok.decode(seq[:T], skip_special_tokens=True)}) + "\n")
            if (b // BATCH) % 20 == 0: print(f"{b}/{n} ({time.time() - t0:.0f}s) bs={bs}", flush=True)
    np.savez(out / "stats.npz", draft_len=L, finished=fin, first_entropy=fe, first_margin=fm, mean_logprob=mlp, mean_entropy=men)
    json.dump({"panel": a.panel, "rows": n, "model": str(MODEL), "max_new": MAX_NEW, "max_in": MAX_IN, "greedy": True, "chat_template": True, "seconds": time.time() - t0}, open(out / "meta.json", "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
