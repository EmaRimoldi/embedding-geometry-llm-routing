#!/usr/bin/env python
"""exp76 — a practical upper bound: a reasoning model given a retrieved lookup table.

For each held-out query (split 0 of the standard family-stratified 70/30 splits) the model
receives its k nearest training rows in the emitted GTE space with per-model reward/cost,
the family/topic-cell aggregate table, the price list and lambda, and returns a JSON
decision that says whether it decided by lookup or by reasoning, plus the choice the table
alone implies.  The deterministic kNN-k and cell-lookup choices are logged as machine
counterfactuals.  --mode lookup_only forbids reasoning.  --lofo restricts the table to the
other families (300 stratified test rows per held-out family).  --smoke builds five prompts,
prints one and writes them; it makes NO network call unless --live is also given.
Backends: openai_compatible (vLLM, OpenAI) and anthropic, both through urllib; API keys are
read from OPENAI_API_KEY / ANTHROPIC_API_KEY, never from files.  Resumable: successful rows
already in decisions.jsonl are skipped by (query_id, lambda); rows that recorded an error
are retried on the next run rather than treated as done; a truncated last line (from a
killed job) is skipped with a warning instead of raising; resuming a tag whose settings.json
records a different backend/model/mode/lambda/k/max_tokens/lofo is refused unless --force,
so one decisions.jsonl can never mix two models.  Every row records the serving model and
backend.  --force restarts the tag from scratch, deleting decisions.jsonl and
smoke_prompts.jsonl first.  --concurrency N runs N queries in flight (one thread each,
appending under a lock); N=1 reproduces the sequential behaviour.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from rb.data import load_panel                                          # noqa: E402
from rb.panelctx import PANEL, RESULTS_ROOT, topic_str                  # noqa: E402
from rb.probes.sweep import split                                       # noqa: E402
from rb.reps import find_rep, load_rep                                  # noqa: E402
from rb import oracle_prompt as O                                       # noqa: E402
from rb.pathological import unit_rows                                   # noqa: E402

EXP = "exp76_lookup_reasoning"
# Client/request errors: retrying cannot help and burns the budget/the operator's time.
NO_RETRY_STATUS = (400, 401, 403, 404, 422)
SETTINGS_KEYS = ("backend", "model", "mode", "lambda", "k", "max_tokens", "lofo")


class Backend:
    def __init__(self, kind, model, base_url, max_tokens, timeout=120):
        self.kind, self.model, self.base_url, self.max_tokens, self.timeout = kind, model, base_url.rstrip("/"), max_tokens, timeout
        env = "ANTHROPIC_API_KEY" if kind == "anthropic" else "OPENAI_API_KEY"
        self.key = os.environ.get(env, "EMPTY")
    def complete(self, system, user):
        if self.kind == "anthropic":
            url = f"{self.base_url}/v1/messages"; hdr = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "system": system, "messages": [{"role": "user", "content": user}]}
        else:
            url = f"{self.base_url}/v1/chat/completions"; hdr = {"Authorization": f"Bearer {self.key}", "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr, method="POST")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read())
                if self.kind == "anthropic":
                    return "".join(b.get("text", "") for b in j["content"]), {"input_tokens": j["usage"]["input_tokens"], "output_tokens": j["usage"]["output_tokens"]}
                return j["choices"][0]["message"]["content"], {"input_tokens": j["usage"]["prompt_tokens"], "output_tokens": j["usage"]["completion_tokens"]}
            except urllib.error.HTTPError as e:
                # 400/401/403/404/422 are bad request, bad key, forbidden, wrong URL or a
                # rejected body: identical on every retry, so fail now and say why.
                if e.code in NO_RETRY_STATUS:
                    try:
                        detail = e.read().decode("utf-8", "replace")[:500]
                    except Exception:                                # noqa: BLE001 — body already consumed/closed
                        detail = "<no response body>"
                    raise RuntimeError(f"HTTP {e.code} {e.reason} from {url} (not retried): {detail}") from None
                if attempt == 4: raise                               # 429 / 5xx: retried with backoff
                time.sleep(2 ** attempt)
            except Exception:                                        # noqa: BLE001 — URLError/timeout/parse: retried, then re-raised
                if attempt == 4: raise
                time.sleep(2 ** attempt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["openai_compatible", "anthropic"], default="openai_compatible")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct"); ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["reasoning", "lookup_only"], default="reasoning"); ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--k", type=int, default=20); ap.add_argument("--max-tokens", type=int, default=800); ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--lofo", action="store_true"); ap.add_argument("--lofo-per-family", type=int, default=300)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--live", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1, help="queries in flight; one thread each, rows appended under a lock")
    a = ap.parse_args(); rng = np.random.default_rng(76)
    if a.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    panel = load_panel(PANEL); fam = panel.dataset_index(); cells = topic_str(panel); models = [str(m) for m in panel.models]
    r = np.nan_to_num(panel.rewards, nan=0.0).astype(float); c = np.nan_to_num(panel.costs, nan=0.0, posinf=0.0).astype(float)
    avail = np.asarray(panel.available, dtype=bool)
    x = unit_rows(load_rep(find_rep(PANEL, "computed/gte-qwen2-7b@prompt"), panel.n, panel.keep))
    prices = {m: float(np.median(c[:, j])) for j, m in enumerate(models)}
    tr, te = split(panel.n, fam, 0); tr = np.sort(tr); te = np.sort(te)
    if a.lofo:
        te = np.sort(np.concatenate([rng.choice(te[fam[te] == f], size=min(a.lofo_per_family, (fam[te] == f).sum()), replace=False) for f in np.unique(fam)]))
    elif a.subsample:
        te = np.sort(rng.choice(te, size=min(a.subsample, len(te)), replace=False))
    if a.smoke:
        te = te[:5]
    tag = f"{a.mode}_lam{a.lam:g}" + ("_lofo" if a.lofo else "") + ("_smoke" if a.smoke else "")
    out = RESULTS_ROOT / "oracle_reasoning" / EXP / tag; out.mkdir(parents=True, exist_ok=True)
    dec_path = out / "decisions.jsonl"; smoke_path = out / "smoke_prompts.jsonl"; done = set()
    if a.force:
        dec_path.unlink(missing_ok=True); smoke_path.unlink(missing_ok=True)
    if not a.live:
        smoke_path.unlink(missing_ok=True)              # any dry run rebuilds the prompt log from scratch
    if dec_path.exists():
        for lineno, line in enumerate(open(dec_path), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[exp76] warning: skipping unparseable line {lineno} in {dec_path} (partial write?)", flush=True)
                continue
            if "error" not in row:                                          # errored rows are retried, not skipped
                done.add((row["query_id"], row["lambda"]))
    settings = {**panel.provenance(), "experiment": EXP, "backend": a.backend, "model": a.model, "mode": a.mode, "lambda": a.lam, "k": a.k,
                "max_tokens": a.max_tokens, "lofo": a.lofo, "n_test": int(len(te)), "split_seed": 0, "representation": "computed/gte-qwen2-7b@prompt", "prices": prices}
    # Resuming a tag with different settings would silently mix two runs in one decisions.jsonl.
    prev_path = out / "settings.json"
    if prev_path.exists() and not a.force:
        try:
            prev = json.loads(prev_path.read_text())
        except json.JSONDecodeError:
            prev = {}
        drift = {k: (prev[k], settings[k]) for k in SETTINGS_KEYS if k in prev and prev[k] != settings[k]}
        if drift:
            raise SystemExit("[exp76] refusing to resume " + str(out) + ": settings differ from the recorded run:\n"
                             + "\n".join(f"  {k}: recorded {was!r} -> requested {now!r}" for k, (was, now) in drift.items())
                             + "\n  pass --force to delete decisions.jsonl and restart this tag, or use a different tag.")
    prev_path.write_text(json.dumps(settings, indent=2))
    be = Backend(a.backend, a.model, a.base_url, a.max_tokens) if a.live else None
    meta = {"model": a.model, "backend": a.backend}
    pending = [int(i) for i in te if (str(panel.query_ids[i]), a.lam) not in done]
    print(f"[exp76] {tag}: {len(pending)} of {len(te)} queries pending (concurrency {a.concurrency if a.live else 1})", flush=True)

    # The retrieval pool, its embedding/outcome slices and the cell table depend only on the
    # held-out family (None outside --lofo), so they are built once per family instead of once
    # per query.  Built lazily: outside --lofo this is a single entry.
    ctx: dict = {}; ctx_lock = threading.Lock()

    def context_for(i: int):
        key = int(fam[i]) if a.lofo else None
        with ctx_lock:
            if key not in ctx:
                pool = tr[fam[tr] != fam[i]] if a.lofo else tr
                ctx[key] = (pool, x[pool], r[pool], c[pool], O.cell_table(r[pool], c[pool], cells[pool], models, a.lam))
            return ctx[key]

    def build(i: int):
        """Prompt, its sha and the machine counterfactuals for one query (no network)."""
        pool, xp, rp, cp, cu = context_for(i)
        nb_pos = O.neighbour_rows(x[i], xp, a.k); nb = pool[nb_pos]
        cell = str(cells[i]) if str(cells[i]) in cu else str(panel.datasets[fam[i]])
        cell_util = cu.get(cell, {m: float("nan") for m in models})
        nbs = [{"text": str(panel.prompts[j]), "rewards": r[j].tolist(), "costs": c[j].tolist(), "cell": str(cells[j])} for j in nb]
        user = O.format_prompt(str(panel.prompts[i]), nbs, cell, cell_util, models, prices, a.lam, mode=a.mode)
        cf = O.machine_counterfactuals(rp, cp, nb_pos, cell_util, models, a.lam, available=avail[i])
        return user, hashlib.sha256(user.encode()).hexdigest()[:16], cf

    def work(i: int):
        """One live query, in a worker thread: build, call, parse.  Never raises."""
        qid = str(panel.query_ids[i])
        try:
            user, sha, cf = build(i)
            txt, usage = be.complete(O.SYSTEM_PROMPT, user)
            dec = O.parse_decision(txt, models)
        except Exception as e:                                                # noqa: BLE001
            return {"query_id": qid, "lambda": a.lam, "mode": a.mode, **meta, "error": str(e)[:300]}, {}
        return O.record(qid, a.lam, a.mode, sha, dec, cf, {"reward": r[i].tolist(), "cost": c[i].tolist()}, txt, usage, meta=meta), usage

    n_err = 0; usage_tot = {"input_tokens": 0, "output_tokens": 0}
    if not a.live:
        with open(smoke_path, "a") as sf:
            for n_, i in enumerate(tqdm(pending, desc=tag)):
                user, sha, cf = build(i)
                if a.smoke and n_ == 0:
                    print(f"[exp76] first user prompt length: {len(user)} chars (~{len(user) // 4} tokens)", flush=True)
                    print(O.SYSTEM_PROMPT + "\n\n" + user[:3000], flush=True)
                sf.write(json.dumps({"query_id": str(panel.query_ids[i]), "prompt_sha256": sha, "chars": len(user), "counterfactuals": cf}) + "\n")
    else:
        write_lock = threading.Lock()
        with open(dec_path, "a") as fh, ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futures = [ex.submit(work, i) for i in pending]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=tag):
                row, usage = fut.result()
                with write_lock:
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    if "error" in row:
                        n_err += 1
                    else:
                        for k_ in usage_tot: usage_tot[k_] += usage.get(k_, 0)
    print(f"[exp76] {tag}: wrote {out}; errors {n_err}; tokens {usage_tot}", flush=True)


if __name__ == "__main__":
    main()
