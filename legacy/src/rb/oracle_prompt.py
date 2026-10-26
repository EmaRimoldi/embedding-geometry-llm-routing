"""Prompt assembly, parsing and counterfactual logic for the lookup-plus-reasoning oracle
(exp76).  Pure functions; the network client lives in the script."""
from __future__ import annotations
import json
import numpy as np

SYSTEM_PROMPT = (
    "You choose which language model should answer a query, given a lookup table of similar past "
    "queries with each model's recorded reward (1 = solved) and cost in USD, an aggregate table for the "
    "query's category, and a cost penalty lambda.  Utility = reward - lambda * cost.  Use the lookup table "
    "when it settles the choice; use your own reasoning about the query only when you judge the table "
    "insufficient.  Answer with one JSON object and nothing else: "
    '{"decision_source": "lookup" or "reasoning", "chosen_model": <model name>, '
    '"lookup_choice": <model name the table alone implies>, "rationale": <one sentence>}.'
)


def cell_table(rewards_tr, costs_tr, cells_tr, models, lam):
    u = np.asarray(rewards_tr, float) - lam * np.asarray(costs_tr, float); out = {}
    for cell in np.unique(cells_tr):
        sel = cells_tr == cell; out[str(cell)] = {m: float(u[sel, j].mean()) for j, m in enumerate(models)}
    return out


def neighbour_rows(x_unit_query, x_unit_train, k):
    s = x_unit_train @ x_unit_query
    k = min(k, len(s)); ind = np.argpartition(-s, k - 1)[:k]
    return ind[np.argsort(-s[ind])]


def format_prompt(query_text, neighbours, cell, cell_util, models, prices, lam, max_chars=300, mode="reasoning"):
    lines = [f"Cost penalty lambda = {lam:g}.  Models and mean price per call (USD): " + ", ".join(f"{m}: {prices[m]:.5f}" for m in models), "",
             f"Aggregate table for category '{cell}' (mean utility per model on past queries):",
             "  " + ", ".join(f"{m}: {cell_util.get(m, float('nan')):+.3f}" for m in models), "",
             f"{len(neighbours)} most similar past queries (text truncated to {max_chars} characters); per model: reward/cost:"]
    for n_, nb in enumerate(neighbours, 1):
        cells_ = ", ".join(f"{m} {int(round(r))}/{c:.5f}" for m, r, c in zip(models, nb["rewards"], nb["costs"]))
        lines.append(f"[{n_}] ({nb['cell']}) {nb['text'][:max_chars]!s}\n     {cells_}")
    lines += ["", "New query:", query_text[:4 * max_chars], ""]
    lines.append('Return one JSON object: {"decision_source": "lookup" or "reasoning", "chosen_model": <model name>, '
                  '"lookup_choice": <model name the table alone implies>, "rationale": <one sentence>}.')
    if mode == "lookup_only":
        lines.append('Use ONLY the tables above; set "decision_source" to "lookup".')
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _balanced_json_candidates(text):
    """Yield each top-level ``{...}`` substring of ``text``.  For every ``{`` walk forward
    counting brace depth, ignoring braces inside double-quoted strings (honouring ``\\"``
    escapes) so a rationale like ``"the set {1,2} is small"`` does not fool the scanner
    into closing the object early; yield the substring once depth returns to zero."""
    n = len(text); i = 0
    while i < n:
        if text[i] != "{":
            i += 1; continue
        depth = 0; in_str = False; esc = False; j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[i:j + 1]
                        break
            j += 1
        i = j + 1


def _validate_decision(cand, models):
    src = cand.get("decision_source")
    if src not in ("lookup", "reasoning"):
        raise ValueError(f"bad decision_source {src!r}")
    def idx(v):
        if v is None: return None
        # ``type(v) is int``, not isinstance: bool is a subclass of int, so a model that
        # answers {"chosen_model": true} would otherwise silently resolve to model 1.
        if type(v) is int and 0 <= v < len(models): return v
        if v in models: return models.index(v)
        raise ValueError(f"unknown model {v!r}")
    chosen = idx(cand.get("chosen_model"))
    if chosen is None:
        raise ValueError("chosen_model missing")
    return {"decision_source": src, "chosen_model": chosen, "lookup_choice": idx(cand.get("lookup_choice")), "rationale": str(cand.get("rationale", ""))[:500]}


def parse_decision(text, models):
    """Scan ``text`` for balanced ``{...}`` substrings (brace-depth, quote-aware) and
    return the decision dict for the first one that both parses as JSON and validates
    against ``models``; raise ``ValueError`` if none do."""
    for chunk in _balanced_json_candidates(text):
        try:
            cand = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(cand, dict):
            continue
        try:
            return _validate_decision(cand, models)
        except ValueError:
            continue
    raise ValueError("no JSON object in model output")


def machine_counterfactuals(rewards_tr, costs_tr, nb, cell_util, models, lam, available=None):
    """Deterministic kNN-``len(nb)`` and cell-lookup choices for one query.

    ``available`` (optional M-length bool) is that query's availability row: models the panel
    never ran on it get -inf and are never chosen, so the logged counterfactual utility is a
    measurement rather than a fill value.
    """
    u = np.asarray(rewards_tr, float)[nb] - lam * np.asarray(costs_tr, float)[nb]
    mean = u.mean(0); cost = np.asarray(costs_tr, float)[nb].mean(0)
    cellv = np.array([cell_util.get(m, -np.inf) for m in models], dtype=float)
    if available is not None:
        av = np.asarray(available, dtype=bool)
        mean = np.where(av, mean, -np.inf); cellv = np.where(av, cellv, -np.inf)
    best = mean.max(); tie = np.isclose(mean, best)
    knn = int(np.where(tie, cost, np.inf).argmin())
    cell = int(np.argmax(cellv))
    return {"knn_choice": knn, "cell_choice": cell}


def record(query_id, lam, mode, prompt_sha, decision, counterfactuals, outcome, raw_text, usage, meta=None):
    """One decisions.jsonl row.

    ``meta`` is an optional dict of run provenance merged into the row (exp76 passes the
    serving ``model`` and ``backend``, so a decisions.jsonl assembled over several resumes
    still says which model produced each row).  It must not reuse the row's own field names.
    """
    r = np.asarray(outcome["reward"], float); c = np.asarray(outcome["cost"], float); j = decision["chosen_model"]
    row = {"query_id": str(query_id), "lambda": float(lam), "mode": mode, "prompt_sha256": prompt_sha,
           **(meta or {}), **decision, **counterfactuals,
           "reward_chosen": float(r[j]), "cost_chosen": float(c[j]), "utility_chosen": float(r[j] - lam * c[j]),
           "utility_knn": float(r[counterfactuals["knn_choice"]] - lam * c[counterfactuals["knn_choice"]]),
           "utility_cell": float(r[counterfactuals["cell_choice"]] - lam * c[counterfactuals["cell_choice"]]),
           "utility_lookup_choice": (None if decision["lookup_choice"] is None else float(r[decision["lookup_choice"]] - lam * c[decision["lookup_choice"]])),
           "utility_oracle": float((r - lam * c).max()), "raw_text": raw_text[:2000], "usage": usage}
    return row
