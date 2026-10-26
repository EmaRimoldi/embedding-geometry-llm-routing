# RouterBench transfer: outcome-free descriptions and NumPy 1.26 topic adapter.
"""GraphRouter (Feng et al., ICLR 2025) re-implemented from the LLMRouterBench vendored code
(baselines/GraphRouter/model/{graph_nn,multi_task_graph_router}.py, repo c77cb050; the
released-code architecture, which GraphRouter issue #9 notes differs from the paper's text)
on the experiments4 protocol.

Graph: one node per query (features: GTE embedding of the prompt and of its task description)
and one per LLM (GTE embedding of the released description text); a directed edge query→LLM
for every pair with edge features [scenario utility, cost min-max normalised within the task,
raw score].  FeatureAlign maps task/query/LLM features to 8/8/16 dims (node width 16); two
GeneralConv layers (torch_geometric, edge features through a 3→3 ReLU MLP) with BatchNorm;
edge prediction = sigmoid(mean(x_ini[query] ⊙ x[llm])); BCE against the label "this LLM has
the maximum scenario utility for the query" (ties all 1).  Scenarios: Performance First
(u = s), Balance (0.5 s − 0.5 c̃), Cost First (0.2 s − 0.8 c̃).  Training: AdamW lr 3e-4,
wd 1e-4; per epoch 32 random maskings hiding half of the training edges (the hidden ones are
predicted from the visible ones), mean loss, one step; up to 2,000 epochs.  Routing: test
edges predicted with only training edges visible; argmax over LLMs.

Deviations, on purpose: (1) the released trainer keeps the epoch with the best TEST utility
(leakage); here 10% of the training queries form an inner validation fold (their edges hidden
from the message passing and from the loss), the epoch with the best validation utility is
kept (patience 200); (2) cost normalisation uses the training rows' min/max per task, not the
whole panel's.  The three scenario routers are cost-aware at fixed weights, so each is evaluated
at every λ with its λ-independent choice; no λ-specific retraining is attempted.
Environment: venvs/env (torch_geometric 2.8). Usage: python graphrouter.py [--epochs 2000]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
if not hasattr(np, "trapezoid"):          # numpy 1.26 in the torch_geometric environment
    np.trapezoid = np.trapz
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv

E4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E4 / "llmrouterbench" / "baselines"))
from common import LAMBDAS, SEEDS, OUT_ROOT, aiq, bootstrap_diff, cell_lookup_choice, evaluate_choices, load_all, split, write_method  # noqa: E402
import common as baseline_common
from rb.panelctx import _rb_record_prefix, _HS

def routerbench_topic_str(panel):
    prefixes = _rb_record_prefix(panel)
    return np.array([prefixes[i] if str(fam) == "mmlu" else
        ("hellaswag/wikihow" if _HS.search(str(panel.prompts[i])) else "hellaswag/activitynet") if str(fam) == "hellaswag" else str(fam)
        for i, fam in enumerate(panel.dataset_ids)], dtype=object)

baseline_common.topic_str = routerbench_topic_str


SCENARIOS = {"performance_first": (1.0, 0.0), "balance": (0.5, 0.5), "cost_first": (0.2, 0.8)}
HIDDEN, EDGE_DIM, LR, WD, MASK_RATE, INNER_BATCH = 8, 3, 3e-4, 1e-4, 0.5, 32


class FeatureAlign(nn.Module):
    def __init__(self, qdim, ldim, common):
        super().__init__(); self.query_transform = nn.Linear(qdim, common); self.llm_transform = nn.Linear(ldim, common * 2); self.task_transform = nn.Linear(ldim, common)

    def forward(self, task, query, llm):
        return torch.cat([torch.cat([self.task_transform(task), self.query_transform(query)], 1), self.llm_transform(llm)], 0)


class EncoderDecoderNet(nn.Module):
    def __init__(self, qdim, ldim, hidden, in_edges):
        super().__init__(); self.in_edges = in_edges; self.model_align = FeatureAlign(qdim, ldim, hidden)
        self.encoder_conv_1 = GeneralConv(hidden * 2, hidden * 2, in_edge_channels=in_edges); self.encoder_conv_2 = GeneralConv(hidden * 2, hidden * 2, in_edge_channels=in_edges)
        self.edge_mlp = nn.Linear(in_edges, in_edges); self.bn1 = nn.BatchNorm1d(hidden * 2); self.bn2 = nn.BatchNorm1d(hidden * 2)

    def forward(self, task, query, llm, edge_index, edge_mask, edge_can_see, edge_weight):
        ei_see, ei_pred = edge_index[:, edge_can_see], edge_index[:, edge_mask]
        ew = F.relu(self.edge_mlp(edge_weight[edge_can_see]))
        x_ini = self.model_align(task, query, llm)
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, ei_see, edge_attr=ew)))
        x = self.bn2(self.encoder_conv_2(x, ei_see, edge_attr=ew))
        return torch.sigmoid((x_ini[ei_pred[0]] * x[ei_pred[1]]).mean(-1))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=2000); ap.add_argument("--patience", type=int, default=200); ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS)); ap.add_argument("--suffix", default="", help="appended to the method names (sensitivity runs)"); ap.add_argument("--no-early-stop", action="store_true", help="train the full epoch budget and keep the last epoch (released schedule without test selection)"); a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device", device, flush=True)
    panel, x, r, c, fam, cell = load_all(); F_ = len(panel.datasets); M = panel.m; n = panel.n
    D = np.load(E4 / "results" / "routerbench_baselines_exp64" / "graphrouter_inputs" / "descriptions.npz")
    assert list(D["models"]) == [str(m) for m in panel.models] and list(D["tasks"]) == list(panel.datasets)
    q_feat = torch.as_tensor(x, device=device); task_feat = torch.as_tensor(D["task_embedding"][fam], device=device); llm_feat = torch.as_tensor(D["llm_embedding"], device=device)
    # edges: query i -> LLM j, edge id = i*M + j; LLM node ids n..n+M-1
    org = np.repeat(np.arange(n), M); des = n + np.tile(np.arange(M), n); edge_index = torch.as_tensor(np.stack([org, des]), device=device, dtype=torch.long)
    per = {f"graphrouter_{s}{a.suffix}": {} for s in a.scenarios}; diag = {}
    for seed in SEEDS:
        tr, te = split(n, fam, seed); tr, te = np.sort(tr), np.sort(te); fam_te = fam[te]
        rng = np.random.default_rng(3000 + seed); val = np.concatenate([rng.choice(tr[fam[tr] == g], size=max(1, int(0.1 * (fam[tr] == g).sum())), replace=False) for g in range(F_)]); fit = np.setdiff1d(tr, val)
        # cost normalised per task with the training rows' min/max
        cn = np.zeros_like(c)
        for g in range(F_):
            rows = np.flatnonzero(fam == g); lo, hi = c[tr][fam[tr] == g].min(), c[tr][fam[tr] == g].max()
            cn[rows] = (c[rows] - lo) / (hi - lo) if hi > lo else 0.0
        e_fit = torch.zeros(n * M, dtype=torch.bool, device=device); e_fit[torch.as_tensor(np.repeat(fit, M) * M + np.tile(np.arange(M), len(fit)), device=device)] = True
        e_val = torch.zeros(n * M, dtype=torch.bool, device=device); e_val[torch.as_tensor(np.repeat(val, M) * M + np.tile(np.arange(M), len(val)), device=device)] = True
        e_te = torch.zeros(n * M, dtype=torch.bool, device=device); e_te[torch.as_tensor(np.repeat(te, M) * M + np.tile(np.arange(M), len(te)), device=device)] = True
        for sc in a.scenarios:
            t0 = time.time(); ws, wc = SCENARIOS[sc]; util = ws * r - wc * cn
            label = np.isclose(util, util.max(1, keepdims=True), atol=1e-8).astype(np.float32).reshape(-1)
            ew = torch.as_tensor(np.stack([util.reshape(-1), cn.reshape(-1), r.reshape(-1)], 1), device=device, dtype=torch.float32); lab = torch.as_tensor(label, device=device)
            torch.manual_seed(seed); net = EncoderDecoderNet(x.shape[1], llm_feat.shape[1], HIDDEN, EDGE_DIM).to(device); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCELoss()
            best = (-np.inf, 0, None); bad = 0; hist = []
            for ep in range(1, a.epochs + 1):
                net.train(); opt.zero_grad(); loss_mean = 0.0
                for _ in range(INNER_BATCH):
                    hide = e_fit & (torch.rand(n * M, device=device) < MASK_RATE)          # hidden training edges = predicted
                    see = e_fit & ~hide
                    pred = net(task_feat, q_feat, llm_feat, edge_index, hide, see, ew); loss = crit(pred, lab[hide]); loss_mean = loss_mean + loss
                (loss_mean / INNER_BATCH).backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    pv = net(task_feat, q_feat, llm_feat, edge_index, e_val, e_fit, ew).view(len(val), M); ch = pv.argmax(1).cpu().numpy()
                    vu = float(util[val][np.arange(len(val)), ch].mean())
                hist.append(vu)
                if a.no_early_stop:
                    best = (vu, ep, None)
                elif vu > best[0]: best, bad = (vu, ep, {k: v.detach().clone() for k, v in net.state_dict().items()}), 0
                else:
                    bad += 1
                    if bad >= a.patience: break
            if best[2] is not None: net.load_state_dict(best[2])
            net.eval()
            with torch.no_grad():
                pt = net(task_feat, q_feat, llm_feat, edge_index, e_te, e_fit, ew).view(len(te), M); ch = pt.argmax(1).cpu().numpy()
            res, pq, curve = evaluate_choices({lam: ch for lam in LAMBDAS}, r, c, te, fam_te, F_)
            per[f"graphrouter_{sc}{a.suffix}"][seed] = {"results": res, "curve": curve.tolist(), "pq": pq}
            diag[f"{sc}/{seed}"] = {"best_epoch": best[1], "best_val_utility": best[0], "epochs_run": len(hist), "seconds": time.time() - t0, "test_reward": float(r[te][np.arange(len(te)), ch].mean()), "test_cost": float(c[te][np.arange(len(te)), ch].mean()),
                                    "choice_shares": {str(panel.models[j]): float((ch == j).mean()) for j in range(M) if (ch == j).any()}}
            if seed == SEEDS[0]:
                ref = {lam: (r - lam * c)[te][np.arange(len(te)), cell_lookup_choice(r - lam * c, tr, te, fam, cell)] for lam in (0.0, 10.0, 100.0, 1000.0)}
                per[f"graphrouter_{sc}{a.suffix}"][seed]["boot"] = {f"vs cell lookup @ lambda={lam:g}": bootstrap_diff(pq[lam], ref[lam], fam_te, F_) for lam in ref}
            print(f"seed {seed} {sc}: best epoch {best[1]} (val utility {best[0]:.4f}), test reward {diag[f'{sc}/{seed}']['test_reward']:.4f} cost {diag[f'{sc}/{seed}']['test_cost']:.5f}, {time.time() - t0:.0f}s", flush=True)
    max_cost = max(float(np.array(v["curve"])[:, 0].max()) for name in per for v in per[name].values())
    for name in per:
        for s in per[name]:
            per[name][s]["aiq"] = aiq(np.array(per[name][s]["curve"]), max_cost); per[name][s].pop("pq", None)
        sc = name[len("graphrouter_"):len(name) - len(a.suffix) if a.suffix else None]
        write_method(name, per[name], {"description": f"GraphRouter (released-code architecture), scenario {sc} weights {SCENARIOS[sc]}; " + (f"full {a.epochs}-epoch schedule, last epoch kept" if a.no_early_stop else f"epoch by inner validation (cap {a.epochs}, patience {a.patience})") + "; choice independent of λ.",
                                       "diagnostics": {k: v for k, v in diag.items() if k.startswith(sc)}}, panel)
    json.dump(diag, open(OUT_ROOT / f"graphrouter_diagnostics{a.suffix}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
