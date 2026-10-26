"""Evaluate paired routing decisions only after inference; reuse campaign metrics."""
import argparse
import ast
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import canonical, choose, digest, parse_prediction, sha_file, write_json


def read_jsonl(path):
    with open(path) as f:
        return list(map(json.loads, f))


def paired_stats(delta, changed, bootstrap=2000, permutations=10000, seed=20260923):
    delta, changed = np.asarray(delta), np.asarray(changed, bool)
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot = []
    for start in range(0, bootstrap, 100):
        idx = rng.integers(n, size=(min(100, bootstrap-start), n))
        boot.extend(delta[idx].mean(axis=1))
    lo, hi = np.quantile(boot, [.025, .975])
    observed = float(delta.mean())
    if n <= 16:
        means = np.asarray([np.mean(delta*np.asarray(s)) for s in itertools.product([-1, 1], repeat=n)])
        p = float(np.mean(np.abs(means) >= abs(observed)-1e-14))
        test = 'exact paired sign randomization'
    else:
        hits = 0
        for start in range(0, permutations, 100):
            signs = rng.choice([-1, 1], size=(min(100, permutations-start), n))
            hits += int(np.sum(np.abs((signs*delta).mean(axis=1)) >= abs(observed)-1e-14))
        p = (hits+1)/(permutations+1)
        test = 'Monte Carlo paired sign randomization'
    pos, neg = delta > 1e-12, delta < -1e-12
    return dict(n=n, delta_utility=observed, ci95_low=float(lo), ci95_high=float(hi),
                randomization_p=p, randomization_test=test, changed_fraction=float(changed.mean()),
                improved_fraction=float(pos.mean()), harmed_fraction=float(neg.mean()),
                improved_given_change=float(pos[changed].mean()) if changed.any() else None,
                harmed_given_change=float(neg[changed].mean()) if changed.any() else None,
                mean_gain_improved=float(delta[pos].mean()) if pos.any() else None,
                mean_loss_harmed=float(delta[neg].mean()) if neg.any() else None,
                net_gain=observed)


def existing_frontier_functions(path):
    """Load the actual frozen repository definitions without running its figure builder."""
    tree = ast.parse(Path(path).read_text())
    wanted = {'nondominated', 'ndch', 'aiq'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {f.name for f in functions} != wanted:
        raise ValueError('Frozen frontier implementation no longer has the expected functions')
    from collections.abc import Iterable
    ns = {'np': np, 'pd': pd, 'Iterable': Iterable}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    return ns['ndch'], ns['aiq']


def campaign_baselines(data, manifest, ids, lams):
    """Only final-test baselines from the identical frozen split may be imported."""
    if manifest['stage'] != 'test':
        return {}, 'Saved campaign test predictions deliberately unavailable for source-validation pilot'
    output = {}
    for kind in ['knn', 'prior']:
        folder = data / 'baselines' / kind
        meta = json.loads((folder / 'result.json').read_text())
        if meta['split_sha256'] != manifest['source_split_sha256']:
            raise ValueError('Baseline split mismatch')
        z = np.load(folder / 'predictions.npz', allow_pickle=False)
        if not np.array_equal(z['query_ids'], manifest['outer_test_ids']):
            raise ValueError('Baseline parent split query ID mismatch')
        allowed = set(manifest.get('excluded_test_alias_ids', []))
        if set(z['query_ids']) - set(ids) != allowed:
            raise ValueError('Unexpected baseline evaluation exclusions')
        positions = {qid: i for i, qid in enumerate(z['query_ids'])}
        take = np.asarray([positions[qid] for qid in ids])
        names = ['knn100', 'knn_validated'] if kind == 'knn' else ['soft_prior']
        for name in names:
            output['campaign_'+name] = {lam: z[f'{lam:g}|{name}'][take] for lam in lams}
    output['campaign_mlp_utility_validated'] = {}
    for lam in lams:
        folder = data / 'baselines/mlp_gpu' / f'lambda_{lam:g}'
        meta = json.loads((folder / 'result.json').read_text())
        if meta['split_sha256'] != manifest['source_split_sha256']:
            raise ValueError('MLP split mismatch')
        z = np.load(folder / 'predictions.npz', allow_pickle=False)
        if not np.array_equal(z['query_ids'], manifest['outer_test_ids']):
            raise ValueError('MLP parent split query ID mismatch')
        positions = {qid: i for i, qid in enumerate(z['query_ids'])}
        take = np.asarray([positions[qid] for qid in ids])
        output['campaign_mlp_utility_validated'][lam] = z['choice'][take]
    return output, ('Exact saved iid_0 GTE predictions with the approved test-alias mask; '
                    'MLP is the documented utility-regression adaptation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--inference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data', type=Path, help='Optional exported-data location when moving artifacts')
    ap.add_argument('--full-prepared', type=Path, help='Frozen full-test packets for usage forecasting')
    args = ap.parse_args()
    prep, run, out = args.prepared.resolve(), args.inference.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    cfg = json.loads((prep / 'config.json').read_text())
    manifest = json.loads((prep / 'split_manifest.json').read_text())
    runmeta = json.loads((run / 'run.json').read_text())
    if runmeta['status'] != 'complete' or runmeta['prepared_manifest_sha256'] != sha_file(prep / 'split_manifest.json'):
        raise ValueError('Incomplete inference or wrong prepared dataset')
    data = args.data.resolve() if args.data else Path(manifest['data_root'])
    evidence = read_jsonl(prep / 'evidence.jsonl')
    preds = read_jsonl(run / 'predictions.jsonl')
    requests = read_jsonl(run / 'request_metadata.jsonl')
    ids = np.asarray(manifest['evaluation_ids'])
    models, lams = manifest['models'], cfg['lambdas']
    n, nm = len(ids), len(models)
    table = {}
    for p in preds:
        key = (p['sample_id'], p['effort'])
        if key in table:
            raise ValueError('Duplicate prediction')
        parse_prediction(canonical(p['prediction']), key[0], models)
        table[key] = p
    if set(table) != {(sid, effort) for sid in ids for effort in cfg['efforts']}:
        raise ValueError('Incomplete paired prediction grid')
    for sid, e in zip(ids, evidence):
        if e['packet']['sample_id'] != sid:
            raise ValueError('Evidence order mismatch')
        if any(table[(sid, effort)]['evidence_hash'] != e['evidence_hash'] for effort in cfg['efforts']):
            raise ValueError('Paired evidence mismatch')
        successful = [r for r in requests if r['sample_id'] == sid and r.get('validation_error') is None]
        if len(successful) != 2:
            raise ValueError('Missing paired request proof')
        normalized = []
        for r in successful:
            if r['requested_model'] != cfg['model'] or r['resolved_model'] != cfg['model'] or r['tool_events'] != 0:
                raise ValueError('Model/tool invariant failed')
            req = json.loads(json.dumps(r['request']))
            if req['thread']['config'].pop('model_reasoning_effort') != r['effort'] or req['turn'].pop('effort') != r['effort']:
                raise ValueError('Wrong effort configuration')
            normalized.append(canonical(req))
        if normalized[0] != normalized[1]:
            raise ValueError('Paired Codex requests differ beyond effort')
    # Test outcomes become accessible here, after the frozen predictions have passed validation.
    z = np.load(prep / 'evaluation_only.npz', allow_pickle=False)
    assert np.array_equal(z['query_ids'], ids)
    r, c, fam, topic = z['r'], z['c'], z['fam'], z['topic']
    estimates = {f'gpt56_lookup_{effort}': np.asarray([[p['expected_performance'] for p in table[(sid, effort)]['prediction']['model_estimates']] for sid in ids]) for effort in cfg['efforts']}
    local_cost = np.asarray([[a['training_cost_estimate'] for a in e['packet']['lookup_aggregates']] for e in evidence])
    estimates['weighted_knn_same_evidence'] = np.asarray([[a['weighted_mean_performance'] for a in e['packet']['lookup_aggregates']] for e in evidence])
    estimates['knn_same_evidence'] = np.asarray([[a['unweighted_mean_performance'] for a in e['packet']['lookup_aggregates']] for e in evidence])
    choices = {name: {lam: choose(p, local_cost, lam) for lam in lams} for name, p in estimates.items()}
    choices['best_single_source'] = {lam: np.full(n, choose(z['source_mean_r'], z['source_mean_c'], lam), int) for lam in lams}
    choices['hindsight_oracle'] = {lam: choose(r, c, lam) for lam in lams}
    external, baseline_note = campaign_baselines(data, manifest, ids, lams)
    choices.update(external)
    spec = importlib.util.spec_from_file_location('campaign_common_eval', data / 'campaign_code/common.py')
    campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
    groups = {'overall': np.arange(n)}
    for g in np.unique(fam):
        groups['dataset:'+manifest['families'][g]] = np.flatnonzero(fam == g)
        for t in np.unique(topic[fam == g]):
            if t:
                groups['task:'+manifest['families'][g]+'/'+t] = np.flatnonzero((fam == g) & (topic == t))
    for attr in ['lookup_uncertainty', 'mean_similarity', 'neighbor_disagreement']:
        v = np.asarray([e[attr] for e in evidence]); median = np.median(v)
        for label, mask in [('low', v <= median), ('high', v > median)]:
            if mask.any():
                groups[f'conditional:{attr}:{label}'] = np.flatnonzero(mask)
    aggregate, detail, paired, frequencies = [], [], [], []
    ii = np.arange(n)
    for lam in lams:
        oracle = choices['hindsight_oracle'][lam]
        oracle_u = (r-lam*c)[ii, oracle]
        knn = choices['weighted_knn_same_evidence'][lam]
        for name, grid in choices.items():
            ch = np.asarray(grid[lam]); u = (r-lam*c)[ii, ch]
            for group, ix in groups.items():
                score = campaign.score(ch[ix], r[ix], c[ix], fam[ix], lam)
                aggregate.append(dict(method=name, group=group, lambda_value=lam, n=len(ix),
                                      performance=score['reward'], cost=score['cost'], utility=score['utility'],
                                      equal_family_utility=score['equal_family'], regret=float((oracle_u[ix]-u[ix]).mean()),
                                      oracle_agreement=float((ch[ix]==oracle[ix]).mean()),
                                      weighted_knn_agreement=float((ch[ix]==knn[ix]).mean())))
                for mi, model in enumerate(models):
                    frequencies.append(dict(method=name, group=group, lambda_value=lam, model=model,
                                            count=int(np.sum(ch[ix]==mi)), fraction=float(np.mean(ch[ix]==mi))))
            for i in ii:
                detail.append(dict(sample_id=ids[i], dataset=manifest['families'][fam[i]], task=topic[i],
                                   method=name, lambda_value=lam, selected_model=models[ch[i]],
                                   performance=float(r[i,ch[i]]), realized_cost=float(c[i,ch[i]]),
                                   utility=float(u[i]), regret=float(oracle_u[i]-u[i])))
        low, high = choices['gpt56_lookup_low'][lam], choices['gpt56_lookup_xhigh'][lam]
        delta = (r-lam*c)[ii,high] - (r-lam*c)[ii,low]
        for group, ix in groups.items():
            paired.append(dict(group=group, lambda_value=lam, **paired_stats(delta[ix], low[ix]!=high[ix],
                cfg['bootstrap_replicates'], cfg['permutation_replicates'], cfg['statistics_seed'])))
    out.mkdir(parents=True)
    pd.DataFrame(aggregate).to_csv(out/'aggregate_results.csv', index=False)
    pd.DataFrame(detail).to_csv(out/'evaluated_predictions.csv', index=False)
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame(frequencies).to_csv(out/'selection_frequencies.csv', index=False)
    ndch, aiq = existing_frontier_functions(data/'campaign_code/frontier_metrics.py')
    curves_out = []
    for group, ix in groups.items():
        if group.startswith('conditional:'):
            continue
        curves = {}
        for name, grid in choices.items():
            points = [(float(c[ix,grid[lam][ix]].mean()), float(r[ix,grid[lam][ix]].mean())) for lam in lams]
            curves[name] = ndch([p[0] for p in points], [p[1] for p in points])
        # Zero Router = hull of every fixed candidate, not source-best static alone.
        curves['zero_router'] = ndch(c[ix].mean(axis=0), r[ix].mean(axis=0))
        scores = aiq(curves)
        lower = max(float(x.min()) for x,y in curves.values()); upper = min(float(x.max()) for x,y in curves.values())
        curves_out.append({'group':group, 'aiq_common_cost_lower':lower,'aiq_common_cost_upper':upper,
                           'aiq':{k:float(v) if np.isfinite(v) else None for k,v in scores.items()},
                           'aiq_note':'null when curves have no positive-width common cost support',
                           'curves':{k:{'cost':x.tolist(),'performance':y.tolist()} for k,(x,y) in curves.items()}})
    write_json(out/'frontiers_aiq.json',curves_out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    for name, points in curves_out[0]['curves'].items():
        ax.plot(points['cost'],points['performance'],'.-',label=name)
    ax.set(xlabel='Realized candidate cost (USD/query)',ylabel='Recorded performance',title=f'RouterBench {manifest["stage"]}: non-decreasing convex hulls')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'cost_performance.png',dpi=180);plt.close(fig)
    usage = {}
    grouped_usage = []
    for group, ix in groups.items():
        group_ids = set(ids[ix])
        for effort in cfg['efforts']:
            rows = [a for a in requests if a['effort'] == effort and a['sample_id'] in group_ids]
            totals = {key:sum(a.get(key, 0) or 0 for a in rows) for key in
                      ['input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens', 'latency_seconds']}
            grouped_usage.append(dict(group=group, effort=effort, queries=len(ix), calls=len(rows),
                                      retries=sum(a['attempt'] > 0 for a in rows), **totals,
                                      mean_latency_seconds=totals['latency_seconds']/len(rows),
                                      estimated_api_cost_usd=None,
                                      note='One inference reused for every lambda; reasoning is included in output tokens'))
    pd.DataFrame(grouped_usage).to_csv(out/'usage_by_group.csv', index=False)
    full_dry = json.loads((args.full_prepared / 'dry_run_report.json').read_text()) if args.full_prepared else None
    forecast_n = full_dry['evaluation_queries'] if full_dry else len(manifest['outer_test_ids']) - len(manifest.get('excluded_test_alias_ids', []))
    for effort in cfg['efforts']:
        rows = [a for a in requests if a['effort']==effort]
        ok = [a for a in rows if a.get('validation_error') is None]
        sums = {key:sum(a.get(key,0) or 0 for a in rows) for key in ['input_tokens','cached_input_tokens','output_tokens','reasoning_tokens','total_tokens','latency_seconds']}
        forecast = {k:v/len(ok)*forecast_n for k,v in sums.items()}
        if full_dry:
            overhead = np.mean([a['input_tokens'] - a['prompt_tokens_estimate'] for a in ok])
            forecast['input_tokens'] = full_dry['paired_prompt_tokens_total_estimate']/2 + float(overhead)*forecast_n
            forecast['total_tokens'] = forecast['input_tokens'] + forecast['output_tokens']
        usage[effort] = {'calls':len(rows),'successful_calls':len(ok),'retries':sum(a['attempt']>0 for a in rows),
                         **sums, 'full_test_forecast':forecast,
                         'estimated_api_cost_usd':None}
    report = {'stage':manifest['stage'],'queries':n,'model':cfg['model'],'efforts':cfg['efforts'],
              'paired_request_identity_passed':True,'usage':usage,'baseline_note':baseline_note,
              'full_forecast_queries':forecast_n,
              'forecast_note':'Full prompt text counts plus mean pilot Codex overhead; outputs and serial latency extrapolated from seven queries. Cache reuse and quota availability are not guaranteed.',
              'inference_limitations':['Resolved client model ID recorded; immutable server snapshot and Responses response_id not exposed',
                'Codex output limit and separate reasoning.mode not configurable through this protocol; identical model defaults used',
                'No seed exposed; one draw per condition; differences include sampling variation',
                'Reasoning content neither requested nor used as evidence',
                'Codex subscription quota is not an API bill; monetary cost unavailable without verified tariff'],
              'statistics_note': ('Pilot estimates are functionality diagnostics on 7 validation queries, not evidence of superiority. ' if manifest['stage']=='pilot' else '') + 'Query bootstrap and randomization tests are paired; no multiplicity adjustment across lambdas, tasks or conditional slices. Pilot-based full-run forecasts are rough.'}
    write_json(out/'summary.json',report)
    lines=['# RouterBench Codex reasoning comparison', '', f'Queries: {n}; stage: {manifest["stage"]}; paired request identity: PASS.', '',
           '| λ | low utility | xhigh utility | difference | changed |', '|---:|---:|---:|---:|---:|']
    for lam in lams:
        a=next(x for x in aggregate if x['group']=='overall' and x['method']=='gpt56_lookup_low' and x['lambda_value']==lam)
        b=next(x for x in aggregate if x['group']=='overall' and x['method']=='gpt56_lookup_xhigh' and x['lambda_value']==lam)
        p=next(x for x in paired if x['group']=='overall' and x['lambda_value']==lam)
        lines.append(f'| {lam} | {a["utility"]:.6f} | {b["utility"]:.6f} | {p["delta_utility"]:+.6f} | {p["changed_fraction"]:.1%} |')
    if manifest['stage'] == 'pilot':
        lines += ['', '**This small validation pilot is not a confirmatory routing result.**']
    lines += ['', baseline_note,
              '', 'Full-run consumption estimates and limitations: `summary.json`. All inference calls use the Codex login; no API dollar price has been assumed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
