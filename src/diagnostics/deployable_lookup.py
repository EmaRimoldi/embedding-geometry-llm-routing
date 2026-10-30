"""Train-only predicted-label lookup on the frozen GTE IID splits."""
import argparse, csv, json, os, sys, time, warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/baselines'))
from common import load, split_path, destination, sha

OUT = ROOT / 'results/diagnostics/deployable_lookup'
PANELS = ['routerbench-0shot', 'sprout-sixfam']

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def run(task):
    panel, fold = PANELS[task // 5], 'iid_' + str(task % 5)
    out = OUT / panel / fold
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    start = time.monotonic()
    d = load(panel, 'gte')
    split = json.loads(split_path(panel).read_text())[fold]
    tr, te = np.asarray(split['train']), np.asarray(split['test'])
    assert not np.intersect1d(tr, te).size
    cfg = json.loads((ROOT / 'configs/baselines.json').read_text())
    scaler = StandardScaler().fit(d['x'][tr])
    xtr = scaler.transform(d['x'][tr]).astype(np.float64)
    xte = scaler.transform(d['x'][te]).astype(np.float64)
    known = json.loads((destination('lookups', panel, fold, 'gte') / 'result.json').read_text())
    arrays = dict(train_rows=tr, test_rows=te, query_ids=d['query_ids'][te])
    report = dict(panel=panel, fold=fold, n_train=len(tr), n_test=len(te),
                  source_sha256=sha(Path(__file__)), split_sha256=sha(split_path(panel)),
                  panel_sha256=sha(ROOT/'datasets'/('routerbench' if task < 5 else 'sprout')/'prepared/panel.npz'),
                  classifier={}, scores=[], warnings=[], slurm_job=os.environ.get('SLURM_JOB_ID'))
    estimators = {}
    for kind, y in [('family', d['fam']), ('cell', d['subgroup'])]:
        clf = LogisticRegression(C=1., solver='lbfgs', max_iter=3000, tol=1e-4,
                                 class_weight=None, random_state=42)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            clf.fit(xtr, y[tr])
        report['warnings'] += [str(w.message) for w in ws]
        if any(issubclass(w.category, ConvergenceWarning) for w in ws):
            write(out/'failed.json', report)
            raise RuntimeError('Classifier did not converge; preserve warning and ask before changing protocol')
        pred = clf.predict(xte)
        arrays[kind + '_predicted_labels'] = pred
        arrays[kind + '_true_labels'] = y[te]
        estimators[kind] = clf
        groups = clf.classes_
        means_r = np.stack([d['r'][tr[y[tr] == g]].mean(0) for g in groups])
        means_c = np.stack([d['c'][tr[y[tr] == g]].mean(0) for g in groups])
        pred_idx = np.searchsorted(groups, pred)
        report['classifier'][kind] = dict(accuracy=float((pred == y[te]).mean()),
                    classes=len(groups), n_iter=clf.n_iter_.tolist(),
                    unseen_test_label_fraction=float((~np.isin(y[te], groups)).mean()))
        for lam in cfg['lambdas'][panel]:
            utility = d['r'] - lam*d['c']
            best_global = utility[tr].mean(0).argmax()
            best_family = {f: utility[tr[d['fam'][tr] == f]].mean(0).argmax() for f in np.unique(d['fam'][tr])}
            best_group = {g: utility[tr[y[tr] == g]].mean(0).argmax() for g in groups}
            true_choice = np.array([best_group.get(g, best_family.get(f, best_global)) for g, f in zip(y[te], d['fam'][te])])
            # Predict group first, then use the same training group utility means.
            predicted_choice = np.asarray([best_group[g] for g in pred])
            # Equivalent separate reward/cost reconstruction must choose the same score up to rounding.
            predicted_u = (means_r - lam*means_c)[pred_idx]
            assert np.allclose(predicted_u[np.arange(len(te)), predicted_choice], predicted_u.max(1), atol=1e-10)
            actual = utility[te]
            u_pred = actual[np.arange(len(te)), predicted_choice]
            u_true = actual[np.arange(len(te)), true_choice]
            reference = known['results'][str(lam)]['family_lookup' if kind == 'family' else 'family_topic_lookup']['utility']
            assert abs(u_true.mean() - reference) < 1e-10
            arrays[f'{kind}|{lam:g}|predicted_choice'] = predicted_choice
            arrays[f'{kind}|{lam:g}|true_choice'] = true_choice
            report['scores'].append(dict(kind=kind, lambda_value=lam,
                predicted_utility=float(u_pred.mean()), true_utility=float(u_true.mean()),
                difference=float((u_pred-u_true).mean()), choice_disagreement=float((predicted_choice!=true_choice).mean())))
    arrays['scaler_mean'] = scaler.mean_
    arrays['scaler_scale'] = scaler.scale_
    np.savez_compressed(out/'predictions.npz', **arrays)
    joblib.dump(dict(scaler=scaler, classifiers=estimators), out/'classifiers.joblib', compress=3)
    # Reopen saved predictions and reconstruct all reported utilities.
    saved = np.load(out/'predictions.npz', allow_pickle=False)
    for row in report['scores']:
        key = f"{row['kind']}|{row['lambda_value']:g}"
        utility = d['r'][te] - row['lambda_value']*d['c'][te]
        for prefix in ['predicted','true']:
            got = utility[np.arange(len(te)), saved[key+'|'+prefix+'_choice']].mean()
            assert abs(got-row[prefix+'_utility']) < 1e-10
    report['elapsed_seconds'] = time.monotonic()-start
    report['audit'] = dict(prediction_reconstruction=True, known_lookup_reproduction=True,
                           train_test_disjoint=True, train_only_scaling_and_classifier=True)
    write(out/'result.json', report)
    write(out/'complete.json', dict(result_sha256=sha(out/'result.json'), prediction_sha256=sha(out/'predictions.npz')))
    print('COMPLETE', panel, fold, report['classifier'], flush=True)

def summarize():
    results = []
    for panel in PANELS:
        for i in range(5):
            path = OUT/panel/f'iid_{i}'
            assert (path/'complete.json').exists(), str(path)
            check = json.loads((path/'complete.json').read_text())
            assert sha(path/'result.json') == check['result_sha256']
            assert sha(path/'predictions.npz') == check['prediction_sha256']
            results.append(json.loads((path/'result.json').read_text()))
    rows=[]; accuracy=[]
    for panel in PANELS:
        rs=[r for r in results if r['panel']==panel]
        for kind in ['family','cell']:
            accuracy.append(dict(panel=panel,kind=kind,accuracy=float(np.mean([r['classifier'][kind]['accuracy'] for r in rs]))))
            for lam in sorted({s['lambda_value'] for r in rs for s in r['scores']}):
                ss=[s for r in rs for s in r['scores'] if s['kind']==kind and s['lambda_value']==lam]
                assert len(ss)==5
                rows.append(dict(panel=panel,kind=kind,lambda_value=lam,**{k:float(np.mean([s[k] for s in ss])) for k in ['predicted_utility','true_utility','difference','choice_disagreement']}))
    write(OUT/'report.json', dict(complete=True,folds=10,accuracy=accuracy,scores=rows,
                                all_prediction_audits_passed=True,source_hashes=sorted({r['source_sha256'] for r in results})))
    with (OUT/'scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(accuracy),flush=True)

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',type=int);ap.add_argument('--report',action='store_true');a=ap.parse_args()
    summarize() if a.report else run(a.task)
