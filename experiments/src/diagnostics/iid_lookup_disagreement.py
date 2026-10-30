"""Compare frozen GTE kNN-100 decisions with native-label lookup decisions."""
from pathlib import Path
import sys, json, csv, os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/baselines'))
from common import load, split_path, destination, sha

OUT = ROOT / 'results/diagnostics/iid_lookup_disagreement'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'report.json').exists(), 'Preserve existing completed report'
    config = ROOT/'configs/baselines.json'
    cfg = json.loads(config.read_text())
    provenance = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                  str(config.relative_to(ROOT)): sha(config)}
    all_rows, summaries, by_split, arrays = [], [], [], {}
    for panel in ('routerbench-0shot', 'sprout-sixfam'):
        d = load(panel)
        splits = json.loads(split_path(panel).read_text())
        lams = cfg['lambdas'][panel]
        for p in [split_path(panel), ROOT/'datasets'/('routerbench' if panel.startswith('router') else 'sprout')/'prepared/panel.npz']:
            provenance[str(p.relative_to(ROOT))] = sha(p)
        fold_data = []
        for i in range(5):
            split = 'iid_'+str(i)
            tr, te = (np.asarray(splits[split][k]) for k in ('train','test'))
            assert not np.intersect1d(tr, te).size
            saved, reports = {}, {}
            for kind in ('knn','lookups'):
                root = destination(kind,panel,split,'gte')
                p = root/'predictions.npz'
                saved[kind] = dict(np.load(p,allow_pickle=False))
                reports[kind] = json.loads((root/'result.json').read_text())
                assert np.array_equal(saved[kind]['test_rows'],te)
                assert np.array_equal(saved[kind]['query_ids'],d['query_ids'][te])
                for f in (p,root/'result.json'):
                    provenance[str(f.relative_to(ROOT))] = sha(f)
            utility, choices = {}, {}
            for method, kind in [('knn100','knn'),('family_lookup','lookups'),('family_topic_lookup','lookups')]:
                ch = np.column_stack([saved[kind][f'{lam:g}|{method}'] for lam in lams])
                assert np.all((ch>=0)&(ch<d['r'].shape[1]))
                u = np.column_stack([(d['r'][te]-lam*d['c'][te])[np.arange(len(te)),ch[:,j]] for j,lam in enumerate(lams)])
                assert np.isfinite(u).all()
                for j,lam in enumerate(lams):
                    assert abs(u[:,j].mean()-reports[kind]['results'][str(lam)][method]['utility'])<1e-10
                choices[method],utility[method]=ch,u
            # Verify the lookup rule directly, independent of its saved choices.
            for j,lam in enumerate(lams):
                train_u=d['r'][tr]-lam*d['c'][tr]
                global_choice=train_u.mean(0).argmax()
                fam={g:train_u[d['fam'][tr]==g].mean(0).argmax() for g in np.unique(d['fam'][tr])}
                cell={g:train_u[d['subgroup'][tr]==g].mean(0).argmax() for g in np.unique(d['subgroup'][tr])}
                a=np.array([fam.get(g,global_choice) for g in d['fam'][te]])
                b=np.array([cell.get(g,a[k]) for k,g in enumerate(d['subgroup'][te])])
                assert np.array_equal(a,choices['family_lookup'][:,j])
                assert np.array_equal(b,choices['family_topic_lookup'][:,j])
                kp=saved['knn']
                assert np.array_equal((kp['knn100|r']-lam*kp['knn100|c']).argmax(1),choices['knn100'][:,j])
            fold_data.append((split,te,choices,utility))
        unique=np.unique(np.concatenate([f[1] for f in fold_data]))
        for ref in ('family_lookup','family_topic_lookup'):
            sums=np.zeros((len(unique),2*len(lams)))
            weights=np.zeros(len(unique))
            for split,te,ch,u in fold_data:
                delta=u['knn100']-u[ref]
                disagree=ch['knn100']!=ch[ref]
                assert np.all(delta[~disagree]==0)
                ix=np.searchsorted(unique,te)
                w=1/(5*len(te))
                np.add.at(sums,ix,np.column_stack([delta,disagree])*w)
                np.add.at(weights,ix,w)
                arrays[f'{panel}|{ref}|{split}|rows']=te
                arrays[f'{panel}|{ref}|{split}|delta']=delta
                arrays[f'{panel}|{ref}|{split}|disagree']=disagree
                for j,lam in enumerate(lams):
                    by_split.append(dict(panel=panel,split=split,reference=ref,lambda_value=lam,
                        n_test=len(te),n_disagree=int(disagree[:,j].sum()),
                        knn_utility=float(u['knn100'][:,j].mean()),lookup_utility=float(u[ref][:,j].mean()),
                        delta=float(delta[:,j].mean())))
            assert abs(weights.sum()-1)<1e-10
            actual=sums.sum(0)
            # One draw per unique query; repeated test appearances and all penalties stay together.
            rng=np.random.default_rng(42)
            boot=np.empty((1000,2*len(lams)))
            for b in range(1000):
                counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
                boot[b]=counts@sums/(counts@weights)
            for j,lam in enumerate(lams):
                gain,dis=float(actual[j]),float(actual[len(lams)+j])
                ci=np.quantile(boot[:,j],[.025,.975]).tolist()
                row=dict(panel=panel,reference=ref,lambda_value=lam,disagreement_fraction=dis,
                         conditional_gain=gain/dis if dis else None,overall_gain=gain,ci95=ci)
                all_rows.append(row)
            mean_gain=float(actual[:len(lams)].mean())
            mean_dis=float(actual[len(lams):].mean())
            summaries.append(dict(panel=panel,reference=ref,mean_overall_gain=mean_gain,
                 mean_disagreement_fraction=mean_dis,conditional_gain=mean_gain/mean_dis if mean_dis else None,
                 mean_gain_ci95=np.quantile(boot[:,:len(lams)].mean(1),[.025,.975]).tolist(),
                 unique_test_queries=len(unique),test_appearances=sum(len(f[1]) for f in fold_data)))
    np.savez_compressed(OUT/'paired_metrics.npz',**arrays)
    report=dict(complete=True,source_hashes=provenance,summaries=summaries,per_penalty=all_rows,by_split=by_split,
        protocol='Frozen GTE kNN-100 and family/native-cell lookups, five original IID splits per benchmark, all thirteen prescribed penalties. Equal query weights within each split, then equal split weights; mean summary gives equal penalty weights.',
        uncertainty='1000 paired bootstrap resamples of unique test query identities, retaining all repeated appearances and all penalties together. Weighted ratios preserve the equal-split estimator. Pointwise percentile 95% intervals conditional on existing fitted routers and splits; not retraining uncertainty or simultaneous intervals.',
        interpretation='Differences on disagreements establish useful query-specific decisions relative to lookup, not an isolated causal effect of within-cell proximity.',
        audit=dict(prediction_scores_reconstructed=True,row_ids_match=True,train_test_disjoint=True,
                   lookup_choices_recomputed=True,knn_choices_recomputed=True,finite_utilities=True,
                   zero_gain_on_agreements=True),slurm_job=os.environ.get('SLURM_JOB_ID'))
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in [('per_penalty',all_rows),('by_split',by_split),('summary',summaries)]:
        with (OUT/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
