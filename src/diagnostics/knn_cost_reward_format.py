"""Frozen OOD kNN cost/reward diagnostic and answer-format restriction."""
from pathlib import Path
import argparse, json, os, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/baselines'))
from common import load, nearest, sha, split_path, destination, DATASETS
sys.path.insert(0, str(ROOT / 'src/diagnostics/appendix_ci'))
from ci_common import boot_sums, record
OUT = ROOT / 'results/diagnostics/knn_cost_reward_format'
CONFIG = ROOT / 'configs/knn_cost_reward_format.json'
PREDICTORS = ['gte', 'l21', 'gte_same_format', 'source_mean']
POLICIES = {
    'gte_reward_gte_cost': ('gte','gte'),
    'l21_reward_gte_cost': ('l21','gte'),
    'gte_reward_l21_cost': ('gte','l21'),
    'l21_reward_l21_cost': ('l21','l21'),
    'gte_same_format': ('gte_same_format','gte_same_format'),
    'source_mean': ('source_mean','source_mean'),
}
def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
def track(path, provenance):
    provenance[str(path.relative_to(ROOT))] = sha(path)
def verify(provenance):
    for path, digest in provenance.items():
        assert sha(ROOT/path) == digest, path
def base_inputs(panel):
    dataset = DATASETS[panel]
    return [CONFIG, Path(__file__), ROOT/'src/baselines/common.py',
        ROOT/'src/diagnostics/appendix_ci/ci_common.py',
        ROOT/'configs/appendix_ci.json', ROOT/'configs/baselines.json',
        split_path(panel), ROOT/f'datasets/{dataset}/prepared/panel.npz',
        ROOT/f'datasets/{dataset}/prepared/manifest.json',
        ROOT/f'datasets/{dataset}/prepared/queries.jsonl',
        ROOT/f'artifacts/embeddings/gte-qwen2-7b/{dataset}/manifest.json',
        ROOT/f'artifacts/embeddings/gte-qwen2-7b/{dataset}/vectors.npy']
def run(panel):
    start = time.time(); dataset = DATASETS[panel]; out = OUT/dataset
    assert not (out/'complete.json').exists()
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for path in base_inputs(panel): track(path, provenance)
    d = load(panel, 'gte')
    raw = dict(np.load(ROOT/f'datasets/{dataset}/prepared/panel.npz', allow_pickle=False))
    fmt = raw['format']; r, c, x = d['r'], d['c'], d['x']
    assert np.all(fmt != '') and len(np.unique(fmt)) == 2
    config = json.loads(CONFIG.read_text())
    lams = json.loads((ROOT/'configs/baselines.json').read_text())['lambdas'][panel]
    splits = json.loads(split_path(panel).read_text())
    fold_summaries = []; seen = []
    for split, s in splits.items():
        if not split.startswith('ood_'): continue
        t0 = time.time(); tr = np.asarray(s['train']); te = np.asarray(s['test'])
        assert not np.intersect1d(tr,te).size
        assert np.unique(d['fam'][te]).size == 1
        assert not np.intersect1d(d['fam'][tr],d['fam'][te]).size
        arrays = dict(test_rows=te, query_ids=d['query_ids'][te])
        for name, rep in [('gte','gte'),('l21','reader')]:
            base = destination('knn',panel,split,rep)
            for file in ['predictions.npz','result.json']: track(base/file,provenance)
            old = json.loads((base/'result.json').read_text())
            assert old['split_sha256'] == sha(split_path(panel))
            with np.load(base/'predictions.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['test_rows'],te)
                np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
                for target in ['r','c']: arrays[name+'|'+target] = z['knn100|'+target]
                for lam in lams:
                    ch = (arrays[name+'|r']-lam*arrays[name+'|c']).argmax(1)
                    np.testing.assert_array_equal(ch,z[f'{lam:g}|knn100'])
                    val = (r[te]-lam*c[te])[np.arange(len(te)),ch].mean()
                    np.testing.assert_allclose(val,old['results'][str(lam)]['knn100']['utility'],rtol=0,atol=1e-12)
        nn = np.empty((len(te),100),dtype=np.int32)
        distances = np.empty((len(te),100),dtype=np.float32)
        pools = {}
        for label in np.unique(fmt[te]):
            loc = np.flatnonzero(fmt[te] == label)
            pool = tr[fmt[tr] == label]
            assert len(pool) >= 100, (dataset,split,label,len(pool))
            ids, dist = nearest(x[te[loc]],x[pool],100)
            nn[loc] = pool[ids]; distances[loc] = dist
            pools[str(label)] = len(pool)
        assert np.isin(nn,tr).all() and (fmt[nn] == fmt[te,None]).all()
        assert all(len(np.unique(row))==100 for row in nn)
        arrays['gte_same_format_neighbors'] = nn
        arrays['gte_same_format_distance'] = distances
        for target, values in [('r',r),('c',c)]:
            pr = np.empty((len(te),values.shape[1]))
            for st in range(0,len(te),512):
                pr[st:st+512] = values[nn[st:st+512]].mean(1)
            arrays['gte_same_format|'+target] = pr
            arrays['source_mean|'+target] = np.broadcast_to(values[tr].mean(0),pr.shape).copy()
        for name,(rp,cp) in POLICIES.items():
            rr,cc=arrays[rp+'|r'],arrays[cp+'|c']
            choices=np.stack([(rr-lam*cc).argmax(1) for lam in lams],axis=1)
            arrays['choice|'+name]=choices.astype(np.int16)
        np.testing.assert_array_equal(arrays['choice|gte_reward_gte_cost'][:,0],arrays['choice|gte_reward_l21_cost'][:,0])
        np.testing.assert_array_equal(arrays['choice|l21_reward_gte_cost'][:,0],arrays['choice|l21_reward_l21_cost'][:,0])
        for name in PREDICTORS:
            assert np.isfinite(arrays[name+'|r']).all()
            assert np.isfinite(arrays[name+'|c']).all()
        path=out/(split+'.npz')
        assert not path.exists(), 'Preserve existing artifacts; resume only explicitly audited missing work'
        np.savez_compressed(path,**arrays)
        fold_summaries.append(dict(split=split,heldout_family=d['meta']['families'][int(d['fam'][te[0]])],
            n_test=len(te),n_train=len(tr),format_pools=pools,seconds=time.time()-t0,sha256=sha(path)))
        seen.extend(te.tolist())
        print('FOLD',dataset,split,len(te),round(time.time()-t0,2),flush=True)
    np.testing.assert_array_equal(np.sort(seen),np.arange(len(r)))
    verify(provenance)
    write(out/'complete.json',dict(complete=True,panel=panel,dataset=dataset,n_queries=len(r),
        n_models=r.shape[1],lambdas=lams,folds=fold_summaries,
        format_by_family={f:np.unique(fmt[d['fam']==i]).tolist() for i,f in enumerate(d['meta']['families'])},
        source_hashes=provenance,seconds=time.time()-start,slurm_job=os.environ.get('SLURM_JOB_ID')))
    print('COMPLETE',dataset,flush=True)

def report(panel):
    dataset=DATASETS[panel]; out=OUT/dataset
    completion=json.loads((out/'complete.json').read_text()); verify(completion['source_hashes'])
    d=load(panel,'gte'); raw=dict(np.load(ROOT/f'datasets/{dataset}/prepared/panel.npz',allow_pickle=False))
    r,c=d['r'].astype(np.float64),d['c'].astype(np.float64); fmt=raw['format']
    lams=completion['lambdas']; splits=json.loads(split_path(panel).read_text())
    preds={name:{t:np.empty_like(r) for t in ['r','c']} for name in PREDICTORS}
    utilities={name:np.empty((len(r),len(lams))) for name in POLICIES}
    choices={name:np.empty((len(r),len(lams)),dtype=np.int16) for name in POLICIES}
    audit_retrieval_rows=0
    for f in completion['folds']:
        split=f['split']; path=out/(split+'.npz')
        assert sha(path)==f['sha256']
        tr=np.asarray(splits[split]['train']); te=np.asarray(splits[split]['test'])
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['test_rows'],te)
            np.testing.assert_array_equal(z['query_ids'],d['query_ids'][te])
            nn=z['gte_same_format_neighbors']
            assert np.isin(nn,tr).all() and (fmt[nn]==fmt[te,None]).all()
            # Outcome-independent exact retrieval spot audit including boundaries and interior.
            for label in np.unique(fmt[te]):
                loc=np.flatnonzero(fmt[te]==label)
                last=((len(loc)-1)//128)*128
                sample=loc[np.unique(np.r_[np.arange(min(128,len(loc))),np.arange(last,len(loc))])]
                pool=tr[fmt[tr]==label]
                ids,dist=nearest(d['x'][te[sample]],d['x'][pool],100)
                np.testing.assert_array_equal(nn[sample],pool[ids])
                audit_retrieval_rows += len(sample)
            for name in PREDICTORS:
                for target,true,original in [('r',r,d['r']),('c',c,d['c'])]:
                    pred=z[name+'|'+target]; preds[name][target][te]=pred
                    if name=='gte_same_format':
                        for st in range(0,len(te),512):
                            np.testing.assert_array_equal(pred[st:st+512],original[nn[st:st+512]].mean(1))
                    elif name=='source_mean':
                        np.testing.assert_array_equal(pred,np.broadcast_to(original[tr].mean(0),pred.shape))
            for name,(rp,cp) in POLICIES.items():
                ch=np.stack([(z[rp+'|r']-lam*z[cp+'|c']).argmax(1) for lam in lams],axis=1)
                np.testing.assert_array_equal(ch,z['choice|'+name]); choices[name][te]=ch
                for il,lam in enumerate(lams):
                    utilities[name][te,il]=(r[te]-lam*c[te])[np.arange(len(te)),ch[:,il]]
    columns=[]; specs=[]
    for name in PREDICTORS:
        for target,true in [('r',r),('c',c)]:
            err=preds[name][target]-true
            for center in [False,True]:
                e=err-err.mean(1,keepdims=True) if center else err
                for metric,v in [('mse',(e*e).mean(1)),('mae',np.abs(e).mean(1))]:
                    specs.append((name,target,'centered' if center else 'absolute',metric))
                    columns.append(v)
    metric_count=len(specs)
    for name in POLICIES:
        for il,lam in enumerate(lams):
            specs.append((name,'utility',str(lam)));columns.append(utilities[name][:,il])
        specs.append((name,'utility','mean'));columns.append(utilities[name].mean(1))
    data=np.column_stack(columns); point=data.mean(0)
    boot=boot_sums(data,np.ones(len(r)),families=d['fam'])
    index={tuple(s):i for i,s in enumerate(specs)}
    rows=[dict(key=list(s),**record(point[i],boot[:,i])) for i,s in enumerate(specs)]
    # Paired contrasts; preserve all controls, penalties and unfavorable results.
    contrasts=[]
    for name in POLICIES:
        for ref in ['gte_reward_gte_cost','l21_reward_l21_cost']:
            if name==ref:continue
            for label in [str(v) for v in lams]+['mean']:
                a=index[(name,'utility',label)];b=index[(ref,'utility',label)]
                contrasts.append(dict(policy=name,reference=ref,penalty=label,
                    **record(point[a]-point[b],boot[:,a]-boot[:,b])))
    error_ratios=[]
    for name in PREDICTORS:
        for ref in ['gte','source_mean']:
            if name==ref:continue
            for target in ['r','c']:
                for centering in ['absolute','centered']:
                    for metric in ['mse','mae']:
                        a=index[(name,target,centering,metric)];b=index[(ref,target,centering,metric)]
                        assert point[b]>0 and np.all(boot[:,b]>0)
                        error_ratios.append(dict(predictor=name,reference=ref,target=target,centering=centering,metric=metric,
                            **record(1-point[a]/point[b],1-boot[:,a]/boot[:,b])))
    per_family=[]
    for group,fam in enumerate(d['meta']['families']):
        use=d['fam']==group
        per_family.append(dict(family=fam,n_queries=int(use.sum()),
            metrics=[dict(key=list(s),mean=float(data[use,i].mean())) for i,s in enumerate(specs)]))
    per_model=[]
    for name in PREDICTORS:
        for target,true in [('r',r),('c',c)]:
            e=preds[name][target]-true
            for j,model in enumerate(d['meta']['models']):
                per_model.append(dict(predictor=name,target=target,model=model,
                    mse=float(np.square(e[:,j]).mean()),mae=float(np.abs(e[:,j]).mean())))
    verify(completion['source_hashes'])
    np.savez_compressed(out/'paired_metrics.npz',rows=np.arange(len(r)),family=d['fam'],
        metric_keys=np.array([json.dumps(s) for s in specs]),per_query=data,bootstrap=boot)
    write(out/'report.json',dict(complete=True,dataset=dataset,rows=rows,paired_utility_differences=contrasts,
        relative_error_reductions=error_ratios,per_family=per_family,per_model=per_model,
        equal_family_means=[dict(key=list(s),mean=float(np.mean([f['metrics'][i]['mean'] for f in per_family]))) for i,s in enumerate(specs)],
        audit=dict(folds=len(completion['folds']),queries=len(r),retrieval_rows_checked=audit_retrieval_rows,
            every_restricted_prediction_reconstructed=True,every_routing_choice_reconstructed=True,
            original_gte_l21_scores_reproduced=True,source_hashes_verified=True),
        slurm_job=os.environ.get('SLURM_JOB_ID')))
    print('AUDITED',dataset,len(r),'queries',flush=True)
    for name in POLICIES:
        print(name,record(point[index[(name,'utility','mean')]],boot[:,index[(name,'utility','mean')]]),flush=True)
    for target in ['r','c']:
        for name in PREDICTORS:
            print(target,name,'MSE',point[index[(name,target,'absolute','mse')]],'centered',point[index[(name,target,'centered','mse')]],flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['run','report']);a.add_argument('--task',type=int,required=True)
    args=a.parse_args();panel=['routerbench-0shot','sprout-sixfam'][args.task]
    (run if args.action=='run' else report)(panel)
