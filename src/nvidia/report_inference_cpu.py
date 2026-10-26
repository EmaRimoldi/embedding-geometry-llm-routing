"""Independent coverage, provenance and prediction audit of CPU latency artifacts."""
import collections,csv,hashlib,json,os,time,subprocess
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common_nv import ROOT,OUT,cfg,data,splits,base,sha,verify,fit_path,atomic_json
DEST=OUT/'inference_cpu_v1'
CONFIG=ROOT/'configs/nvidia_inference_cpu.json'

def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    assert os.environ.get('SLURM_JOB_ID')
    verify();conf=json.loads(CONFIG.read_text());job=os.environ['SLURM_JOB_ID']
    for name in ['source_hashes.json','audit_source_hashes.json']:
        for path,h in json.loads((DEST/name).read_text()).items():assert sha(ROOT/path)==h,path
    output=DEST/'audits'/job;output.mkdir(parents=True,exist_ok=False)
    missing=[];folds=[];latencies=[];sweeps=[];measurement_rows=[];numerics=[];artifacts={}
    with threadpool_limits(limits=conf['threads']):
      for panel,sp in cfg()['tasks']:
        marker=DEST/'completed'/base.DATASETS[panel]/(sp+'.json')
        if not marker.exists():missing.append([panel,sp]);continue
        mr=json.loads(marker.read_text());assert mr['config_sha256']==sha(CONFIG)
        p=ROOT/mr['result'];assert sha(p)==mr['sha256'];r=json.loads(p.read_text());dest=p.parent
        assert r['panel']==panel and r['split']==sp and r['config_sha256']==sha(CONFIG)
        assert r['hardware']['device']=='cpu' and int(r['hardware']['cpu_threads'])==conf['threads']
        for name,h in r['artifact_hashes'].items():assert sha(ROOT/name)==h,name
        artifacts[str(p.relative_to(ROOT))]=sha(p)
        d=data(panel);s=splits(panel)[sp];tr=np.asarray(s['train']);te=np.asarray(s['test'])
        order=sorted(range(len(te)),key=lambda i:hashlib.sha256(
            (conf['sampling_salt']+'|'+panel+'|'+sp+'|'+str(d['query_ids'][te[i]])).encode()).hexdigest())
        positions=np.asarray(order[:conf['max_queries']]);rows=te[positions];n=len(rows)
        np.testing.assert_array_equal(r['rows'],rows)
        np.testing.assert_array_equal(r['query_ids'],d['query_ids'][rows])
        assert r['n_train']==len(tr) and r['n_test']==len(te)
        source=OUT/'results'/base.DATASETS[panel]/sp
        nv=json.loads((source/'result.json').read_text());saved=dict(np.load(source/'predictions.npz'))
        np.testing.assert_array_equal(saved['test_rows'],te)
        hist=dict(np.load(base.destination('knn',panel,sp,'reader')/'predictions.npz'))
        raw=np.load(ROOT/'artifacts/embeddings/qwen2.5-7b-instruct/layer_21'/base.DATASETS[panel]/'vectors.npy',mmap_mode='r')
        train_x=base.unit(raw[tr]);query_x=base.unit(raw[rows])
        for batch in conf['batch_sizes']:
            z=dict(np.load(dest/f'knn_batch_{batch}.npz'))
            np.testing.assert_array_equal(z['rows'],rows);np.testing.assert_array_equal(z['query_ids'],d['query_ids'][rows])
            ids=z['neighbours'];k=min(100,len(tr));assert ids.shape==(n,k)
            assert np.isin(ids,tr).all() and not np.isin(ids,rows).any()
            assert all(len(set(x))==k for x in ids)
            np.testing.assert_allclose(z['reward'],d['r'][ids].mean(1),rtol=0,atol=0)
            np.testing.assert_allclose(z['cost'],d['c'][ids].mean(1),rtol=0,atol=0)
            discrepancy=0;largest=0.
            for st in range(0,n,batch):
                sim=query_x[st:st+batch]@train_x.T
                order_now=np.argsort(-sim,axis=1,kind='stable')
                expected=tr[order_now[:,:k]]
                for j,actual in enumerate(ids[st:st+batch]):
                    if not np.array_equal(actual,expected[j]):
                        # Check membership optimality up to documented float32 rounding,
                        # without changing any measured neighbours or predictions.
                        local=np.searchsorted(tr,actual) if np.all(tr[:-1]<=tr[1:]) else np.asarray([np.flatnonzero(tr==v)[0] for v in actual])
                        gap=float(sim[j,order_now[j,k-1]]-np.min(sim[j,local]))
                        assert gap<=8*np.finfo(np.float32).eps,(panel,sp,batch,gap)
                        discrepancy+=1;largest=max(largest,gap)
            row=dict(panel=panel,split=sp,method='knn100',batch=batch,lambda_value='all',
                     retrieval_rounding_rows=discrepancy,max_boundary_gap=largest,
                     max_prediction_difference=float(np.max(np.abs(z['reward']-hist['knn100|r'][positions]))),
                     choice_disagreements=sum(int(np.count_nonzero((z['reward']-lam*z['cost']).argmax(1)!=hist[f'{lam:g}|knn100'][positions])) for lam in cfg()['lambdas'][panel]))
            numerics.append(row)
            assert row['choice_disagreements']==sum(r['audits']['knn_batch_'+str(batch)]['choice_disagreements'].values())
        del train_x,query_x,raw
        for control in ['selected_layer','fixed_l21']:
          for lam in cfg()['lambdas'][panel]:
            en=nv['ensembles'][control][str(lam)];layer=en['layer']
            expected_seeds=sorted(cfg()['ensemble_seeds'],key=lambda seed:(json.loads(fit_path(panel,sp,layer,lam,seed,'final').read_text())['validation_mse'],seed))[:5]
            assert en['retained_seeds']==expected_seeds
            alias=r['audits'].get(control+'|'+str(lam),{}).get('reused_from')
            actual=alias.split('|')[0] if alias else control
            if alias:assert en==nv['ensembles'][actual][str(lam)]
            for batch in conf['batch_sizes']:
                z=dict(np.load(dest/f'{actual}_lambda_{lam:g}_batch_{batch}.npz'))
                np.testing.assert_array_equal(z['rows'],rows)
                pred=z['prediction'];expected=saved[f'{control}|{lam:g}|prediction'][positions]
                assert pred.shape==expected.shape and np.isfinite(pred).all()
                np.testing.assert_allclose(pred,expected,rtol=conf['rtol'],atol=conf['atol'])
                np.testing.assert_array_equal(z['choice'],pred.argmax(1))
                diff=int(np.count_nonzero(z['choice']!=expected.argmax(1)))
                assert diff==r['audits'][actual+'|'+str(lam)+'|batch_'+str(batch)]['choice_disagreements']
                numerics.append(dict(panel=panel,split=sp,method=control,batch=batch,lambda_value=lam,
                  retrieval_rounding_rows=0,max_boundary_gap=0.,
                  max_prediction_difference=float(np.max(np.abs(pred-expected))),choice_disagreements=diff))
        def get(method,phase,batch,lam=None,layer=None):
            matches=[v for v in r['measurements'] if v['method']==method and v['phase']==phase and v['batch_size']==batch
                     and (lam is None or v.get('lambda_value')==lam) and (layer is None or v.get('layer')==layer)]
            assert len(matches)==1,(panel,sp,method,phase,batch,lam,layer)
            return float(np.median(matches[0]['seconds']))
        for v in r['measurements']:
            assert v['n_queries']==n and len(v['seconds'])==conf['repeats']
            assert np.isfinite(v['seconds']).all() and min(v['seconds'])>0
            for repeat,seconds in enumerate(v['seconds']):
                measurement_rows.append(dict(panel=panel,split=sp,host=r['hardware']['host'],cpu=r['hardware']['cpu_model'],
                    method=v['method'],phase=v['phase'],batch_size=v['batch_size'],lambda_value=v.get('lambda_value','shared'),
                    layer=v.get('layer',''),repetition=repeat,n_queries=n,seconds=seconds))
        common=dict(panel=panel,split=sp,protocol=sp.split('_')[0],host=r['hardware']['host'],
                    cpu=r['hardware']['cpu_model'],n_queries=n,n_source=len(tr),job_id=mr['job_id'])
        for batch in conf['batch_sizes']:
          kpre=get('knn100','query_normalization',batch)
          kpred=get('knn100','retrieval_and_reward_cost_means',batch)
          for method in ['knn100','selected_layer','fixed_l21']:
            total_prediction=0.;preprocess=None
            for lam in cfg()['lambdas'][panel]:
                if method=='knn100':
                    preprocess=kpre;decision=get(method,'utility_and_choice',batch,lam)
                    prediction=kpred+decision;total_prediction+=decision
                else:
                    en=nv['ensembles'][method][str(lam)]
                    alias=r['audits'].get(method+'|'+str(lam),{}).get('reused_from')
                    actual=alias.split('|')[0] if alias else method
                    preprocess=get('nvidia','scaling_pca_and_repeated_blocks',batch,layer=en['layer'])
                    prediction=get(actual,'five_networks_average_and_choice',batch,lam,en['layer'])
                    total_prediction+=prediction
                latencies.append(dict(**common,method=method,batch_size=batch,lambda_value=lam,
                    preprocessing_seconds=preprocess,prediction_seconds=prediction,
                    total_seconds=preprocess+prediction,ms_per_query=1000*(preprocess+prediction)/n))
            if method=='knn100':total_prediction+=kpred
            sweeps.append(dict(**common,method=method,batch_size=batch,penalties=len(cfg()['lambdas'][panel]),
                        total_seconds=preprocess+total_prediction,ms_per_query_all_penalties=1000*(preprocess+total_prediction)/n))
        folds.append(dict(panel=panel,split=sp,queries=n,hardware=r['hardware'],result_sha256=sha(p)))
    write_csv(output/'per_fold_penalty_latency.csv',latencies)
    write_csv(output/'all_penalty_amortized_latency.csv',sweeps)
    write_csv(output/'raw_measurements.csv',measurement_rows)
    write_csv(output/'numerical_differences.csv',numerics)
    # Hardware-stratified summaries; all penalties retained, equal penalty weights.
    groups=collections.defaultdict(list)
    for r in latencies:groups[(r['panel'],r['protocol'],r['cpu'],r['method'],r['batch_size'])].append(r)
    aggregates=[]
    for key,rs in sorted(groups.items()):
        aggregates.append(dict(zip(['panel','protocol','cpu','method','batch_size'],key),
            folds=len({r['split'] for r in rs}),queries=sum(r['n_queries'] for r in rs)/13,
            ms_per_query=1000*sum(r['total_seconds'] for r in rs)/sum(r['n_queries'] for r in rs)))
    write_csv(output/'hardware_stratified_summary.csv',aggregates)
    jobs=sorted({r['job_id'] for r in latencies})
    scheduler=subprocess.check_output(['sacct','-X','-P','-j',','.join(jobs+[job]),
                   '--format=JobIDRaw,State,ElapsedRaw,AllocCPUS,AllocTRES,Submit,Start,End,NodeList'],text=True)
    (output/'scheduler.txt').write_text(scheduler)
    result=dict(completed_folds=len(folds),expected_folds=23,all_complete=not missing,missing=missing,
        audited_latency_rows=len(latencies),expected_latency_rows=23*13*2*3,folds=folds,result_hashes=artifacts,
        numerical_choice_disagreements=sum(r['choice_disagreements'] for r in numerics),
        notes=['CPU deployment from precomputed features only; four threads, warm caches, deterministic bounded test sample.',
               'Per-penalty latency sums independently measured phase medians; not a single fused end-to-end timer.',
               'All-penalty work reuses preprocessing and kNN retrieval. Single-penalty results do not divide retrieval by13.',
               'Setup includes integrity hashing; raw setup fields are not pure cold-start latency.',
               'Summaries are stratified by CPU model and protocol, weighted by sampled query count with equal penalty weight.',
               'Point measurements are not uncertainty estimates. Full frozen performance comparison is reported separately.',
               'Original validated-kNN runtime bundles its search and is not kNN100 training time.'])
    atomic_json(output/'audit.json',result)
    atomic_json(DEST/'latest_audit.json',dict(path=str((output/'audit.json').relative_to(ROOT)),sha256=sha(output/'audit.json'),all_complete=not missing))
    print('INFERENCE_AUDIT',len(folds),'/23 folds;',len(latencies),'latency rows;',len(missing),'missing',flush=True)
if __name__=='__main__':main()
