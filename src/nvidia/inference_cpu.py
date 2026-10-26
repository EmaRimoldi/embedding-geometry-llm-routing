"""Matched CPU deployment latency; separate from frozen training and selection."""
import argparse, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common_nv import ROOT, OUT, cfg, data, splits, base, verify, sha, sig, features_dir, transform_dir, fit_path, trunk, atomic_json, atomic_npz
from timing import hardware

CONFIG = ROOT/'configs/nvidia_inference_cpu.json'
DEST = OUT/'inference_cpu_v1'

def timed(call, repeats):
    call()  # One untimed warmup; retain every subsequent measurement.
    seconds = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        seconds.append(time.perf_counter()-start)
    return result, seconds

def chunks(x, batch):
    return [x[i:i+batch] for i in range(0, len(x), batch)]

def setup_transform(path):
    z = dict(np.load(path/'transform.npz', allow_pickle=False))
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = z['scaler_mean'], z['scaler_scale']
    scaler.n_features_in_ = len(scaler.mean_)
    pca = PCA(n_components=100, whiten=False)
    pca.components_, pca.mean_ = z['components'], z['pca_mean']
    pca.explained_variance_ = z['explained_variance']
    pca.n_components_ = 100
    return z, scaler, pca

def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run under Slurm'
    conf = json.loads(CONFIG.read_text())
    verify()
    for filename in ['results/nvidia/provenance/input_hashes.json',
                     'results/nvidia/inference_cpu_v1/source_hashes.json']:
        for p, h in json.loads((ROOT/filename).read_text()).items():
            assert base.sha(ROOT/p) == h, p
    torch.set_num_threads(conf['threads'])
    torch.set_num_interop_threads(1)
    job = os.environ['SLURM_JOB_ID']
    started = time.monotonic()
    hw = dict(hardware(), torch_version=torch.__version__, numpy_version=np.__version__,
              threadpools=threadpool_info(), device='cpu')
    invocation = DEST/'runs'/job
    invocation.mkdir(parents=True, exist_ok=False)
    atomic_json(invocation/'hardware.json', hw)
    atomic_json(invocation/'protocol.json', conf)
    mod = trunk()
    completed = []
    missing = []
    with threadpool_limits(limits=conf['threads']):
      for panel, sp in cfg()['tasks']:
        old = DEST/'completed'/base.DATASETS[panel]/(sp+'.json')
        if old.exists():
            assert json.loads(old.read_text())['config_sha256'] == sha(CONFIG)
            continue
        source = OUT/'results'/base.DATASETS[panel]/sp
        if not (source/'result.json').exists():
            missing.append([panel, sp]); continue
        if time.monotonic()-started > conf['yield_seconds']:
            missing.append([panel, sp]); continue
        dest = invocation/base.DATASETS[panel]/sp
        dest.mkdir(parents=True)
        rec = json.loads((source/'result.json').read_text())
        assert rec['config_sha256'] == sha(ROOT/'configs/nvidia.json')
        s = splits(panel)[sp]
        tr, te = np.asarray(s['train']), np.asarray(s['test'])
        assert not np.intersect1d(tr, te).size
        d = data(panel); m = d['r'].shape[1]
        order = sorted(range(len(te)), key=lambda i: hashlib.sha256(
            (conf['sampling_salt']+'|'+panel+'|'+sp+'|'+str(d['query_ids'][te[i]])).encode()).hexdigest())
        positions = np.asarray(order[:conf['max_queries']])
        rows = te[positions]
        setup = {}; audits = {}; records = []; artifacts = {}
        def track(path):
            artifacts[str(path.relative_to(ROOT))] = sha(path)
        track(source/'result.json'); track(source/'predictions.npz'); track(base.split_path(panel))
        saved = dict(np.load(source/'predictions.npz', allow_pickle=False))
        np.testing.assert_array_equal(saved['test_rows'], te)
        np.testing.assert_array_equal(saved['query_ids'], d['query_ids'][te])
        raw = {}
        for layer in sorted({21, rec['selection']['layer']}):
            folder = features_dir(panel, layer)
            track(folder/'manifest.json'); track(folder/'vectors.npy')
            manifest = json.loads((folder/'manifest.json').read_text())
            assert artifacts[str((folder/'vectors.npy').relative_to(ROOT))] == manifest['vector_sha256']
            start = time.perf_counter()
            raw[layer] = np.load(folder/'vectors.npy', mmap_mode='r')
            assert raw[layer].shape[0] == len(d['r'])
            # Materializing the common deterministic query sample is I/O/setup.
            q = np.asarray(raw[layer][rows]).copy()
            assert np.isfinite(q).all()
            setup['raw_query_materialization_layer_'+str(layer)] = time.perf_counter()-start
            raw[layer] = (raw[layer], q)
        start = time.perf_counter()
        train_x = base.unit(raw[21][0][tr])
        train_r, train_c = d['r'][tr], d['c'][tr]
        setup['knn_source_normalization_and_targets_seconds'] = time.perf_counter()-start
        knn_predictions = {}
        for batch in conf['batch_sizes']:
            batch_raw = chunks(raw[21][1], batch)
            def preprocess_knn():
                return [base.unit(v) for v in batch_raw]
            qs, preprocess_times = timed(preprocess_knn, conf['repeats'])
            def knn():
                rr, cc, ii = [], [], []
                for q in qs:
                    ix, _ = base.nearest(q, train_x, min(100, len(tr)))
                    # Exactly the frozen arithmetic, unweighted neighbour means.
                    rr.append(train_r[ix].mean(1).astype(np.float64))
                    cc.append(train_c[ix].mean(1).astype(np.float64))
                    ii.append(tr[ix])
                return np.concatenate(rr), np.concatenate(cc), np.concatenate(ii)
            (rr, cc, ids), prediction_times = timed(knn, conf['repeats'])
            assert np.isfinite(rr).all() and np.isfinite(cc).all()
            assert np.isin(ids, tr).all() and not np.isin(ids, rows).any()
            assert all(len(set(x)) == min(100,len(tr)) for x in ids)
            np.testing.assert_allclose(rr, d['r'][ids].mean(1), rtol=0, atol=0)
            np.testing.assert_allclose(cc, d['c'][ids].mean(1), rtol=0, atol=0)
            knn_predictions[batch] = dict(reward=rr, cost=cc, neighbours=ids)
            records.append(dict(method='knn100', batch_size=batch, phase='query_normalization',
                                seconds=preprocess_times, n_queries=len(rows), shared_across_penalties=True))
            records.append(dict(method='knn100', batch_size=batch, phase='retrieval_and_reward_cost_means',
                                seconds=prediction_times, n_queries=len(rows), shared_across_penalties=True))
            for lam in cfg()['lambdas'][panel]:
                def route():
                    return [(r-lam*c).argmax(1) for r,c in zip(chunks(rr,batch),chunks(cc,batch))]
                _, times = timed(route, conf['repeats'])
                records.append(dict(method='knn100', batch_size=batch, lambda_value=lam,
                                    phase='utility_and_choice', seconds=times, n_queries=len(rows)))
        # Compare CPU retrieval batch sizes and historical scores without hiding precision differences.
        baseline = base.destination('knn', panel, sp, 'reader')/'predictions.npz'
        track(baseline)
        with np.load(baseline, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['test_rows'], te)
            for batch, pred in knn_predictions.items():
                audits['knn_batch_'+str(batch)] = dict(
                    max_reward_difference_from_saved=float(np.max(np.abs(pred['reward']-z['knn100|r'][positions]))),
                    max_cost_difference_from_saved=float(np.max(np.abs(pred['cost']-z['knn100|c'][positions]))),
                    choice_disagreements={str(lam): int(np.count_nonzero(
                      (pred['reward']-lam*pred['cost']).argmax(1) != z[f'{lam:g}|knn100'][positions]))
                      for lam in cfg()['lambdas'][panel]})
                atomic_npz(dest/f'knn_batch_{batch}.npz', rows=rows, query_ids=d['query_ids'][rows], **pred)
        # Release source matrix before checkpoint measurements; limits are the same four CPU threads.
        del train_x
        transformed = {}
        for layer in sorted(raw):
            path = transform_dir(panel, tr, layer)
            for file in ['ready.json','transform.npz','features.npy']: track(path/file)
            info = json.loads((path/'ready.json').read_text())
            assert info['identity']['train_row_hash'] == sig(tr)
            assert info['identity']['feature_sha256'] == sha(features_dir(panel,layer)/'vectors.npy')
            assert info['feature_output_sha256'] == sha(path/'features.npy')
            start = time.perf_counter()
            z, scaler, pca = setup_transform(path)
            np.testing.assert_array_equal(z['train_rows'],tr)
            setup['transform_loading_layer_'+str(layer)] = time.perf_counter()-start
            cached = np.load(path/'features.npy',mmap_mode='r')[rows]
            for batch in conf['batch_sizes']:
                qr = chunks(raw[layer][1],batch)
                def preprocess_nv():
                    return [np.tile(pca.transform(scaler.transform(q)).astype(np.float32),(1,m)) for q in qr]
                xs, times = timed(preprocess_nv, conf['repeats'])
                projected = np.concatenate(xs)[:,:100]
                np.testing.assert_allclose(projected,cached,rtol=conf['rtol'],atol=conf['atol'])
                transformed[layer,batch] = xs
                records.append(dict(method='nvidia', layer=layer, batch_size=batch,
                                    phase='scaling_pca_and_repeated_blocks', seconds=times,n_queries=len(rows),
                                    shared_across_penalties=True))
        unique = {}
        for control in ['selected_layer','fixed_l21']:
          for lam in cfg()['lambdas'][panel]:
            ensemble = rec['ensembles'][control][str(lam)]
            layer, seeds = ensemble['layer'], ensemble['retained_seeds']
            assert len(seeds)==5 and len(set(seeds))==5
            identity = (layer,float(lam),tuple(seeds))
            if identity in unique:
                audits[control+'|'+str(lam)] = dict(reused_from=unique[identity])
                continue
            unique[identity] = control+'|'+str(lam)
            start = time.perf_counter()
            nets = []
            for seed in seeds:
                cp = fit_path(panel,sp,layer,lam,seed,'final').with_suffix('.pt')
                track(cp)
                net = mod.SharedTrunkNet(100*m,m)
                net.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
                nets.append(net.eval())
            setup[control+'|'+str(lam)+'|checkpoint_load_seconds'] = time.perf_counter()-start
            for batch in conf['batch_sizes']:
                xs = transformed[layer,batch]
                def predict():
                    ys = []
                    with torch.inference_mode():
                      for x in xs:
                        tensor = torch.as_tensor(x)
                        preds = [net(tensor).numpy() for net in nets]
                        y = np.mean(np.asarray(preds),axis=0)
                        ys.append((y,y.argmax(1)))
                    return np.concatenate([p[0] for p in ys]),np.concatenate([p[1] for p in ys])
                (pred, choices), times = timed(predict, conf['repeats'])
                expected = saved[f'{control}|{lam:g}|prediction'][positions]
                assert np.isfinite(pred).all()
                np.testing.assert_allclose(pred,expected,rtol=conf['rtol'],atol=conf['atol'])
                audits[control+'|'+str(lam)+'|batch_'+str(batch)] = dict(
                    max_prediction_difference=float(np.max(np.abs(pred-expected))),
                    choice_disagreements=int(np.count_nonzero(choices!=expected.argmax(1))))
                atomic_npz(dest/f'{control}_lambda_{lam:g}_batch_{batch}.npz',
                           rows=rows,prediction=pred,choice=choices)
                records.append(dict(method=control,layer=layer,retained_seeds=seeds,lambda_value=lam,
                                    batch_size=batch,phase='five_networks_average_and_choice',
                                    seconds=times,n_queries=len(rows)))
            del nets
        result = dict(panel=panel,split=sp,config_sha256=sha(CONFIG),hardware=hw,
                      rows=rows.tolist(),query_ids=d['query_ids'][rows].tolist(),n_train=len(tr),
                      n_test=len(te),setup=setup,measurements=records,audits=audits,
                      artifact_hashes=artifacts,scope=conf['scope'])
        atomic_json(dest/'result.json',result)
        atomic_json(old,dict(config_sha256=sha(CONFIG),result=str((dest/'result.json').relative_to(ROOT)),
                             sha256=base.sha(dest/'result.json'),job_id=job))
        completed.append([panel,sp])
        print('INFERENCE_COMPLETE',panel,sp,len(rows),flush=True)
        del raw, transformed, saved, knn_predictions
        gc.collect()
    atomic_json(invocation/'status.json',dict(completed=completed,missing=missing,hardware=hw,
                 elapsed_seconds=time.monotonic()-started))
    print('INFERENCE_DONE',len(completed),'new folds;',len(missing),'not ready',flush=True)

if __name__ == '__main__':
    main()
