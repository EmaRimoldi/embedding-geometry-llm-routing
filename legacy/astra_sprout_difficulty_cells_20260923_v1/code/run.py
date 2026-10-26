"""Training-only grouping controls; isolated from all existing experiments."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
E4 = Path(os.environ.get('E4_ROOT', ROOT.parent))
DEFINITIVE = E4 / 'astra_definitive_paper_20260923_v1'
sys.path.insert(0, str(DEFINITIVE / 'code'))
from common import load, sha

PROXIES = ['solve_fraction_050', 'mean_reward', 'reward_profile',
           'solve_fraction_025', 'solve_fraction_075']

def compress(a):
    return np.stack([a[..., 0], a[..., 1], a[..., 2:-2].mean(-1),
                     a[..., -2], a[..., -1]], axis=-1)

def abs_expect(q, donors):
    """Exact E|q-Y|, in O(n log n), with no pairwise-outcome tensor."""
    ans = np.empty_like(q, dtype=np.float64)
    for j in range(q.shape[1]):
        y = np.sort(donors[:, j]); c = np.r_[0., np.cumsum(y)]
        k = np.searchsorted(y, q[:, j], side='right')
        ans[:, j] = (q[:, j]*k-c[k]+c[-1]-c[k]-q[:, j]*(len(y)-k))/len(y)
    return ans

def nearest(sim, k):
    # Full stable sort explicitly defines tied cosine ordering by sorted row ID.
    return np.argsort(-sim, axis=1, kind='stable')[:, :min(k, sim.shape[1])]

def evaluate(x, y, tr, te, labels, fam, threshold=1, cutoff=None):
    nt, p = len(te), y.shape[1]
    out = {'base_abs': np.full((nt, 5), np.nan),
           'base_mse': np.full((nt, 5), np.nan),
           'pool_n': np.zeros(nt, np.int32), 'fallback': np.zeros(nt, bool)}
    ks = [10] if cutoff is not None else [10, 100]
    for k in ks:
        out[f'knn{k}_abs'] = np.full((nt, 5), np.nan)
        out[f'knn{k}_mse'] = np.full((nt, 5), np.nan)
        out[f'nn{k}'] = np.full((nt, k), -1, np.int32)
    for lab in np.unique(labels[te]):
        pos = np.flatnonzero(labels[te] == lab); q = te[pos]
        donors = tr[labels[tr] == lab]
        if len(donors) < threshold:
            assert len(np.unique(fam[q])) == 1
            donors = tr[fam[tr] == fam[q[0]]]; out['fallback'][pos] = True
        assert len(donors) and not np.intersect1d(donors, q).size
        donors = np.sort(donors)
        base = abs_expect(y[q], y[donors])
        out['base_abs'][pos] = compress(base)
        out['base_mse'][pos] = compress((y[q]-y[donors].mean(0))**2)
        out['pool_n'][pos] = len(donors)
        for start in range(0, len(q), 128):
            qb = q[start:start+128]; ip = pos[start:start+128]
            sim = x[qb] @ x[donors].T
            if cutoff is not None:
                eligible = sim <= cutoff
                counts = eligible.sum(1); valid = counts >= 10
                # Baseline and neighbours have exactly the same filtered pool.
                out['base_abs'][ip] = np.nan; out['base_mse'][ip] = np.nan
                out['pool_n'][ip] = counts
                if not valid.any(): continue
                qb, ip, sim, eligible, counts = qb[valid], ip[valid], sim[valid], eligible[valid], counts[valid]
                av = np.empty((len(qb), p), float)
                for j in range(p):
                    av[:, j] = (np.abs(y[qb, j, None]-y[donors, j][None])*eligible).sum(1)/counts
                pred = eligible @ y[donors] / counts[:, None]
                out['base_abs'][ip] = compress(av)
                out['base_mse'][ip] = compress((y[qb]-pred)**2)
                sim[~eligible] = -np.inf
            order = nearest(sim, max(ks))
            for k in ks:
                nn = donors[order[:, :k]]; n_eff = nn.shape[1]
                out[f'nn{k}'][ip, :n_eff] = nn
                out[f'knn{k}_abs'][ip] = compress(np.abs(y[qb, None]-y[nn]).mean(1))
                out[f'knn{k}_mse'][ip] = compress((y[qb]-y[nn].mean(1))**2)
    return out

def refine(x, tr, te, cells, requested, seed):
    labels = np.full(len(x), -1, np.int32); records = []; serial = 0
    for cell in np.unique(cells):
        ti = tr[cells[tr] == cell]; qi = te[cells[te] == cell]
        k = max(1, min(requested, len(ti)//50))
        if not len(ti):
            # Unseen cells retain a unique empty code and use family fallback.
            labels[qi] = serial; records.append(dict(cell=str(cell), k=0, n_train=0, n_test=len(qi)))
            serial += 1; continue
        if k == 1:
            labt = np.zeros(len(ti), int); labq = np.zeros(len(qi), int)
        else:
            km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=seed,
                        algorithm='lloyd').fit(x[ti])
            labt = km.labels_; labq = km.predict(x[qi]) if len(qi) else np.array([], int)
        labels[ti] = serial+labt; labels[qi] = serial+labq
        # Collapse clusters with fewer than 10 training rows to the full cell.
        # All queries in that cell use the unsplit cell in this rare case,
        # preserving a partition for the exact donor-composition decomposition.
        counts = np.bincount(labt, minlength=k)
        collapsed = bool(counts.min() < 10)
        if collapsed:
            labels[ti] = serial; labels[qi] = serial
        records.append(dict(cell=str(cell), requested=requested, k=k,
                            counts=counts.tolist(), collapsed=collapsed,
                            n_train=len(ti), n_test=len(qi)))
        serial += k
    assert (labels[np.r_[tr, te]] >= 0).all()
    return labels, records

def matched_donor_random(y, tr, te, nn, labels):
    """Replace each global neighbour by a random train donor of its group."""
    out = np.zeros((len(te), y.shape[1]), float)
    groups = labels[nn]
    for group in np.unique(groups):
        positions = np.flatnonzero((groups == group).any(1))
        weight = (groups[positions] == group).mean(1)
        out[positions] += weight[:, None]*abs_expect(y[te[positions]], y[tr[labels[tr] == group]])
    return compress(out)

def conditional_permutation(y, tr, te, cells, fam, nn, seed):
    rng = np.random.default_rng(51000+seed)
    pos = np.concatenate([np.sort(rng.choice(np.flatnonzero(fam[te] == g),
                   min(250, int((fam[te] == g).sum())), replace=False)) for g in np.unique(fam)])
    valid = (nn[pos] >= 0).all(1); pos = pos[valid]; q = te[pos]; nb = nn[pos]
    # Exclude unseen-cell family fallbacks from this conditional cell null.
    valid = (cells[nb] == cells[q, None]).all(1); q, nb = q[valid], nb[valid]
    pools = [tr[cells[tr] == c] for c in np.unique(cells[tr])]
    observed = compress(np.abs(y[q, None]-y[nb]).mean(1)).mean(0)
    null = []; index = np.arange(len(y))
    for _ in range(199):
        for pool in pools: index[pool] = rng.permutation(pool)
        null.append(compress(np.abs(y[q, None]-y[index[nb]]).mean(1)).mean(0))
    null = np.asarray(null)
    return dict(sample_rows=q.tolist(), n=len(q), observed=observed.tolist(),
                null_mean=null.mean(0).tolist(), null=null.tolist(),
                p_one_sided=((1+(null <= observed).sum(0))/(1+len(null))).tolist())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args(); assert a.seed in range(5)
    t0 = time.time(); outdir = ROOT/'results'/f'seed{a.seed}'; outdir.mkdir(parents=True, exist_ok=False)
    lock_path = DEFINITIVE/'config/SPROUT_CANONICAL_PROTOCOL.json'
    lock = json.loads(lock_path.read_text())
    for path, expected in lock['file_sha256'].items():
        if Path(path).name.endswith('manifest.json') or path.endswith('.npy') or not Path(path).exists():
            continue  # supplementary package: anonymized manifests and upstream arrays are not verified
        assert sha(path) == expected, f'Canonical SPROUT input changed: {path}'
    d = load('sprout-sixfam', 'gte'); x, r, fam = d['x'], d['r'], d['fam']
    frozen = np.load(DEFINITIVE/'inputs/sprout-sixfam/panel.npz')
    for key in ('query_ids', 'r', 'c', 'fam'): assert np.array_equal(d[key], frozen[key]), key
    approved_path = DEFINITIVE/'inputs/sprout_cells_native_openhermes_v1/cells.npz'
    approved = np.load(approved_path); assert np.array_equal(approved['query_ids'], d['query_ids'])
    cells = approved['cells']; assert len(np.unique(cells)) == 53
    candidate_path = DEFINITIVE/'results/sprout_cell_audit/candidate_cells.npz'
    candidates = np.load(candidate_path); assert np.array_equal(candidates['query_ids'], d['query_ids'])
    split_path = DEFINITIVE/'inputs/sprout-sixfam/splits.json'
    split = json.loads(split_path.read_text())[f'iid_{a.seed}']
    tr, te = np.asarray(split['train']), np.asarray(split['test'])
    assert not np.intersect1d(tr, te).size
    y = np.column_stack([(r >= .5).mean(1), r.mean(1), r,
                         (r >= .25).mean(1), (r >= .75).mean(1)])
    arrays = dict(test_rows=te, query_ids=d['query_ids'][te], family=fam[te])
    specs = [('global', np.zeros(len(r), int), 1), ('family', fam, 1),
             ('approved_cell', cells, 1), ('native_metadata', candidates['native_metadata_cells'], 1),
             ('historical_mixed', candidates['existing_v2_mixed'], 1), ('approved_min50', cells, 50)]
    metrics = {}; retained = {}; refinements = {}
    for name, labels, minimum in specs:
        z = evaluate(x, y, tr, te, labels, fam, threshold=minimum)
        for k, v in z.items(): arrays[name+'__'+k] = v
        metrics[name] = {k:np.nanmean(v, axis=0).tolist() for k,v in z.items() if k.endswith(('_abs','_mse'))}
        retained[name] = z
        print('evaluated', a.seed, name, round(time.time()-t0, 1), flush=True)
    z = evaluate(x, y, tr, te, cells, fam, cutoff=.9)
    for k,v in z.items(): arrays['approved_cos090__'+k] = v
    metrics['approved_cos090'] = {k:np.nanmean(v, axis=0).tolist() for k,v in z.items() if k.endswith(('_abs','_mse'))}
    for k in (2, 4, 8):
        labels, info = refine(x, tr, te, cells, k, 20260923+a.seed)
        name = f'cell_clusters{k}'; z = evaluate(x, y, tr, te, labels, fam)
        for key,v in z.items(): arrays[name+'__'+key] = v
        arrays[name+'__assignments'] = labels
        refinements[name] = info
        metrics[name] = {key:np.nanmean(v, axis=0).tolist() for key,v in z.items() if key.endswith(('_abs','_mse'))}
        if k == 8: fine = labels
        print('refined', a.seed, k, round(time.time()-t0,1), flush=True)
    for k in (10,100):
        nn = retained['global'][f'nn{k}']
        for name, labels in [('family', fam), ('cell', cells), ('cluster8', fine)]:
            arrays[f'global{k}_matched_{name}'] = matched_donor_random(y, tr, te, nn, labels)
    perm = conditional_permutation(y, tr, te, cells, fam, retained['approved_cell']['nn10'], a.seed)
    np.savez_compressed(outdir/'query_metrics.npz', **arrays)
    manifest = dict(seed=a.seed, train_rows=len(tr), test_rows=len(te), proxies=PROXIES,
                    families=d['meta']['families'], approved_cells=53,
                    input_hashes={str(p):sha(p) for p in [approved_path,candidate_path,split_path,
                                      DEFINITIVE/'inputs/sprout-sixfam/panel.npz']},
                    code_sha256=sha(__file__), protocol_sha256=sha(ROOT/'PROTOCOL.md'),
                    canonical_protocol_sha256=sha(lock_path),
                    metrics=metrics, refinement=refinements, permutation=perm,
                    elapsed_seconds=time.time()-t0)
    (outdir/'summary.json').write_text(json.dumps(manifest, indent=2))
    print('DONE', a.seed, time.time()-t0, flush=True)

if __name__ == '__main__': main()
