"""Strict nested family holdout; matched reward predictors and geometry controls."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
ALPHAS = (10., 100., 1000.)
RANKS = (2, 4, 0)  # 0 = full output rank
BETAS = (0., .25, .5, 1.)
COST_BETAS = (0., .5, 1.)
LAMS = {'routerbench-0shot': (0., 1., 10., 50., 100.),
        'llmrouterbench-performance-cost': (0., .5, 1., 2., 5., 10.)}
TASKS = [(p, g, rep) for p, nf in [('routerbench-0shot', 7),
         ('llmrouterbench-performance-cost', 10)] for g in range(nf) for rep in ('gte', 'reader')]

def center(y): return y - y.mean(axis=1, keepdims=True)
def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
def avg(x, w): return np.average(x, axis=0, weights=w)
def key(a, k): return f'{a:g}:{k}'

def ridge_many(x, y, w, xt, alphas=ALPHAS):
    """Full-dimensional weighted ridge; one eigendecomposition for all alphas/targets."""
    x = np.asarray(x, np.float64); xt = np.asarray(xt, np.float64)
    xm = avg(x, w); scale = np.sqrt(avg((x-xm)**2, w)); scale[scale < 1e-8] = 1.
    z = (x-xm)/scale; zt = (xt-xm)/scale; ym = avg(y, w); yc = y-ym
    gram = z.T @ (w[:, None]*z)
    ev, vec = eigh(gram, overwrite_a=True, check_finite=False)
    ev = np.maximum(ev, 0.); rhs = vec.T @ (z.T @ (w[:, None]*yc))
    result = {}
    for a in alphas:
        coef = vec @ (rhs/(ev[:, None]+a))
        result[a] = (zt @ coef + ym, z @ coef + ym)
    return result

def basis(y, w):
    mu = avg(y, w); z = y-mu
    _, v = eigh(z.T @ (w[:, None]*z), check_finite=False)
    return mu, v[:, ::-1]

def project(pred, mu, v, rank):
    if rank == 0: return center(pred)
    b = v[:, :rank]
    return center(mu + (pred-mu) @ b @ b.T)

def neighbors(q, x, k=100, cosine=False):
    if cosine: q, x = unit(q), unit(x)
    q = np.asarray(q, np.float32); x = np.asarray(x, np.float32)
    k = min(k, len(x)); out = np.empty((len(q), k), np.int64)
    nx = (x*x).sum(1)
    for st in range(0, len(q), 256):
        z = q[st:st+256]
        d = (z*z).sum(1)[:, None] + nx[None, :] - 2*z @ x.T
        ix = np.argpartition(d, k-1, axis=1)[:, :k]
        ds = np.take_along_axis(d, ix, axis=1)
        out[st:st+len(z)] = np.take_along_axis(ix, np.argsort(ds, axis=1), axis=1)
    return out

def predict(x, gx, r, c, f, xt, gxt, do_raw_neighbors=True):
    """Only SOURCE outcomes accepted. No target outcome/family label enters any fit."""
    families, fi, counts = np.unique(f, return_inverse=True, return_counts=True)
    w = len(f)/(len(families)*counts[fi])
    rt = np.stack([r[f == s].mean(0) for s in families])
    ct = np.stack([c[f == s].mean(0) for s in families])
    clf = LogisticRegression(C=1., solver='lbfgs', max_iter=500,
        class_weight='balanced', random_state=42).fit(unit(gx), f)
    assert np.array_equal(clf.classes_, families)
    prob = clf.predict_proba(unit(gxt))
    prior, pcost = prob @ rt, prob @ ct
    q = center(r); qr = q-center(rt)[fi]
    # Axes and all targets are fit from source data only.
    logct = np.log(ct+1e-7); cmu = logct.mean(0)
    _, _, cv = np.linalg.svd(logct-cmu, full_matrices=False)
    axis = cv[0]; family_gamma = (logct-cmu) @ axis
    lg = np.log(c+1e-7); gamma = (lg-cmu) @ axis
    cscale = np.maximum(c.mean(0), 1e-8)
    y = np.column_stack([q, qr, lg, c/cscale, gamma])
    pred = ridge_many(x, y, w, xt)
    dm, dv = basis(q, w); rm, rv = basis(qr, w)
    m = r.shape[1]; heads = {'direct': {}, 'residual': {}}
    costs = {}; rawcost = {}
    for a, (yp, yp_train) in pred.items():
        for rank in RANKS:
            heads['direct'][key(a, rank)] = project(yp[:, :m], dm, dv, rank).astype(np.float32)
            heads['residual'][key(a, rank)] = project(yp[:, m:2*m], rm, rv, rank).astype(np.float32)
        span = np.ptp(family_gamma)
        gh = np.clip(yp[:, -1], family_gamma.min()-.25*span, family_gamma.max()+.25*span)
        costs[a] = np.exp(np.clip(cmu+gh[:, None]*axis, -30, 5))
        rawcost[a] = np.maximum(0., yp[:, 3*m:4*m]*cscale)
    yp, yp_train = pred[100.]
    learned_q = project(yp[:, :m], dm, dv, 4)
    learned_s = project(yp_train[:, :m], dm, dv, 4)
    ln = neighbors(learned_q, learned_s)
    result = dict(prior=prior, pcost=pcost, heads=heads, costs=costs,
        rawcost=rawcost, capability=r[ln].mean(1), learned_neighbors=ln,
        family_posterior=prob, family_order=families, source_mean_r=r.mean(0), source_mean_c=c.mean(0))
    if do_raw_neighbors:
        nb = neighbors(xt[:, :-1], x[:, :-1], cosine=True)
        result.update(raw_neighbors=nb, knnr=r[nb].mean(1), knnc=c[nb].mean(1))
    return result

def reward_prediction(z, branch, params):
    if branch == 'direct': return z['heads']['direct'][params['direct']]
    if branch == 'residual': return z['prior']+z['heads']['residual'][params['residual']]
    if branch == 'capability': return z['capability']
    raise KeyError(branch)

def policy(z, branch, beta, cbeta, params, lam):
    rp = z['prior'] if beta == 0 else (1-beta)*z['prior']+beta*reward_prediction(z, branch, params)
    cp = z['pcost'] if cbeta == 0 else (1-cbeta)*z['pcost']+cbeta*z['costs'][params['cost']]
    return (rp-lam*cp).argmax(1)

def tune(inner, outcomes, lambdas):
    # All outcomes here belong to inner validation families, never the outer family.
    losses = {}; params = {}
    for branch in ('direct', 'residual'):
        scores = {}
        for a in ALPHAS:
            for rank in RANKS:
                k = key(a, rank); vals = []
                for h, z in inner.items():
                    r, _ = outcomes[h]
                    yp = z['heads'][branch][k]
                    if branch == 'residual': yp = yp+z['prior']
                    vals.append(float(np.mean((center(yp)-center(r))**2)))
                scores[k] = float(np.mean(vals))
        params[branch] = min(scores, key=scores.get); losses[branch] = scores
    cs = {a: float(np.mean([np.mean((np.log(z['costs'][a]+1e-7)-np.log(outcomes[h][1]+1e-7))**2)
          for h, z in inner.items()])) for a in ALPHAS}
    params['cost'] = min(cs, key=cs.get); losses['cost'] = cs
    rs = {a: float(np.mean([np.mean((z['heads']['direct'][key(a,0)]-center(outcomes[h][0]))**2)
          for h,z in inner.items()])) for a in ALPHAS}
    params['ridge'] = min(rs, key=rs.get)
    chosen = {}; score_tables = {}
    for lam in lambdas:
        choices = [('direct', 0., 0.)]
        choices += [(b, beta, cbeta) for b in ('direct','residual','capability')
                    for beta in BETAS for cbeta in COST_BETAS
                    if beta > 0 or (b == 'direct' and cbeta > 0)]
        vals = {}
        for cand in choices:
            v = []
            for h,z in inner.items():
                r,c = outcomes[h]; ch = policy(z, *cand, params, lam)
                v.append(float((r-lam*c)[np.arange(len(r)), ch].mean()))
            vals[cand] = float(np.mean(v))
        def choose(allowed):
            # Stable tie handling favors less correction and then the first branch.
            return max(allowed, key=lambda a: (vals[a], -a[1]-a[2]))
        chosen[str(lam)] = {
            'selected_reward': choose([a for a in choices if a[2] == 0]),
            'selected_reward_cost': choose(choices),
            'cost_only': choose([a for a in choices if a[1] == 0]),
        }
        for b in ('direct','residual','capability'):
            chosen[str(lam)]['selected_'+b+'_reward'] = choose(
                [a for a in choices if a[2] == 0 and (a[0] == b or a[1] == 0)])
        score_tables[str(lam)] = {str(k):v for k,v in vals.items()}
    return params, chosen, dict(prediction_losses=losses, utility_scores=score_tables)

def geometry(nb, train_r, train_f, target_r):
    qs, qt = center(train_r), center(target_r)
    fams = np.unique(train_f)
    mu = {int(f):qs[train_f==f].mean(0) for f in fams}
    second = {int(f):float((qs[train_f==f]**2).sum(1).mean()) for f in fams}
    out = {}
    for k in (10, 100):
        jj = nb[:, :min(k, nb.shape[1])]; ff = train_f[jj]; p = qs[jj].mean(1)
        expected = np.zeros_like(qt); sq = np.zeros(len(qt))
        shares = {}
        for f in fams:
            wt = (ff==f).mean(1); expected += wt[:,None]*mu[int(f)]
            sq += wt*second[int(f)]; shares[str(f)] = float(wt.mean())
        near_dist = ((qt[:,None,:]-qs[jj])**2).sum(2).mean(1)
        random_dist = (qt**2).sum(1)-2*(qt*expected).sum(1)+sq
        out[str(k)] = dict(centered_prediction_mse=float(((p-qt)**2).mean()),
            same_family_composition_mse=float(((expected-qt)**2).mean()),
            donor_squared_distance=float(near_dist.mean()),
            matched_random_expected_distance=float(random_dist.mean()),
            source_family_shares=shares,
            paired_neighbor_distance_gain_mean=float((random_dist-near_dist).mean()))
    return out

def evaluate(ch, r, c, missing, lam):
    ii = np.arange(len(r)); rr = r[ii,ch]; cc = c[ii,ch]
    return dict(utility=float((rr-lam*cc).mean()), reward=float(rr.mean()),
        cost=float(cc.mean()), missing_choice_rate=float(missing[ii,ch].mean()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', type=int, required=True)
    a = ap.parse_args(); panel, outer, rep = TASKS[a.task]; t0 = time.time()
    inp = ROOT/'inputs'/panel; meta = json.loads((inp/'manifest.json').read_text())
    z = np.load(inp/'panel.npz', allow_pickle=False)
    r,c,f = z['r'].astype(float), z['c'].astype(float), z['fam']; missing=z['missing']
    gx = np.load(inp/'gte.npy', mmap_mode='r'); xx = np.load(inp/f'{rep}.npy', mmap_mode='r')
    ln = np.log1p(z['reader_tokens']).astype(np.float32)[:,None]
    x = np.concatenate([xx,ln], axis=1); del xx
    fs = sorted(set(f.tolist())); sources = [h for h in fs if h != outer]
    assert outer in fs and len(sources) >= 2
    dest = ROOT/'results'/panel/rep/f'fold_{outer:02d}'; dest.mkdir(parents=True, exist_ok=False)
    (dest/'started.json').write_text(json.dumps(dict(panel=panel, outer=outer, representation=rep)))
    inner = {}; truth = {}; split_audit = []
    for h in sources:
        tr = np.flatnonzero((f!=outer)&(f!=h)); va = np.flatnonzero(f==h)
        assert not np.any(f[tr]==outer) and not np.any(f[tr]==h)
        inner[h] = predict(x[tr], gx[tr], r[tr], c[tr], f[tr], x[va], gx[va], False)
        truth[h] = (r[va],c[va])
        split_audit.append(dict(inner_family=int(h), train_families=np.unique(f[tr]).tolist(),
            n_train=len(tr), n_validation=len(va)))
        print(panel, rep, 'outer',outer,'inner',h,'done',round(time.time()-t0),flush=True)
    params, selected, tuning = tune(inner, truth, LAMS[panel])
    # Lock selected settings to disk before loading the outer predictions or evaluating them.
    (dest/'selection.json').write_text(json.dumps(dict(params=params, selected=selected, tuning=tuning),indent=2))
    del inner, truth
    tr=np.flatnonzero(f!=outer); te=np.flatnonzero(f==outer)
    final = predict(x[tr],gx[tr],r[tr],c[tr],f[tr],x[te],gx[te])
    result = dict(panel=panel, representation=rep, heldout_family=meta['families'][outer],
        family_index=outer,n_train=len(tr),n_test=len(te),params=params,selected=selected,
        snapshot_sha256=meta['snapshot_sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        split_audit=split_audit,results={},geometry={},prediction_metrics={})
    saves={'test_rows':z['panel_rows'][te], 'query_ids':z['query_ids'][te]}
    for lam in LAMS[panel]:
        ch={'prior':(final['prior']-lam*final['pcost']).argmax(1),
            'knn100':(final['knnr']-lam*final['knnc']).argmax(1),
            'knn100_reward':(final['knnr']-lam*final['pcost']).argmax(1),
            'ridge_both':(final['heads']['direct'][key(params['ridge'],0)]-lam*final['rawcost'][params['ridge']]).argmax(1),
            'direct_reward':policy(final,'direct',1.,0.,params,lam),
            'best_single_train':np.full(len(te),(r[tr]-lam*c[tr]).mean(0).argmax()),
            'family_hindsight_reference':np.full(len(te),(r[te]-lam*c[te]).mean(0).argmax()),
            'oracle':(r[te]-lam*c[te]).argmax(1)}
        for name, cfg in selected[str(lam)].items(): ch[name]=policy(final,*cfg,params,lam)
        result['results'][str(lam)]={name:evaluate(v,r[te],c[te],missing[te],lam) for name,v in ch.items()}
        for name,v in ch.items(): saves[f'{lam:g}|{name}']=v.astype(np.int16)
    for name,nb in [('cosine',final['raw_neighbors']),('learned_advantage',final['learned_neighbors'])]:
        result['geometry'][name]=geometry(nb,r[tr],f[tr],r[te])
    for name, pr in [('prior',final['prior'])]+[(b,reward_prediction(final,b,params)) for b in ('direct','residual','capability')]:
        err=center(pr)-center(r[te]); m=r.shape[1]
        result['prediction_metrics'][name] = dict(centered_mse=float((err**2).mean()),
            all_pairs_difference_mse=float(2*m/(m-1)*(err**2).mean()))
        saves[f'prediction|{name}']=pr.astype(np.float32)
    saves['prediction|prior_cost']=final['pcost'].astype(np.float32)
    saves['prediction|corrected_cost']=final['costs'][params['cost']].astype(np.float32)
    result['seconds']=time.time()-t0
    np.savez_compressed(dest/'predictions.npz',**saves)
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    print('DONE',panel,rep,outer,result['seconds'],flush=True)

if __name__ == '__main__': main()
