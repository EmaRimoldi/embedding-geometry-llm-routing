"""Isolated six-family SPROUT reproduction of exp07 raw/z=0 and exp25 gates.

Frozen outcomes/cohort from astra_sprout_l21_20260922_v1. No target labels enter
fits, clipping, lambda-specific gating or hyperparameter selection.
"""
import os, json, hashlib, argparse, time, warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
E4 = Path(os.environ['E4_ROOT'])
SRC = E4 / 'astra_sprout_l21_20260922_v1'
LAMS = (0., .5, 1., 2., 5., 10., 25., 50., 100., 250., 500., 1000.)
BETAS = (0., .25, .5, .75, 1.)
ALPHAS = np.logspace(-1, 5, 13)
EPS = 1e-7

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tables(r, c, fam, sources):
    return (np.stack([r[fam == s].mean(0) for s in sources]),
            np.stack([c[fam == s].mean(0) for s in sources]))

def swap_losses(rt, ct, lam):
    u = rt - lam * ct
    lc, lr = [], []
    for f in range(len(rt)):
        for g in range(len(rt)):
            if f == g: continue
            lc.append(u[f].max() - u[f, (rt[f] - lam * ct[g]).argmax()])
            lr.append(u[f].max() - u[f, (rt[g] - lam * ct[f]).argmax()])
    return float(np.mean(lc)), float(np.mean(lr))

def main(fold):
    t0 = time.time()
    warnings.filterwarnings('error', category=ConvergenceWarning)
    meta = json.loads((SRC/'inputs/manifest.json').read_text())
    canary = json.loads((SRC/'canary.json').read_text())
    assert canary['pass'] and canary['snapshot_sha256'] == meta['snapshot_sha256']
    z = np.load(SRC/'inputs/panel.npz')
    r, c, fam = z['r'].astype(float), z['c'].astype(float), z['fam']
    assert len(r) == 43887 and len(np.unique(fam)) == 6
    assert np.isfinite(r).all() and np.isfinite(c).all() and (c >= 0).all()
    assert [int((fam == i).sum()) for i in range(6)] == [540,9884,11786,748,19340,1589]
    raw = {}
    for rep in ('gte','reader'):
        ref = meta['representations'][rep]; p = Path(ref['path']); st = p.stat()
        assert st.st_size == ref['bytes'] and st.st_mtime_ns == ref['mtime_ns']
        ix = z['gte_rows'] if rep == 'gte' else z['panel_rows']
        raw[rep] = np.asarray(np.load(p, mmap_mode='r')[ix], np.float32)
        assert np.isfinite(raw[rep]).all()
    norm = np.linalg.norm(raw['gte'], axis=1, keepdims=True)
    assert np.isfinite(norm).all() and (norm > 0).all()
    x = raw['gte'] / norm
    ln = np.log1p(z['reader_tokens'])[:,None].astype(np.float32)
    features = {rep:np.hstack([v, ln]) for rep,v in raw.items()}
    features['length'] = ln
    del raw
    y = np.log(c + EPS)
    sources = [s for s in range(6) if s != fold]
    te = np.flatnonzero(fam == fold)
    outdir = ROOT/'results'; outdir.mkdir(exist_ok=True)
    assert not (outdir/f'fold{fold}.json').exists(), 'Never overwrite a finished fold'

    def build(S, target):
        tr = np.flatnonzero(np.isin(fam,S))
        assert not np.intersect1d(tr,target).size
        assert fold not in fam[tr]
        rt,ct = tables(r,c,fam,S)
        clf = LogisticRegression(C=1., class_weight='balanced', max_iter=500,
                                 solver='lbfgs', random_state=42).fit(x[tr],fam[tr])
        assert np.array_equal(clf.classes_, S)
        post = clf.predict_proba(x[target])
        logct = np.log(ct + EPS); alpha = logct.mean(0)
        U,s,V = np.linalg.svd(logct-alpha,full_matrices=False)
        b = V[0]*s[0]/np.sqrt(len(S)); G = U[:,0]*np.sqrt(len(S))
        assert b@b > 1e-12
        targetcoord = (y[tr]-alpha)@b/(b@b)
        counts = {s:int((fam[tr]==s).sum()) for s in S}
        weights = np.array([len(tr)/(len(S)*counts[s]) for s in fam[tr]])
        lo,hi = G.min(),G.max(); span=hi-lo
        result = dict(pr=post@rt, pc=post@ct, fits={})
        for rep,X in features.items():
            sc = StandardScaler().fit(X[tr])
            model = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]),targetcoord,
                                            sample_weight=weights)
            coord = model.predict(sc.transform(X[target]))
            coord = np.clip(coord,lo-.25*span,hi+.25*span)
            pred = np.exp(alpha[None,:]+coord[:,None]*b[None,:])
            assert np.isfinite(pred).all() and (pred>0).all()
            result[rep] = pred
            result['fits'][rep] = {'alpha':float(model.alpha_), 'clip':[float(lo-.25*span),float(hi+.25*span)]}
        print('fit sources',S,'target',len(target),'seconds',round(time.time()-t0),flush=True)
        return result

    inner = {s:build([q for q in sources if q!=s],np.flatnonzero(fam==s)) for s in sources}
    outer = build(sources,te)
    rt,ct = tables(r,c,fam,sources)
    result = dict(fold=fold,family=meta['families'][fold],n=len(te),rows=len(r),
                  lambdas=LAMS,betas=BETAS,cohort_sha=meta['snapshot_sha256'],
                  input_sha=sha(SRC/'inputs/panel.npz'),code_sha=sha(__file__),
                  outer_fits=outer['fits'],inner_fits={str(s):v['fits'] for s,v in inner.items()},
                  scores={},selections={},diagnostics={})
    choices={}
    def choose(block,rep,lam,beta):
        cost=block['pc'] if beta==0 else (1-beta)*block['pc']+beta*block[rep]
        return (block['pr']-lam*cost).argmax(1)
    for lam in LAMS:
        L=str(lam); u=r[te]-lam*c[te]; ii=np.arange(len(te))
        lc,lr=swap_losses(rt,ct,lam); bd=float(lc>lr)
        result['diagnostics'][L]={'wrong_cost':lc,'wrong_reward':lr,'beta':bd}
        ch={'soft_prior':choose(outer,'reader',lam,0),
            'best_single':np.full(len(te),(r[fam!=fold]-lam*c[fam!=fold]).mean(0).argmax())}
        result['selections'][L]={}
        for rep in features:
            scores={}
            for beta in BETAS:
                vals=[]
                for s in sources:
                    ix=np.flatnonzero(fam==s);us=r[ix]-lam*c[ix]
                    vals.append(float(us[np.arange(len(ix)),choose(inner[s],rep,lam,beta)].mean()))
                scores[beta]=[float(np.mean(vals)),float(np.min(vals))]
            bm=max(BETAS,key=lambda b:scores[b][0])
            bw=max(BETAS,key=lambda b:(scores[b][1],scores[b][0]))
            result['selections'][L][rep]={'mean':bm,'worst':bw,'inner_scores':scores}
            for method,beta in [('fixed',1.),('gated_mean',bm),('gated_worst',bw),('gated_diag',bd)]:
                ch[f'{rep}_{method}']=choose(outer,rep,lam,beta)
        result['scores'][L]={}
        for method,sel in ch.items():
            if lam==0 and method!='best_single':assert np.array_equal(sel,ch['soft_prior'])
            result['scores'][L][method]={'utility':float(u[ii,sel].mean()),
                'reward':float(r[te][ii,sel].mean()),'cost':float(c[te][ii,sel].mean())}
            choices[f'{L}_{method}']=sel
    np.savez_compressed(outdir/f'fold{fold}_predictions.npz',test_rows=te,query_ids=z['query_ids'][te],
                        predicted_reward=outer['pr'],prior_cost=outer['pc'],
                        **{f'cost_{k}':outer[k] for k in features},**choices)
    result['elapsed_seconds']=time.time()-t0
    temp=outdir/f'fold{fold}.json.tmp';temp.write_text(json.dumps(result,indent=2));temp.rename(outdir/f'fold{fold}.json')
    print('DONE',fold,result['elapsed_seconds'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True)
    args=ap.parse_args();assert 0<=args.fold<6;main(args.fold)
