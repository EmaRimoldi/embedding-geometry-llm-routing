"""Per-target, per-penalty Ridge selection using the reference ternary search."""
import argparse, ast, json, time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from common_u import *
def ternary(layers, evaluate):
    callback=evaluate; cache={}
    def evaluate(layer):
        if layer not in cache: cache[layer]=callback(layer)
        return cache[layer]
    lo, hi = 0, len(layers)-1
    while hi-lo > 2:
        m1 = lo+(hi-lo)//3
        m2 = hi-(hi-lo)//3
        if evaluate(layers[m1]) < evaluate(layers[m2]): lo = m1+1
        else: hi = m2-1
    best = lo
    score = evaluate(layers[lo])
    for i in range(lo+1, hi+1):
        value = evaluate(layers[i])
        if value > score: score, best = value, i
    return layers[best], score

def get_folds(panel, split):
    tr, te = rows(panel, split)
    assert not np.intersect1d(tr, te).size
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(tr))
    p = dest(panel, split) / "folds.npz"
    arrays = {"outer_train":tr, "outer_test":te}
    for i,(a,b) in enumerate(folds):
        arrays[f"fit_{i}"] = tr[a]; arrays[f"val_{i}"] = tr[b]
    if p.exists():
        with np.load(p) as z:
            for k,v in arrays.items(): np.testing.assert_array_equal(z[k], v)
    else: atomic_npz(p, **arrays)
    return folds

def utility_targets(panel, split):
    d=data(panel); tr,_=rows(panel,split); ls=cfg()["lambdas"][panel]
    # Columns are penalty-major, then the frozen candidate-model order.
    y=np.stack([d["r"][tr].astype(np.float64)-lam*d["c"][tr].astype(np.float64) for lam in ls],axis=1)
    return y.reshape(len(tr),-1)

def layer_scores(panel, split, layer, folds, y):
    p=dest(panel,split)/"probes"/f"layer_{layer}.json"
    ident=dict(**identity(panel,split),layer=layer)
    if p.exists():
        rec=json.loads(p.read_text()); assert rec["identity"]==ident
        assert sha(p.with_suffix(".npz"))==rec["coefficient_sha256"]
        return np.asarray(rec["mean_mse"])
    start=time.perf_counter(); tr,_=rows(panel,split)
    x=np.asarray(features(panel,split,layer)[tr],np.float64)
    coefs=[]; intercepts=[]; scores=[]
    for a,b in folds:
        # Multi-output Ridge shares the factorization, but fits each output independently.
        reg=Ridge(alpha=1.0,fit_intercept=True,solver="cholesky")
        reg.fit(x[a],y[a]); pred=reg.predict(x[b])
        mse=np.mean((pred-y[b])**2,axis=0)
        assert np.isfinite(mse).all()
        coefs.append(reg.coef_);intercepts.append(reg.intercept_);scores.append(mse)
    atomic_npz(p.with_suffix(".npz"),coefficients=np.asarray(coefs),intercepts=np.asarray(intercepts),fold_mse=np.asarray(scores))
    rec=dict(identity=ident,mean_mse=np.mean(scores,axis=0).tolist(),
             coefficient_sha256=base.sha(p.with_suffix(".npz")),
             seconds=time.perf_counter()-start,hardware=hardware(),
             scalar_target_fits=5*y.shape[1],batched_fits=5,
             note="Cached scores for all targets; each ternary search accesses only its own visited layers.")
    atomic_json(p,rec);print("PROBE_LAYER",panel,split,layer,round(rec["seconds"],2),flush=True)
    return np.asarray(rec["mean_mse"])

def run(i):
    verify();panel,split=task(i);d=data(panel);tr,te=rows(panel,split)
    if split.startswith("ood"):
        assert not set(d["fam"][tr]) & set(d["fam"][te])
    out=dest(panel,split);out.mkdir(parents=True,exist_ok=True)
    if (out/"selection.json").exists(): selection(panel,split);return
    folds=get_folds(panel,split);y=utility_targets(panel,split);cache={};values={};traces={}
    start=time.perf_counter();m=d["r"].shape[1]
    for li,lam in enumerate(cfg()["lambdas"][panel]):
        selected=[];mses=[];paths=[]
        for model in range(m):
            visited=[]
            def evaluate(layer):
                if layer not in cache: cache[layer]=layer_scores(panel,split,layer,folds,y)
                visited.append(layer)
                return -float(cache[layer][li*m+model])
            layer,score=ternary(cfg()["layers"],evaluate)
            selected.append(layer);mses.append(-score);paths.append(visited)
        values[str(lam)]=dict(layers=selected,validation_mse=mses)
        traces[str(lam)]=paths
    atomic_json(out/"selection.json",dict(identity=identity(panel,split),selected=values,traces=traces,
        evaluated_layers=sorted(cache),folds_sha256=base.sha(out/"folds.npz"),
        seconds_this_invocation=time.perf_counter()-start,hardware=hardware()))
    print("SELECTION_DONE",panel,split,values,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",type=int,required=True);run(p.parse_args().task)
