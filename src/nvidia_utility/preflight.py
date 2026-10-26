"""Reference equivalence, input isolation and cached-transform checks before training."""
import ast, json, time
import numpy as np
import torch
from sklearn.linear_model import Ridge
from common_u import *
from probe import ternary, get_folds
def main():
    verify();previous.verify();start=time.perf_counter()
    ref=trunk();assert ref.DEFAULT_TRUNK_HIDDEN==(256,128) and ref.DEFAULT_DROPOUT==(.3,.2)
    assert (ref.DEFAULT_LR,ref.DEFAULT_EPOCHS,ref.DEFAULT_BATCH_SIZE,ref.DEFAULT_PATIENCE,ref.DEFAULT_WEIGHT_DECAY,ref.DEFAULT_TRAIN_VAL_SPLIT)==(.001,150,512,15,.0001,.85)
    # Execute the reference search itself, replacing only its score callback.
    tree=ast.parse((REFERENCE/"sweep.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_ternary_search_layer")
    mod=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(mod)
    env={"PrefillResult":object,"np":np};exec(compile(mod,"reference-search","exec"),env)
    rng=np.random.default_rng(42)
    for values in [np.zeros(14),-np.ones(14),*[-rng.random(14) for _ in range(20)]]:
        layers=cfg()["layers"];seen=[]
        def evaluate(p,l,*args):seen.append(l);return values[layers.index(l)]
        env["_eval_layer"]=evaluate
        expected=env["_ternary_search_layer"](None,layers,"last",100,None,None,5)
        ours=[]
        def score(l):ours.append(l);return values[layers.index(l)]
        assert ternary(layers,score)==expected and ours==seen
    # Verify batched targets have precisely the independent Ridge objective.
    x=rng.normal(size=(200,100));y=rng.normal(size=(200,12))
    a=Ridge(alpha=1.,solver="cholesky").fit(x,y)
    for j in range(y.shape[1]):
        b=Ridge(alpha=1.,solver="cholesky").fit(x,y[:,j])
        np.testing.assert_allclose(a.predict(x)[:,j],b.predict(x),rtol=1e-9,atol=1e-9)
    # Smoke-test unrestricted negative utility and reference checkpoint restoration.
    torch.set_num_threads(2)
    net=ref.train_mlp(ref.SharedTrunkNet(100,3),x.astype(np.float32),y[:,:3].astype(np.float32)-2,
                      device="cpu",seed=42,epochs=4)
    assert np.isfinite(net._best_val_loss)
    inputs={};checked=0
    for panel in base.PANELS:
        d=data(panel);dataset=base.DATASETS[panel]
        for layer in cfg()["layers"]:
            p=previous.features_dir(panel,layer);meta=json.loads((p/"manifest.json").read_text())
            assert meta["rows"]==len(d["r"]) and not meta["chat_template"] and not meta["truncation"]
            assert meta["layer"]==layer and meta["query_sha256"]==sha(ROOT/"datasets"/dataset/"prepared/queries.jsonl")
            assert sha(p/"vectors.npy")==meta["vector_sha256"]
            inputs[str((p/"vectors.npy").relative_to(ROOT))]=meta["vector_sha256"]
    for panel,split in cfg()["tasks"]:
        tr,te=rows(panel,split);d=data(panel)
        assert len(set(tr))==len(tr) and not np.intersect1d(tr,te).size
        if split.startswith("ood"):assert not set(d["fam"][tr]) & set(d["fam"][te])
        folds=get_folds(panel,split)
        assert sorted(np.concatenate([b for _,b in folds]).tolist())==list(range(len(tr)))
        for layer in cfg()["layers"]:
            p=pca_dir(panel,split,layer);x=features(panel,split,layer)
            info=json.loads((p/"ready.json").read_text())
            assert info["identity"]["feature_sha256"]==sha(previous.features_dir(panel,layer)/"vectors.npy")
            with np.load(p/"transform.npz") as z:
                assert z["components"].shape==(100,3584)
                # Independently reconstruct both training and test rows from frozen raw vectors.
                ids=np.asarray([tr[0],tr[-1],te[0],te[-1]])
                raw=np.load(previous.features_dir(panel,layer)/"vectors.npy",mmap_mode="r")
                scaled=np.asarray(raw[ids]).copy()
                scaled-=z["scaler_mean"];scaled/=z["scaler_scale"]
                # sklearn's PCA transform subtracts the projected mean.
                reconstructed=scaled@z["components"].T-z["pca_mean"]@z["components"].T
                np.testing.assert_allclose(reconstructed,x[ids],rtol=3e-4,atol=5e-4)
            for name in ["features.npy","transform.npz","ready.json"]:
                inputs[str((p/name).relative_to(ROOT))]=sha(p/name)
            checked+=1
        features.cache_clear()
        print("PREFLIGHT_SPLIT",panel,split,flush=True)
    atomic_json(OUT/"provenance/cached_inputs.json",inputs)
    atomic_json(OUT/"preflight.json",dict(passed=True,outer_evaluations=len(cfg()["tasks"]),
        cached_transforms=checked,ternary_matches_reference=True,batched_ridge_matches_independent=True,
        source_only=True,seconds=time.perf_counter()-start,hardware=hardware(),config_sha256=sha(CONFIG)))
if __name__=="__main__":main()
