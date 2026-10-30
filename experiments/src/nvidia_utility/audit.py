"""Reconstruct every selected layer, seed, utility and comparison from saved artifacts."""
import argparse,csv,json,time
import numpy as np
import torch
from common_u import *
from probe import get_folds,utility_targets,ternary
from train import fitpath,fitidentity,predict
def audit_task(i):
    verify();panel,split=task(i);tr,te=rows(panel,split);d=data(panel)
    out=dest(panel,split);sel=selection(panel,split);folds=get_folds(panel,split)
    assert base.sha(out/"folds.npz")==sel["folds_sha256"]
    y=utility_targets(panel,split);cache={};checked=0;start=time.perf_counter()
    for layer in sel["evaluated_layers"]:
        p=out/"probes"/f"layer_{layer}.json";rec=json.loads(p.read_text())
        assert rec["identity"]==dict(**identity(panel,split),layer=layer)
        assert base.sha(p.with_suffix(".npz"))==rec["coefficient_sha256"]
        x=np.asarray(features(panel,split,layer)[tr],np.float64);mses=[]
        with np.load(p.with_suffix(".npz")) as z:
            for j,(a,b) in enumerate(folds):
                co=z["coefficients"][j];inter=z["intercepts"][j]
                xa=x[a];ya=y[a];xm=xa.mean(0);ym=ya.mean(0)
                xc=xa-xm;yc=ya-ym
                # Independent normal-equation check of alpha=1 and intercept.
                lhs=(xc.T@xc+np.eye(100))@co.T;rhs=xc.T@yc
                np.testing.assert_allclose(lhs,rhs,rtol=1e-7,atol=1e-6)
                np.testing.assert_allclose(inter,ym-xm@co.T,rtol=1e-8,atol=1e-8)
                mse=np.mean((x[b]@co.T+inter-y[b])**2,axis=0)
                np.testing.assert_allclose(mse,z["fold_mse"][j],rtol=1e-10,atol=1e-10)
                mses.append(mse);checked+=len(mse)
        cache[layer]=np.mean(mses,axis=0)
        np.testing.assert_allclose(cache[layer],rec["mean_mse"],rtol=1e-10,atol=1e-10)
    m=d["r"].shape[1]
    for li,lam in enumerate(cfg()["lambdas"][panel]):
        for model in range(m):
            trace=[]
            def evaluate(layer):
                trace.append(layer);return -float(cache[layer][li*m+model])
            l,s=ternary(cfg()["layers"],evaluate)
            assert trace==sel["traces"][str(lam)][model]
            assert l==sel["selected"][str(lam)]["layers"][model]
            np.testing.assert_allclose(-s,sel["selected"][str(lam)]["validation_mse"][model])
    result=json.loads((out/"result.json").read_text())
    assert result["identity"]==identity(panel,split)
    assert base.sha(out/"predictions.npz")==result["predictions_sha256"]
    scores=[];network_checks=0
    with np.load(out/"predictions.npz") as ensembles:
        np.testing.assert_array_equal(ensembles["test_rows"],te)
        np.testing.assert_array_equal(ensembles["query_ids"],d["query_ids"][te])
        for lam in cfg()["lambdas"][panel]:
            layers=sel["selected"][str(lam)]["layers"];x,xe=assemble_features(panel,split,layers)
            recs=[];preds=[]
            yy=(d["r"][tr]-lam*d["c"][tr]).astype(np.float32)
            for seed in cfg()["seeds"]:
                p=fitpath(panel,split,lam,seed);r=json.loads(p.read_text())
                assert r["identity"]==fitidentity(panel,split,lam,seed,layers)
                assert base.sha(p.with_suffix(".pt"))==r["checkpoint_sha256"]
                assert base.sha(p.with_suffix(".npz"))==r["predictions_sha256"]
                order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed)).numpy()
                vi=order[int(.85*len(tr)):]
                with np.load(p.with_suffix(".npz")) as z:
                    np.testing.assert_array_equal(z["rows"],te)
                    np.testing.assert_array_equal(z["validation_rows"],tr[vi])
                    mse=float(np.mean((z["validation_prediction"]-yy[vi])**2))
                    np.testing.assert_allclose(mse,r["validation_mse"],rtol=2e-5,atol=1e-7)
                    preds.append(z["prediction"].copy())
                    assert np.isfinite(z["prediction"]).all()
                    # Check saved network versus saved predictions on fixed train/test subsets.
                    net=trunk().SharedTrunkNet(x.shape[1],m)
                    net.load_state_dict(torch.load(p.with_suffix(".pt"),map_location="cpu",weights_only=True));net.eval()
                    ti=np.unique(np.linspace(0,len(te)-1,min(16,len(te)),dtype=int))
                    vj=np.unique(np.linspace(0,len(vi)-1,min(16,len(vi)),dtype=int))
                    np.testing.assert_allclose(predict(net,xe[ti],"cpu"),z["prediction"][ti],rtol=3e-4,atol=3e-5)
                    np.testing.assert_allclose(predict(net,x[vi[vj]],"cpu"),z["validation_prediction"][vj],rtol=3e-4,atol=3e-5)
                    network_checks+=1
                recs.append(r)
            keep=sorted(range(10),key=lambda j:(recs[j]["validation_mse"],cfg()["seeds"][j]))[:5]
            assert [cfg()["seeds"][j] for j in keep]==result["ensembles"][str(lam)]["seeds"]
            pred=np.mean(np.asarray([preds[j] for j in keep]),axis=0);choice=pred.argmax(1)
            np.testing.assert_array_equal(pred,ensembles[f"prediction_{lam:g}"])
            np.testing.assert_array_equal(choice,ensembles[f"choice_{lam:g}"])
            score=base.score(choice,d["r"][te],d["c"][te],d["fam"][te],lam)
            for key in ["utility","reward","cost","equal_family"]:
                np.testing.assert_allclose(score[key],result["scores"][str(lam)][key],rtol=1e-12,atol=1e-12)
            scores.append(dict(panel=panel,split=split,protocol=split.split("_")[0],lambda_value=lam,
                               n_test=len(te),**{k:score[k] for k in ["utility","reward","cost","equal_family"]}))
    atomic_json(out/"audit.json",dict(passed=True,identity=identity(panel,split),probe_fold_target_scores=checked,
        checkpoint_checks=network_checks,utility_scores=len(scores),scores=scores,
        seconds=time.perf_counter()-start,hardware=hardware()))
    print("AUDIT_DONE",panel,split,flush=True)

def report():
    verify();all_scores=[];timing=dict(training_seconds=0.,prediction_and_validation_seconds=0.,feature_assembly_seconds=0.)
    probe_seconds=0.;fits=0;records=[]
    for panel,split in cfg()["tasks"]:
        p=dest(panel,split)/"audit.json";a=json.loads(p.read_text())
        assert a["passed"] and a["identity"]==identity(panel,split);all_scores.extend(a["scores"])
        for f in (dest(panel,split)/"fits").glob("*/*.json"):
            r=json.loads(f.read_text());fits+=1;records.append(r["hardware"])
            for k in timing:timing[k]+=r["timing"][k]
        for f in (dest(panel,split)/"probes").glob("*.json"):
            r=json.loads(f.read_text());probe_seconds+=r["seconds"];records.append(r["hardware"])
    assert len(all_scores)==23*13 and fits==23*13*10
    summary=[]
    for panel in base.PANELS:
        for protocol in ["iid","ood"]:
            rr=[r for r in all_scores if r["panel"]==panel and r["protocol"]==protocol]
            per=[]
            for lam in cfg()["lambdas"][panel]:
                ss=[r for r in rr if r["lambda_value"]==lam];n=sum(r["n_test"] for r in ss)
                per.append(dict(lambda_value=lam,utility=sum(r["utility"]*r["n_test"] for r in ss)/n))
            summary.append(dict(panel=panel,protocol=protocol,mean_utility=float(np.mean([r["utility"] for r in per])),per_penalty=per))
    comparison=[]
    score_path=ROOT/"results/baselines/_summary_reduced/scores.csv"
    with score_path.open() as f: baseline=list(csv.DictReader(f))
    for row in summary:
        for encoder,method in [("reader","kNN-100"),("reader","Family Adaptation"),("reader","MLP"),("gte","kNN-100"),("gte","MLP")]:
            selected=[r for r in baseline if r["dataset"]==row["panel"] and r["protocol"]==row["protocol"] and r["encoder"]==encoder and r["method"]==method]
            assert len(selected)==13 and all(r["complete"]=="True" for r in selected)
            value=float(np.mean([float(r["utility"]) for r in selected]))
            comparison.append(dict(panel=row["panel"],protocol=row["protocol"],encoder=encoder,method=method,
                                   baseline_mean_utility=value,nvidia_mean_utility=row["mean_utility"],
                                   nvidia_minus_baseline=row["mean_utility"]-value))
    # Preserve all penalties and complete comparator scores without selecting by outcome.
    atomic_json(OUT/"summary/report.json",dict(all_complete=True,outer_evaluations=23,utility_scores=299,
        final_fits=fits,summary=summary,comparison=comparison,network_timing_seconds=timing,probe_seconds=probe_seconds,
        hardware_records=records,source_baseline_scores=str(score_path.relative_to(ROOT)),
        source_baseline_scores_sha256=base.sha(score_path),
        notes=["Five-fold source-query validation follows the reference; not source-family validation.",
               "Raw encoder features and outer-training PCA reused; incremental timing excludes their original compute.",
               "Per-target Ridge alpha1/MSE and direct utility ensembles adapt the released correctness pipeline.",
               "All outcomes retained. No win or equivalence conclusion is imposed."]))
    with (OUT/"summary/scores.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(all_scores[0]));w.writeheader();w.writerows(all_scores)
    atomic_json(OUT/"summary/reference_baselines.json",baseline)
    lines=["# NVIDIA per-target utility adaptation","", "All 23 outer evaluations, 299 penalty evaluations and 2,990 final fits audited.","",
           "| Dataset | Protocol | Mean utility across penalties |","|---|---|---|"]
    for r in summary:lines.append(f"| {r['panel']} | {r['protocol']} | {r['mean_utility']:.6f} |")
    lines+=["","| Dataset | Protocol | Encoder / comparator | Comparator mean | NVIDIA minus comparator |",
            "|---|---|---|---|---|"]
    for r in comparison:lines.append(f"| {r['panel']} | {r['protocol']} | {r['encoder']} / {r['method']} | {r['baseline_mean_utility']:.6f} | {r['nvidia_minus_baseline']:+.6f} |")
    lines+=["","Selection uses only outer training data; features are selected per model and penalty.",
            "Network training seconds: "+str(timing["training_seconds"]), "Probe seconds: "+str(probe_seconds),
            "Cached extraction/PCA costs are separate from incremental work. Queue time is not compute time."]
    (OUT/"summary/report.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",type=int);p.add_argument("--report",action="store_true");a=p.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
    if a.report:report()
    else:audit_task(a.task)
