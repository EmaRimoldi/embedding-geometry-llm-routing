"""Resumable official SharedTrunkNet ensembles on selected per-target utility features."""
import argparse, functools, json, os, signal, subprocess, time
import numpy as np
import torch
from common_u import *
STOP=False
def predict(net,x,device):
    parts=[]
    with torch.inference_mode():
        for i in range(0,len(x),1024):
            parts.append(net(torch.as_tensor(x[i:i+1024],device=device)).cpu().numpy())
    return np.concatenate(parts)
def fitpath(panel,split,lam,seed):
    return dest(panel,split)/"fits"/f"lambda_{lam:g}"/f"seed_{seed}.json"
def fitidentity(panel,split,lam,seed,layers):
    return dict(**identity(panel,split),lambda_value=lam,seed=seed,layers=layers,
                selection_sha256=sha(dest(panel,split)/"selection.json"))
def completed(panel,split,lam,seed,layers):
    p=fitpath(panel,split,lam,seed)
    if not p.exists(): return None
    rec=json.loads(p.read_text())
    assert rec["identity"]==fitidentity(panel,split,lam,seed,layers)
    for suffix,key in [(".npz","predictions_sha256"),(".pt","checkpoint_sha256")]:
        assert sha(p.with_suffix(suffix))==rec[key]
    return rec
def fit(panel,split,lam,seed,layers,x,xe,feature_seconds):
    d=data(panel);tr,te=rows(panel,split)
    y=(d["r"][tr]-lam*d["c"][tr]).astype(np.float32)
    mod=trunk();torch.cuda.synchronize();start=time.perf_counter()
    net=mod.train_mlp(mod.SharedTrunkNet(x.shape[1],y.shape[1]),x,y,device="cuda",seed=seed)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start;phase=time.perf_counter()
    prediction=predict(net,xe,"cuda")
    order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed)).numpy()
    val=order[int(.85*len(tr)):]
    vp=predict(net,x[val],"cuda")
    mse=float(np.mean((vp-y[val])**2))
    np.testing.assert_allclose(mse,net._best_val_loss,rtol=2e-5,atol=1e-7)
    assert np.isfinite(prediction).all() and np.isfinite(vp).all()
    torch.cuda.synchronize();prediction_seconds=time.perf_counter()-phase
    p=fitpath(panel,split,lam,seed);p.parent.mkdir(parents=True,exist_ok=True)
    atomic_npz(p.with_suffix(".npz"),rows=te,prediction=prediction,
               validation_rows=tr[val],validation_prediction=vp)
    temp=p.with_suffix(".tmp.pt");torch.save(net.cpu().state_dict(),temp);temp.replace(p.with_suffix(".pt"))
    rec=dict(identity=fitidentity(panel,split,lam,seed,layers),validation_mse=mse,
        input_dimensions=x.shape[1],predictions_sha256=base.sha(p.with_suffix(".npz")),
        checkpoint_sha256=base.sha(p.with_suffix(".pt")),hardware=hardware(cuda=True),
        timing=dict(training_seconds=train_seconds,prediction_and_validation_seconds=prediction_seconds,
                    feature_assembly_seconds=feature_seconds,total_seconds=time.perf_counter()-start))
    atomic_json(p,rec)
    print("UTILITY_FIT",panel,split,lam,seed,round(train_seconds,2),flush=True)
    del net
def assemble(panel,split):
    d=data(panel);tr,te=rows(panel,split);sel=selection(panel,split)
    arrays=dict(test_rows=te,query_ids=d["query_ids"][te]);scores={};ensembles={}
    for lam in cfg()["lambdas"][panel]:
        layers=sel["selected"][str(lam)]["layers"]
        recs=[completed(panel,split,lam,s,layers) for s in cfg()["seeds"]];assert all(recs)
        keep=sorted(range(10),key=lambda j:(recs[j]["validation_mse"],cfg()["seeds"][j]))[:5]
        preds=[]
        for j in keep:
            with np.load(fitpath(panel,split,lam,cfg()["seeds"][j]).with_suffix(".npz")) as z:
                np.testing.assert_array_equal(z["rows"],te);preds.append(z["prediction"])
        prediction=np.mean(np.asarray(preds),axis=0);choice=prediction.argmax(1)
        arrays[f"prediction_{lam:g}"]=prediction;arrays[f"choice_{lam:g}"]=choice
        scores[str(lam)]=base.score(choice,d["r"][te],d["c"][te],d["fam"][te],lam)
        ensembles[str(lam)]=dict(layers=layers,seeds=[cfg()["seeds"][j] for j in keep],
                                 validation_mse=[r["validation_mse"] for r in recs])
    out=dest(panel,split);atomic_npz(out/"predictions.npz",**arrays)
    atomic_json(out/"result.json",dict(identity=identity(panel,split),n_test=len(te),
        scores=scores,ensembles=ensembles,predictions_sha256=base.sha(out/"predictions.npz")))
    print("UTILITY_SPLIT_DONE",panel,split,flush=True)
def continuation(index):
    job=subprocess.check_output(["sbatch","--parsable","--dependency=afterany:"+os.environ["SLURM_JOB_ID"],
        str(ROOT/"jobs/nvidia_utility_train.sbatch"),str(index)],text=True).strip().split(";")[0]
    atomic_json(OUT/"logs"/f"continuation-{os.environ['SLURM_JOB_ID']}-{index}.json",
                dict(worker=index,parent=os.environ["SLURM_JOB_ID"],job_id=job,submitted_at=time.time()))
def worker(index):
    verify();assert json.loads((OUT/"preflight.json").read_text())["passed"]
    assert torch.cuda.is_available();torch.zeros(1,device="cuda")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","4")))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic()
    for panel,split in cfg()["tasks"][index::2]:
        out=dest(panel,split)
        if (out/"result.json").exists():
            assert json.loads((out/"result.json").read_text())["identity"]==identity(panel,split);continue
        sel=selection(panel,split)
        for lam in cfg()["lambdas"][panel]:
            layers=sel["selected"][str(lam)]["layers"];x=xe=None;feature_seconds=0.
            for seed in cfg()["seeds"]:
                if completed(panel,split,lam,seed,layers):continue
                if STOP or time.monotonic()-start>4.75*3600:
                    continuation(index);return
                if x is None:
                    phase=time.perf_counter();x,xe=assemble_features(panel,split,layers)
                    feature_seconds=time.perf_counter()-phase
                fit(panel,split,lam,seed,layers,x,xe,feature_seconds)
                feature_seconds=0.
            del x,xe
        assemble(panel,split);features.cache_clear()
    finish=OUT/"finished";finish.mkdir(exist_ok=True);(finish/str(index)).write_text(os.environ["SLURM_JOB_ID"])
    if all((finish/str(i)).exists() for i in (0,1)):
        import fcntl
        with (finish/"report.lock").open("w") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if not (finish/"report.json").exists():
                jid=subprocess.check_output(["sbatch","--parsable",str(ROOT/"jobs/nvidia_utility_audit.sbatch")],text=True).strip().split(";")[0]
                atomic_json(finish/"report.json",dict(job_id=jid))
def stop(*args):
    global STOP
    STOP=True
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--worker",type=int,choices=[0,1],required=True)
    a=p.parse_args();signal.signal(signal.SIGUSR1,stop);worker(a.worker)
