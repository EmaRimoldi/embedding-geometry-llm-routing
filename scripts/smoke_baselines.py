#!/usr/bin/env python3
"""Run actual baseline CLIs on explicitly synthetic IID/OOD fixtures, without paper-data changes."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=ROOT/"results/reproduced/smoke")
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix="baseline-",dir=a.output))
    shutil.copytree(ROOT/"src",work/"src",ignore=shutil.ignore_patterns("__pycache__"))
    (work/"configs/baseline_splits").mkdir(parents=True)
    for name in ["baselines.json","baseline_tasks.json","ridge.json"]:
        shutil.copy2(ROOT/"configs"/name,work/"configs"/name)
    rng=np.random.RandomState(20261006);n=420;m=4;dim=12
    families=np.repeat(np.array(["a","b","c","d","e","f","g"]),60)
    x=rng.normal(size=(n,dim)).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
    rewards=rng.uniform(size=(n,m));costs=rng.uniform(.0001,.015,size=(n,m))
    keys=np.array(["smoke-"+str(i) for i in range(n)])
    # Fixed deterministic fixture only, distinct from every paper cohort.
    for panel,ds in [("routerbench-0shot","routerbench"),("sprout-sixfam","sprout")]:
        out=work/"datasets"/ds/"prepared";out.mkdir(parents=True)
        (out/"queries.jsonl").write_text("".join(json.dumps({"key":str(k),"prompt":"Synthetic smoke query "+str(i)})+"\n" for i,k in enumerate(keys)))
        np.savez_compressed(out/"panel.npz",reward=rewards,cost=costs,family=families,
            topic=np.repeat("topic",n),subgroup=families,format=np.repeat("choice",n),keys=keys,minilm_keep=np.ones(n,bool))
        save(out/"manifest.json",{"models":["model"+str(i) for i in range(m)],"rows":n})
        train,test=train_test_split(np.arange(n),train_size=.7,stratify=families,random_state=0)
        fit,val=train_test_split(train,test_size=.2,stratify=families[train],random_state=10000)
        tr=np.flatnonzero(families!="a");te=np.flatnonzero(families=="a")
        splits={"iid_0":{"train":sorted(train.tolist()),"test":sorted(test.tolist()),"inner":[{"fit":sorted(fit.tolist()),"val":sorted(val.tolist())}]},
            "ood_0":{"train":tr.tolist(),"test":te.tolist(),"held_out_family":"a","inner":[{"fit":tr[families[tr]!=f].tolist(),"val":tr[families[tr]==f].tolist()} for f in sorted(set(families[tr]))]}}
        save(work/"configs/baseline_splits"/(panel+".json"),splits)
        for rep in ["gte-qwen2-7b","qwen2.5-7b-instruct/layer_21"]:
            ep=work/"artifacts/embeddings"/rep/ds;ep.mkdir(parents=True)
            np.save(ep/"vectors.npy",x)
            save(ep/"manifest.json",{"rows":n,"query_sha256":sha(out/"queries.jsonl"),"vector_sha256":sha(ep/"vectors.npy")})
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",XDG_CACHE_HOME=str(work/".cache"),MPLCONFIGDIR=str(work/".cache/matplotlib"))
    def run(name,args):
        r=subprocess.run([sys.executable,"-B"]+args,cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (work/(name+".log")).write_text(r.stdout)
        if r.returncode: raise RuntimeError(name+" failed: "+r.stdout[-2000:])
    run("mlp_algebra_preflight",["src/baselines/check.py"])
    run("ridge_preflight",["src/baselines/run_ridge.py","--check"])
    for task,split in [(0,"iid_0"),(10,"ood_0")]:
        for kind in ["knn","prior"]:
            run(split+"_"+kind,["src/baselines/run_fixed.py","--task",str(task),"--kind",kind])
    run("ridge_iid",["src/baselines/run_ridge.py","--pack","0"])
    run("ridge_ood",["src/baselines/run_ridge.py","--pack","5"])
    cfg=json.loads((work/"configs/baselines.json").read_text());lams=cfg["lambdas"]["routerbench-0shot"]
    splits=json.loads((work/"configs/baseline_splits/routerbench-0shot.json").read_text())
    checks=0
    for split in ["iid_0","ood_0"]:
        s=splits[split];tr=np.array(s["train"]);te=np.array(s["test"])
        assert not np.intersect1d(tr,te).size
        if split.startswith("ood"):assert not set(families[tr])&set(families[te])
        kn=work/"results/baselines/gte-qwen2-7b/routerbench"/split/"knn"
        kz=np.load(kn/"predictions.npz");kr=json.loads((kn/"result.json").read_text())
        # Independent full similarity sort and neighbor means.
        neighbors=tr[np.argsort(-(x[te]@x[tr].T),axis=1,kind="stable")[:,:100]]
        np.testing.assert_allclose(kz["knn100|r"],rewards[neighbors].mean(1),atol=1e-12)
        np.testing.assert_allclose(kz["knn100|c"],costs[neighbors].mean(1),atol=1e-12)
        for kind in ["knn","prior","ridge"]:
            p=work/"results/baselines/gte-qwen2-7b/routerbench"/split/kind
            rec=json.loads((p/"result.json").read_text());z=np.load(p/"predictions.npz")
            np.testing.assert_array_equal(z["test_rows"],te)
            for lam in lams:
                for method,score in rec["results"][str(lam)].items():
                    ch=z[f"{lam:g}|{method}"]
                    utility=float(np.mean(rewards[te,ch]-lam*costs[te,ch]))
                    np.testing.assert_allclose(score["utility"],utility,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(score["utility"],score["reward"]-lam*score["cost"],rtol=1e-12,atol=1e-12)
                    if method=="best_single_train":
                        np.testing.assert_array_equal(ch,np.full(len(te),np.argmax((rewards[tr]-lam*costs[tr]).mean(0))))
                    if method=="knn100":
                        np.testing.assert_array_equal(ch,(kz["knn100|r"]-lam*kz["knn100|c"]).argmax(1))
                    checks+=1
        assert set(kr["selections"])==set(map(str,lams))
        vals=np.zeros((len(lams),len(cfg["knn_candidates"])));vn=0
        for fold in s["inner"]:
            fi=np.array(fold["fit"]);va=np.array(fold["val"]);assert set(fi)<=set(tr) and set(va)<=set(tr)
            nb=fi[np.argsort(-(x[va]@x[fi].T),axis=1,kind="stable")[:,:max(cfg["knn_candidates"])]]
            for il,lam in enumerate(lams):
                for ik,k in enumerate(cfg["knn_candidates"]):
                    rr=rewards[nb[:,:min(k,len(fi))]].mean(1);cc=costs[nb[:,:min(k,len(fi))]].mean(1)
                    choice=(rr-lam*cc).argmax(1);vals[il,ik]+=(rewards[va,choice]-lam*costs[va,choice]).sum()
            vn+=len(va)
        for il,lam in enumerate(lams):
            np.testing.assert_allclose(kr["selections"][str(lam)]["candidate_utilities"],vals[il]/vn,rtol=1e-12,atol=1e-12)
            assert kr["selections"][str(lam)]["k"]==cfg["knn_candidates"][int(np.argmax(vals[il]))]
    result={"passed":True,"scope":"Synthetic fixture; real kNN/prior/Ridge CLI fit-predict-save paths, all prescribed RouterBench penalties, IID/OOD; not paper performance.",
            "score_checks":checks,"mlp_sklearn_equivalence_cases":16,"ridge_direct_utility_checks":36,
            "workspace":str(work.relative_to(ROOT)) if work.is_relative_to(ROOT) else str(work)}
    save(work/"report.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
