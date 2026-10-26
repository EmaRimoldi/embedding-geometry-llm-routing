"""Matched, nontruncated difficulty audit for MiniLM-L12 and BGE-M3.

New read-only analysis of frozen panels/embeddings. Writes one JSON and the
exact BGE tokenizer keep mask per panel into results/difficulty_geometry/exp63.
No source experiment, panel, embedding, or model file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BIG = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from rb.data import RUNTIME_ROOT, SUCCESS_THRESHOLD, load_panel
import rb.panelctx as panelctx
from rb.probes.constants import RIDGE_ALPHAS
from rb.probes.sweep import SEEDS, split
from rb.reps import find_rep, load_rep


def source(panel, rep, p):
    if rep == "bge-m3":
        name = "routerbench-0shot" if panel.startswith("routerbench") else "performance-cost"
        path = BIG / "experiments/embedder_analysis/features" / name / "prompt/bge-m3.npy"
        x = np.load(path, mmap_mode="r")
        if p.keep is not None and len(x) == len(p.keep):
            x = x[p.keep]
        x = np.asarray(x, dtype=np.float32)
    else:
        path = find_rep(panel, f"computed/{rep}@prompt").path
        x = load_rep(find_rep(panel, f"computed/{rep}@prompt"), p.n, p.keep)
    assert x.shape[0] == p.n and np.isfinite(x).all(), (rep, x.shape, p.n)
    return path, x


def mask_for(panel, rep, p, out):
    if panel.startswith("routerbench"):
        # Both stored sidecars report zero truncation on RouterBench.
        return np.ones(p.n, dtype=bool), "all RouterBench rows fit both token windows"
    if rep == "minilm-l12":
        path = ROOT / "features/llmrouterbench-performance-cost-le512/keep_mask.npz"
        m = np.load(path)["keep"].astype(bool)
        assert len(m) == p.n and m.sum() == 10911
        return m, "saved MiniLM tokenizer mask: <=512 with special tokens"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(RUNTIME_ROOT / "artifacts/models/bge-m3"))
    lengths = np.array([len(tok(str(t), add_special_tokens=True, truncation=False)["input_ids"])
                        for t in p.prompts], dtype=np.int32)
    m = lengths <= 8192
    np.savez_compressed(out / f"{panel}_bge_token_mask.npz", keep=m, lengths=lengths)
    assert len(m) == p.n and (~m).sum() == 347, (m.sum(), (~m).sum())
    return m, "exact BGE-M3 tokenizer mask: <=8192 with special tokens"


def nearest(x, y, cell, fam):
    rows = []
    for seed in SEEDS:
        tr, te = split(len(x), fam, seed)
        for c in np.unique(cell):
            train = tr[cell[tr] == c]
            test = te[cell[te] == c]
            if len(train) < 2 or len(test) == 0:
                continue
            k = min(10, len(train))
            for batch in np.array_split(test, max(1, (len(test)+95)//96)):
                with threadpool_limits(limits=8):
                    sim = x[batch] @ x[train].T
                near = train[np.argpartition(-sim, k-1, axis=1)[:, :k]]
                for i, q in enumerate(batch):
                    ref = np.abs(y[q] - y[train]).mean()
                    d = np.abs(y[q] - y[near[i]]).mean()
                    rows.append((int(q), int(fam[q]), float(d), float(ref)))
    a = np.asarray(rows)
    qid=a[:,0].astype(int)
    counts=np.bincount(qid,minlength=len(x))
    ids=np.flatnonzero(counts)
    near=np.bincount(qid,weights=a[:,2],minlength=len(x))[ids]/counts[ids]
    ref=np.bincount(qid,weights=a[:,3],minlength=len(x))[ids]/counts[ids]
    fi = fam[ids]
    pooled = 1-near.mean()/ref.mean()
    per = {str(f):float(1-near[fi==f].mean()/ref[fi==f].mean()) for f in np.unique(fi)}
    return {"queries":len(ids), "k":10, "relative_gap_reduction":float(pooled),
            "equal_family":float(np.mean(list(per.values()))),"per_family_by_index":per}


def tail_preference(x, y, cell, fam):
    rng = np.random.default_rng(90117)
    scores=[]; families=[]; cells=[]
    for c in np.unique(cell):
        ids=np.flatnonzero(cell==c)
        if len(ids)<60: continue
        lo,hi=np.quantile(y[ids],[.25,.75])
        if lo==hi: continue
        low=ids[y[ids]<=lo]; high=ids[y[ids]>=hi]
        low=low[y[low]<hi];high=high[y[high]>lo]
        if min(len(low),len(high))<15: continue
        for group,other in ((low,high),(high,low)):
            for batch in np.array_split(group,max(1,(len(group)+63)//64)):
                a=rng.choice(group,(len(batch),64)); b=rng.choice(other,(len(batch),64))
                bad=a==batch[:,None]
                while bad.any():
                    a[bad]=rng.choice(group,int(bad.sum()));bad=a==batch[:,None]
                s=np.einsum('ijk,ik->ij',x[a],x[batch],optimize=True)
                t=np.einsum('ijk,ik->ij',x[b],x[batch],optimize=True)
                scores.extend(((s>t)+.5*(s==t)).mean(1).tolist())
                families.extend(fam[batch].tolist());cells.extend([c]*len(batch))
    s=np.asarray(scores);f=np.asarray(families)
    per={str(j):float(s[f==j].mean()) for j in np.unique(f)}
    return {"queries":len(s),"cells":len(set(cells)),"preference":float(s.mean()),
            "equal_family":float(np.mean(list(per.values()))),"per_family_by_index":per}


def ridge(x,y,cell,fam):
    rows=[]
    for seed in SEEDS:
        tr,te=split(len(x),fam,seed)
        base=np.empty(len(te),dtype=float)
        fmean={f:float(y[tr[fam[tr]==f]].mean()) for f in np.unique(fam)}
        for i,q in enumerate(te):base[i]=fmean[fam[q]]
        for c in np.unique(cell[te]):
            z=tr[cell[tr]==c]
            if len(z)>=50:base[cell[te]==c]=float(y[z].mean())
        with threadpool_limits(limits=8):
            sc=StandardScaler().fit(x[tr])
            model=RidgeCV(alphas=RIDGE_ALPHAS).fit(sc.transform(x[tr]),y[tr])
            pred=model.predict(sc.transform(x[te]))
        r2=1-np.sum((y[te]-pred)**2)/np.sum((y[te]-base)**2)
        rows.append(float(r2))
    return {"mean_r2_beyond_cell":float(np.mean(rows)),"per_split":rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--panel",required=True,choices=["routerbench-0shot","llmrouterbench-performance-cost"])
    ap.add_argument("--rep",required=True,choices=["minilm-l12","bge-m3"]);args=ap.parse_args()
    out=ROOT/"results/difficulty_geometry/exp63";out.mkdir(exist_ok=True)
    dest=out/f"{args.panel}_{args.rep}.json"
    if dest.exists():raise FileExistsError(dest)
    p=load_panel(args.panel)
    panelctx.PANEL=args.panel
    panelctx.IS_ROUTERBENCH=args.panel.startswith("routerbench")
    keep,note=mask_for(args.panel,args.rep,p,out)
    if not args.panel.startswith("routerbench"):
        gte_lengths=np.load(ROOT/"features"/args.panel/"prompt/lengths.npz")["gte_tokens"]
        assert len(gte_lengths)==p.n and int((gte_lengths>8192).sum())==337
        keep=keep & (gte_lengths<=8192)
        note += "; intersected with exact GTE tokenizer <=8192"
    ids=np.flatnonzero(keep)
    fam=p.dataset_index()[ids]
    _,cell=np.unique(np.asarray(panelctx.topic_str(p),dtype=str)[ids],return_inverse=True)
    reward=np.nan_to_num(p.rewards,nan=0.0)[ids]
    targets={"solve_fraction":(reward>=SUCCESS_THRESHOLD).mean(1),"mean_reward":reward.mean(1)}
    gpath,g=source(args.panel,"gte-qwen2-7b",p)
    cpath,v=source(args.panel,args.rep,p)
    g=g[ids];v=v[ids]
    record={"panel":args.panel,"rep":args.rep,"rows":len(ids),"removed":int((~keep).sum()),"mask":note,
      "gte_path":str(gpath),"gte_sha256":hashlib.sha256(gpath.read_bytes()).hexdigest(),
      "candidate_path":str(cpath),"candidate_sha256":hashlib.sha256(cpath.read_bytes()).hexdigest(),
      "seeds":list(SEEDS),"cell_count":int(len(np.unique(cell))),"results":{}}
    for name,x in (("gte",g),(args.rep,v)):
        unit=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        record["results"][name]={}
        for target,y in targets.items():
            print(args.panel,name,target,"nearest",flush=True)
            loc=nearest(unit,y,cell,fam)
            print(args.panel,name,target,"tails",flush=True)
            tail=tail_preference(unit,y,cell,fam)
            print(args.panel,name,target,"ridge",flush=True)
            lin=ridge(x,y,cell,fam)
            record["results"][name][target]={"knn10":loc,"tails":tail,"ridge":lin}
            print(name,target,loc["relative_gap_reduction"],tail["preference"],lin["mean_r2_beyond_cell"],flush=True)
    # Same-encoder comparison: freeze the intermediate layer choice on
    # RouterBench, validate it on LLMRouterBench, and hold pooling fixed.
    layers=("L7_mean","L12_mean") if args.rep=="minilm-l12" else ("L17_cls","L24_cls")
    record["matched_internal"]={"intermediate_selected_on_routerbench":layers[0],
        "final_same_pooling":layers[1],"results":{}}
    for layer in layers:
        path=ROOT/"features"/args.panel/"prompt/hidden"/args.rep/(layer+".npy")
        raw=np.load(path,mmap_mode="r")
        if p.keep is not None and len(raw)==len(p.keep):raw=raw[p.keep]
        raw=np.asarray(raw[ids],dtype=np.float32)
        assert raw.shape==v.shape and np.isfinite(raw).all()
        unit=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
        record["matched_internal"]["results"][layer]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"targets":{}}
        for target,y in targets.items():
            print(args.panel,args.rep,layer,target,"tail + ridge",flush=True)
            record["matched_internal"]["results"][layer]["targets"][target]={
                "tails":tail_preference(unit,y,cell,fam),"ridge_raw":ridge(raw,y,cell,fam),
                "ridge_unit":ridge(unit,y,cell,fam)}
    dest.write_text(json.dumps(record,indent=2))
    print("SAVED",dest,flush=True)

if __name__=="__main__":main()
