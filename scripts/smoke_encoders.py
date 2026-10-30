#!/usr/bin/env python3
"""Exercise unchanged encoders on the four shipped illustrative prompts; requires one GPU."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model-root",type=Path,default=ROOT/"embedders")
 p.add_argument("--encoders",nargs="+",default=["minilm-l12","gte-qwen2-7b","qwen2.5-7b-instruct"])
 a=p.parse_args()
 base=ROOT/"results/reproduced/reviewer_encoders";base.mkdir(parents=True,exist_ok=True)
 work=Path(tempfile.mkdtemp(prefix="run-",dir=base))
 (work/"src").mkdir();(work/"configs").mkdir()
 for f in ["common.py","encode.py"]:shutil.copy2(ROOT/"src"/f,work/"src"/f)
 shutil.copy2(ROOT/"configs/geometry.json",work/"configs/geometry.json")
 data=work/"datasets/smoke/prepared";data.mkdir(parents=True)
 shutil.copy2(ROOT/"reproducibility/examples/proof_twins.jsonl",data/"queries.jsonl")
 (work/"embedders").symlink_to(a.model_root.resolve(),target_is_directory=True)
 env=os.environ.copy()
 env.update(PYTHONDONTWRITEBYTECODE="1",HF_HOME=str(work/".cache/huggingface"),
   HF_MODULES_CACHE=str(work/".cache/modules"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
   TMPDIR=str(work/".cache/tmp"),PYTHONPYCACHEPREFIX=str(work/".cache/pycache"),
   XDG_CACHE_HOME=str(work/".cache"),TOKENIZERS_PARALLELISM="false")
 Path(env["TMPDIR"]).mkdir(parents=True)
 records=[]
 for enc in a.encoders:
  subprocess.run([sys.executable,"-B",str(work/"src/encode.py"),"--encoder",enc,"--dataset","smoke"],cwd=work,env=env,check=True)
  layers=json.loads((work/"configs/geometry.json").read_text())["qwen_layers"] if enc=="qwen2.5-7b-instruct" else [None]
  for layer in layers:
   out=work/"artifacts/embeddings"/enc
   if layer is not None:out/=f"layer_{layer}"
   out/="smoke"
   m=json.loads((out/"manifest.json").read_text());v=np.load(out/"vectors.npy");done=np.load(out/"done.npy")
   assert v.shape==(4,m["dimension"]) and np.isfinite(v).all() and (np.linalg.norm(v,axis=1)>0).all()
   assert done.all() and m["valid_rows"]==4
   assert sha(out/"vectors.npy")==m["vector_sha256"] and sha(data/"queries.jsonl")==m["query_sha256"]
   assert sha(ROOT/"src/encode.py")==m["code_sha256"]
   records.append(dict(encoder=enc,layer=layer,shape=list(v.shape),finite=True,hashes_match=True,completed_rows=int(done.sum())))
 report=dict(passed=True,scope="Four illustrative prompts, canonical unchanged encode.py; cached pinned weights; fresh dependency installation. No paper performance or model-answer generation.",vectors=records,encoder_code_sha256=sha(ROOT/"src/encode.py"))
 (work/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))
 print("Report:",(work/"report.json").relative_to(ROOT))
if __name__=="__main__":main()
