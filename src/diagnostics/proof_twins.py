"""Approved four-prompt comparison; canonical extraction, no generation or routing."""
from pathlib import Path
import argparse, json, sys, os, time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common import sha,write_json
OUT=ROOT/'results/geometry/_diagnostics/proof_twins_v1'
DATA='proof_twins_v1'
ENCODERS=['gte-qwen2-7b','qwen2.5-7b-instruct']
def verify():
 frozen=json.loads((OUT/'source_hashes.json').read_text())
 for p,h in frozen.items():assert sha(ROOT/p)==h,p
 rows=[json.loads(s) for s in (ROOT/'datasets'/DATA/'prepared/queries.jsonl').read_text().splitlines()]
 assert len(rows)==4 and len({r['id'] for r in rows})==4
 assert [r['pair'] for r in rows]==['pythagoras_fermat']*2+['prime_odd_order']*2
 return rows
def extract(enc):
 verify()
 import encode
 encode.CONFIG=dict(encode.CONFIG,qwen_layers=[21])
 start=time.time()
 encode.main(enc,DATA)
 import torch
 write_json(OUT/(enc+'_timing.json'),dict(elapsed_seconds=time.time()-start,job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),node=os.environ.get('SLURMD_NODENAME')))
def report():
 import numpy as np
 rows=verify();result={};audit={}
 for enc in ENCODERS:
  rep=Path(enc)/'layer_21' if enc==ENCODERS[1] else Path(enc)
  p=ROOT/'artifacts/embeddings'/rep/DATA
  m=json.loads((p/'manifest.json').read_text())
  assert m['query_sha256']==sha(ROOT/'datasets'/DATA/'prepared/queries.jsonl')
  assert m['code_sha256']==sha(ROOT/'src/encode.py')
  assert m['vector_sha256']==sha(p/'vectors.npy')
  assert m['rows']==m['valid_rows']==4 and not m['truncation'] and not m['chat_template']
  x=np.asarray(np.load(p/'vectors.npy'),dtype=np.float64)
  assert x.shape==(4,m['dimension']) and np.isfinite(x).all()
  assert np.load(p/'done.npy').all()
  norms=np.linalg.norm(x,axis=1,keepdims=True);assert (norms>0).all()
  unit=x/norms;cos=unit@unit.T
  assert np.max(np.abs(cos))<=1+1e-12
  values={rows[i]['pair']:float(cos[i,i+1]) for i in (0,2)}
  result[enc]=values
  write_json(ROOT/'results/geometry'/rep/'_diagnostics/proof_twins_v1/results.json',dict(cosines=values,cosine_matrix=cos.tolist(),row_ids=[v['id'] for v in rows],feature_manifest=m))
  audit[enc]=dict(finite=True,rows=4,dimension=m['dimension'],vector_sha256=m['vector_sha256'])
 write_json(OUT/'report.json',dict(prompts=rows,cosines=result,scope='Embedding-only descriptive comparison; no generated answers or routing outcomes. Cross-encoder cosine scales are not calibrated.',audit=audit))
 lines=['# Proof-twin cosine comparison','','| Pair | GTE | Qwen L21 |','|---|---:|---:|']
 for i in (0,2):
  key=rows[i]['pair'];lines.append('| '+key+' | '+format(result[ENCODERS[0]][key],'.6f')+' | '+format(result[ENCODERS[1]][key],'.6f')+' |')
 lines+=['','Canonical raw-prompt encoders; no answer generation or routing. These four selected examples do not establish general difficulty sensitivity or actual output cost. Cosine scales differ across representation spaces.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--extract',choices=ENCODERS);p.add_argument('--report',action='store_true');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 if a.extract:extract(a.extract)
 elif a.report:report()
 else:verify();print('PROOF_TWINS_PREFLIGHT_OK')
