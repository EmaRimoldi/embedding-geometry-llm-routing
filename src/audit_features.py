"""Validate complete, aligned, untruncated feature files before geometry starts."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,load_data,sha,write_json

def main():
 out=[]
 for rep in ['gte-qwen2-7b','qwen2.5-7b-instruct/layer_21','qwen2.5-7b-instruct/layer_23','qwen2.5-7b-instruct/layer_25','minilm-l12']:
  for ds in ['routerbench','sprout']:
   data,rows,ix,meta=load_data(ds);p=ROOT/'artifacts/embeddings'/rep/ds;em=json.loads((p/'manifest.json').read_text());x=np.load(p/'vectors.npy',mmap_mode='r');done=np.load(p/'done.npy');eligible=data['minilm_keep'] if rep=='minilm-l12' else np.ones(len(rows),bool)
   assert x.shape[0]==len(rows) and done[eligible].all() and em['valid_rows']==int(eligible.sum())
   assert em['query_sha256']==sha(ROOT/'datasets'/ds/'prepared/queries.jsonl')
   assert em['vector_sha256']==sha(p/'vectors.npy')
   norms=[]
   for start in range(0,len(rows),512):
    block=np.asarray(x[start:start+512])[eligible[start:start+512]];norm=np.linalg.norm(block.astype(np.float32),axis=1);assert np.isfinite(block).all() and np.isfinite(norm).all() and (norm>0).all();norms.extend(norm.tolist())
   assert not (ROOT/'datasets/sprout/annotations').exists(),'Generated labels must not be reintroduced'
   out.append(dict(encoder=rep,dataset=ds,rows=len(rows),eligible_rows=int(eligible.sum()),dimension=x.shape[1],norm_quantiles=np.quantile(norms,[0,.25,.5,.75,1]).tolist(),passed=True))
 write_json(ROOT/'results/geometry/_run/feature_audit.json',out);print('PASSED',len(out),'feature files',flush=True)
if __name__=='__main__':main()
