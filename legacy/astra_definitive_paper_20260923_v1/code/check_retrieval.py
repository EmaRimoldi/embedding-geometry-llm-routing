"""Independent exact-dot-product checks of a fixed sample of saved predictions."""
import json
import numpy as np
from common import ROOT,load,sha,source
def main():
 rng=np.random.default_rng(20260923);records=[]
 for panel in ['routerbench-0shot','sprout-sixfam']:
  splits=json.loads((ROOT/'inputs'/panel/'splits.json').read_text());manifest=json.loads((ROOT/'inputs'/panel/'source_manifest.json').read_text())
  assert sha(source(panel)/'panel.npz')==json.loads((ROOT/'inputs/audit.json').read_text())[panel]['source_snapshot']
  for rep in ['gte','reader']:
   d=load(panel,rep);x=d['x']
   for file in sorted((ROOT/'results/knn'/panel).glob(f'*/{rep}/predictions.npz')):
    z=np.load(file);split=file.parent.parent.name;tr=np.array(splits[split]['train']);te=z['test_rows'];chosen=rng.choice(len(te),min(5,len(te)),replace=False)
    for j in chosen:
     sims=x[tr]@x[te[j]];order=np.lexsort((np.arange(len(tr)),-sims))
     for k in [5,10,20,40,100,200]:
      nb=tr[order[:min(k,len(tr))]]
      assert np.allclose(d['r'][nb].mean(0),z[f'knn{k}|r'][j],atol=1e-8,rtol=1e-7),(panel,rep,split,k,j,'reward')
      assert np.allclose(d['c'][nb].mean(0),z[f'knn{k}|c'][j],atol=1e-10,rtol=1e-7),(panel,rep,split,k,j,'cost')
    records.append(dict(panel=panel,rep=rep,split=split,queries=len(chosen),k_values=6))
  print('DONE',panel,flush=True)
 (ROOT/'reports/retrieval_audit.json').write_text(json.dumps(dict(records=records,queries=sum(r['queries'] for r in records),seed=20260923,code_sha256=sha(__file__)),indent=2))
if __name__=='__main__':main()
