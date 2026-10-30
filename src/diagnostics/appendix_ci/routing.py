"""Reconstruct routing tables and bootstrap fixed predictions, without refitting."""
import sys,json,csv,os
from pathlib import Path
import numpy as np
from ci_common import ROOT,OUT,CFG,write,track,boot_sums,record
sys.path.insert(0,str(ROOT/'src/baselines'))
import common as base
sys.path.insert(0,str(ROOT/'src/layer_sweep'))
import common_layer as layers
def run(task):
 panel=['routerbench-0shot','sprout-sixfam'][task//2];prot=['iid','ood'][task%2]
 dataset=base.DATASETS[panel];dest=OUT/'routing'/panel/prot
 assert not (dest/'report.json').exists(),'Preserve completed output'
 d=base.load(panel);splits=json.loads(base.split_path(panel).read_text());names=[s for s in splits if s.startswith(prot+'_')]
 names.sort(key=lambda s:int(s.split('_')[1]))
 n=len(d['r']);lams=json.loads((ROOT/'configs/baselines.json').read_text())['lambdas'][panel]
 prov={};track(base.split_path(panel),prov);track(ROOT/'datasets'/dataset/'prepared/panel.npz',prov)
 weights=np.zeros(n);tes={}
 for sp in names:
  te=np.asarray(splits[sp]['test']);tr=np.asarray(splits[sp]['train'])
  assert not np.intersect1d(te,tr).size
  tes[sp]=te;weights[te]+=1/(len(names)*len(te)) if prot=='iid' else 1/n
 active=np.flatnonzero(weights>0)
 if prot=='ood':assert len(active)==n and np.allclose(weights,1/n)
 sums=[];columns=[];audit_count=0
 def add(key,sp,te,values,old=None):
  nonlocal audit_count
  assert np.array_equal(te,tes[sp]) and np.isfinite(values).all()
  if old is not None:
   assert abs(values.mean()-old)<1e-10,(key,sp,values.mean(),old)
   audit_count+=1
  if key not in columns:columns.append(key);sums.append(np.zeros(n))
  w=1/(len(names)*len(te)) if prot=='iid' else 1/n
  sums[columns.index(key)][te]+=values*w
 for rep in ['gte','reader']:
  for sp in names:
   te=tes[sp];ii=np.arange(len(te))
   for kind in ['knn','prior','clusters','prox','ridge','mlp_reduced']+(['lookups'] if prot=='iid' else []):
    p=base.destination(kind,panel,sp,rep);j=json.loads((p/'result.json').read_text());z=np.load(p/'predictions.npz')
    track(p/'result.json',prov);track(p/'predictions.npz',prov)
    np.testing.assert_array_equal(te,z['test_rows']);np.testing.assert_array_equal(d['query_ids'][te],z['query_ids'])
    for lam in lams:
     for method,old in j['results'][str(lam)].items():
      if method=='cluster_utility_validated':continue
      ch=z[f'{lam:g}|{method}'];assert ((ch>=0)&(ch<d['r'].shape[1])).all()
      u=(d['r'][te]-lam*d['c'][te])[ii,ch]
      add(f'base|{rep}|{method}|{lam:g}',sp,te,u,old['utility'])
   print('BASE',panel,rep,sp,flush=True)
 # Final authorized NVIDIA per-target utility adaptation.
 for sp in names:
  p=ROOT/'results/nvidia_layer_validation/utility_v1'/dataset/sp
  j=json.loads((p/'result.json').read_text());z=np.load(p/'predictions.npz');te=tes[sp];ii=np.arange(len(te))
  track(p/'result.json',prov);track(p/'predictions.npz',prov)
  np.testing.assert_array_equal(te,z['test_rows']);np.testing.assert_array_equal(d['query_ids'][te],z['query_ids'])
  for lam in lams:
   ch=z[f'choice_{lam:g}'];u=(d['r'][te]-lam*d['c'][te])[ii,ch]
   add(f'nvidia|utility|nvidia|{lam:g}',sp,te,u,j['scores'][str(lam)]['utility'])
 # Layer-table means only; retain every penalty within a query before averaging.
 layer_values=list(range(15,29)) if prot=='ood' else [21,23,25]
 for layer in layer_values:
  for sp in names:
   te=tes[sp];ii=np.arange(len(te))
   for kind,methods in [('knn',['knn100','knn_validated']),('prior',['family_adaptation'])]:
    p=layers.result_dir(panel,layer,sp,kind)
    j=json.loads((p/'result.json').read_text());z=np.load(p/'predictions.npz');track(p/'result.json',prov);track(p/'predictions.npz',prov)
    np.testing.assert_array_equal(te,z['test_rows']);np.testing.assert_array_equal(d['query_ids'][te],z['query_ids'])
    for method in methods:
     v=[]
     for lam in lams:
      u=(d['r'][te]-lam*d['c'][te])[ii,z[f'{lam:g}|{method}']]
      assert abs(u.mean()-j['results'][str(lam)][method]['utility'])<1e-10
      v.append(u)
     add(f'layer|{layer}|{method}|mean',sp,te,np.mean(v,axis=0))
  print('LAYER',panel,layer,flush=True)
 selections={}
 if prot=='ood':
  for sp in names:
   js=[]
   for layer in layer_values:
    p=layers.fold_dir(panel,layer,sp)/'result.json';track(p,prov);js.append(json.loads(p.read_text()))
   te=tes[sp];ii=np.arange(len(te));selections[sp]={}
   for method in ['knn100','knn_validated','family_adaptation']:
    vals=np.array([j['validation_utilities'][method] for j in js])
    layer=layer_values[int(vals.mean(1).argmax())];selections[sp][method]=layer
    p=layers.result_dir(panel,layer,sp,'prior' if method=='family_adaptation' else 'knn')
    z=np.load(p/'predictions.npz')
    v=[(d['r'][te]-lam*d['c'][te])[ii,z[f'{lam:g}|{method}']] for lam in lams]
    add(f'layer|nested|{method}|mean',sp,te,np.mean(v,axis=0))
 matrix=np.column_stack(sums)[active];w=weights[active]
 point=matrix.sum(0)/w.sum()
 boot=boot_sums(matrix,w,d['fam'][active] if prot=='ood' else None)
 rows=[]
 for j,key in enumerate(columns):
  scope,rep,method,lam=key.split('|')
  rows.append(dict(scope=scope,representation=rep,method=method,lambda_value=lam,**record(point[j],boot[:,j])))
 # Mean over the entire grid, same bootstrap draws for every penalty.
 for scope,rep,method in sorted({tuple(c.split('|')[:3]) for c in columns if not c.startswith('layer|')}):
  ix=[columns.index(f'{scope}|{rep}|{method}|{lam:g}') for lam in lams]
  rows.append(dict(scope=scope,representation=rep,method=method,lambda_value='mean',**record(point[ix].mean(),boot[:,ix].mean(1))))
 differences=[]
 for method in ['knn100','knn_validated','family_adaptation','avengerspro_validated','km_prox32_validated','ridge','mlp']:
  a=[columns.index(f'base|reader|{method}|{lam:g}') for lam in lams]
  b=[columns.index(f'base|gte|{method}|{lam:g}') for lam in lams]
  for j,lam in enumerate(lams):
   differences.append(dict(comparison='L21 minus GTE',method=method,lambda_value=f'{lam:g}',**record(point[a[j]]-point[b[j]],boot[:,a[j]]-boot[:,b[j]])))
  differences.append(dict(comparison='L21 minus GTE',method=method,lambda_value='mean',**record((point[a]-point[b]).mean(),(boot[:,a]-boot[:,b]).mean(1))))
 for method in ['knn100','family_adaptation']:
  a=[columns.index(f'base|reader|{method}|{lam:g}') for lam in lams]
  for ref,prefix in [('NVIDIA','nvidia|utility|nvidia'),('Best single model','base|gte|best_single_train')]:
   b=[columns.index(f'{prefix}|{lam:g}') for lam in lams]
   differences.append(dict(comparison='L21 minus '+ref,method=method,lambda_value='mean',**record((point[a]-point[b]).mean(),(boot[:,a]-boot[:,b]).mean(1))))
 dest.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest/'bootstrap.npz',columns=np.array(columns),draws=boot,point=point,query_ids=d['query_ids'][active],weights=w)
 write(dest/'report.json',dict(panel=panel,protocol=prot,complete=True,replicates=CFG['replicates'],audit_score_count=audit_count,rows=rows,differences=differences,selections=selections,source_hashes=prov,slurm_job=os.environ.get('SLURM_JOB_ID')))
 print('COMPLETE',panel,prot,len(rows),audit_count,flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
