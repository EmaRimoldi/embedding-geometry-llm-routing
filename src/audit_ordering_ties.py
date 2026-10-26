"""Audit/correct numerical predicted ties from saved retrieval; no experiment redesign."""
import json,io,os,zipfile,shutil
import numpy as np
from common import ROOT,load_data,write_json,sha
from geometry import pair_accuracy

def main():
 tasks=json.loads((ROOT/'configs/geometry_tasks.json').read_text());changes=[];updated=0;cache={}
 for t in tasks:
  p=ROOT/'results/geometry'/t['encoder']/t['dataset']/t['subset']/t['fold']
  if not (p/'complete.json').exists():continue
  m=json.loads((p/'metrics.json').read_text());ds=(t['dataset'],t['subset'])
  if ds not in cache:cache[ds]=load_data(*ds)[0]
  d=cache[ds];z=np.load(p/'per_query.npz');arr={};blocks={};affected=[]
  for rec in m['records']:
   block=rec['block'];k=rec['k']
   if block not in blocks:blocks[block]=(z[block+'/query_rows'],z[block+'/neighbors'])
   q,nn=blocks[block];nn=nn[:,:k];rr=d['reward'][nn].mean(1);cc=d['cost'][nn].mean(1)
   for row in rec['utility_ordering']:
    lam=row['lambda_value'];acc,n=pair_accuracy(rr-lam*cc,d['reward'][q]-lam*d['cost'][q]);value=float(np.nanmean(acc)) if np.isfinite(acc).any() else None
    previous=row['pair_accuracy']
    if previous!=value:affected.append(dict(block=block,k=k,lambda_value=lam,old=previous,new=value))
    row['pair_accuracy']=value
    if k==100:arr[block+f'/utility_ordering/lambda{lam:g}.npy']=acc.astype(np.float32)
  z.close()
  if not affected:continue
  old=json.loads((p/'complete.json').read_text());tmp=p/'per_query.tie_corrected.npz'
  with zipfile.ZipFile(p/'per_query.npz') as source,zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as target:
   for item in source.infolist():
    if item.filename in arr:
     buffer=io.BytesIO();np.save(buffer,arr.pop(item.filename));target.writestr(item.filename,buffer.getvalue())
    else:
     with source.open(item) as src,target.open(item.filename,'w',force_zip64=True) as dst:shutil.copyfileobj(src,dst,1024*1024)
  assert not arr
  os.replace(tmp,p/'per_query.npz');m['numerical_tie_correction']=dict(rule='Predicted absolute utility difference <= 1e-12 receives half credit only; observed ties excluded',code_sha256=sha(__file__),previous_artifact_hashes=old)
  write_json(p/'metrics.json',m);write_json(p/'complete.json',dict(metrics_sha256=sha(p/'metrics.json'),per_query_sha256=sha(p/'per_query.npz')));changes.append(dict(**t,affected=affected));updated+=1
  print('corrected',t,'entries',len(affected),flush=True)
 write_json(ROOT/'results/geometry/_summary/ordering_tie_audit.json',dict(updated_runs=updated,changes=changes,rule='Predicted ties get half credit, never both sign credit and tie credit.'))
 print('AUDIT COMPLETE',updated,flush=True)
if __name__=='__main__':main()
