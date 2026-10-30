"""Historical absolute-distance metric on the new frozen IID neighbour sets."""
import numpy as np
from common import ROOT, load_data, write_json, sha

def expected_absolute(q, source):
 out=np.empty_like(q,dtype=np.float64)
 for j in range(q.shape[1]):
  s=np.sort(source[:,j]); prefix=np.r_[0.,np.cumsum(s)]; k=np.searchsorted(s,q[:,j],side='right')
  out[:,j]=(q[:,j]*k-prefix[k]+prefix[-1]-prefix[k]-q[:,j]*(len(s)-k))/len(s)
 return out

def main():
 # Check the exact expectation against explicit enumeration, including ties.
 q=np.array([[0.,.3],[1.,.5],[.5,.3]]);s=np.array([[0.,.3],[1.,.3],[.5,.9]])
 np.testing.assert_allclose(expected_absolute(q,s),np.abs(q[:,None]-s[None]).mean(1),atol=1e-14)
 records=[]; provenance=[]
 for ds in ['routerbench','sprout']:
  d,_,original,meta=load_data(ds);r=d['reward']
  y=np.column_stack([(r>=.5).mean(1),r.mean(1),r])
  for seed in range(5):
   folder=ROOT/'results/geometry/gte-qwen2-7b'/ds/'primary'/f'iid_{seed}'
   assert (folder/'complete.json').exists()
   split=np.load(folder/'split.npz');tr=np.searchsorted(original,split['train_rows']);assert np.array_equal(original[tr],split['train_rows'])
   z=np.load(folder/'per_query.npz');assert np.array_equal(z['original_rows'],original)
   for level in ['family','subgroup']:
    observed=[];expected=[];covered=[]
    for key in z.files:
     if not key.startswith(level+'/') or not key.endswith('/neighbors'):continue
     block=key[:-len('/neighbors')];qr=z[block+'/query_rows'];nb=z[key][:,:10]
     if nb.shape[1]<10:continue
     group=block[len(level)+1:];pool=tr[d[level][tr]==group]
     assert len(pool)>=10 and not np.intersect1d(qr,pool).size and np.isin(nb,pool).all()
     obs=np.abs(y[qr,None]-y[nb]).mean(1);base=expected_absolute(y[qr],y[pool])
     observed.append(np.column_stack([obs[:,:2],obs[:,2:].mean(1)]));expected.append(np.column_stack([base[:,:2],base[:,2:].mean(1)]));covered.extend(qr.tolist())
    assert len(set(covered))==len(covered)
    a=np.concatenate(observed);b=np.concatenate(expected)
    for j,target in enumerate(['solve_fraction_050','mean_reward','reward_profile']):
     records.append(dict(dataset=ds,seed=seed,group=level,k=10,target=target,n=len(a),neighbor_absolute_difference=float(a[:,j].mean()),random_absolute_difference=float(b[:,j].mean()),relative_reduction=float(1-a[:,j].mean()/b[:,j].mean())))
   provenance.append(dict(path=str(folder),metrics_sha256=sha(folder/'metrics.json'),panel_sha256=meta['panel_sha256']))
 pooled=[]
 for ds in ['routerbench','sprout']:
  for group in ['family','subgroup']:
   for target in ['solve_fraction_050','mean_reward','reward_profile']:
    rows=[r for r in records if r['dataset']==ds and r['group']==group and r['target']==target];w=[r['n'] for r in rows]
    a=np.average([r['neighbor_absolute_difference'] for r in rows],weights=w);b=np.average([r['random_absolute_difference'] for r in rows],weights=w)
    pooled.append(dict(dataset=ds,group=group,target=target,relative_reduction=float(1-a/b),n_evaluations=sum(w),splits=len(rows)))
 out=ROOT/'results/geometry/_summary'
 write_json(out/'absolute_difficulty_audit.json',dict(records=records,pooled=pooled,provenance=provenance,code_sha256=sha(__file__),notes=['GTE primary rows; five frozen IID splits; identical saved nearest neighbours; absolute difference averaged over queries and reward-profile model dimensions.', 'Random expectation uses all eligible training examples in the same native group. Groups with fewer than ten eligible examples are excluded and coverage is reported.', 'This matches the historical error metric, but does not assert identical row masks, source metadata, or train/test splits. No generated topics are imported.']))
 lines=['# Absolute-distance difficulty audit','','| Dataset | Group | Target | Reduction vs random same-group examples |','|---|---|---|---:|']
 for row in pooled:lines.append(f"| {row['dataset']} | {row['group']} | {row['target']} | {100*row['relative_reduction']:.2f}% |")
 (out/'absolute_difficulty_audit.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
