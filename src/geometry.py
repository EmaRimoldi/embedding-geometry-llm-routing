"""Matched in-domain/held-out geometry. Every learned statistic uses training rows."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from sklearn.model_selection import train_test_split
from common import ROOT,CONFIG,load_data,sha,write_json,unit

class Neighbours:
 def __init__(self,x):
  self.x=torch.tensor(x,device='cuda' if torch.cuda.is_available() else 'cpu');torch.backends.cuda.matmul.allow_tf32=False
 def get(self,query,source,k=100):
  query=np.asarray(query);source=np.asarray(source)
  if query.ndim!=1 or source.ndim!=1 or not len(query) or not len(source):
   raise ValueError('Query and source must be nonempty one-dimensional row lists')
  if len(np.unique(source))!=len(source):raise ValueError('Reference rows must be distinct')
  if k<1:raise ValueError('k must be positive')
  # One rectangular result: cap k by the smallest non-self pool across queries.
  available=len(source)-int(np.isin(query,source).any())
  if available<1:raise ValueError('No distinct reference query remains after self exclusion')
  k=min(k,available);s=self.x[source].T;ii=[];cc=[]
  for st in range(0,len(query),128):
   ids=query[st:st+128];sim=self.x[ids]@s
   overlap={int(v):j for j,v in enumerate(source)}
   for a,v in enumerate(ids):
    if int(v) in overlap:sim[a,overlap[int(v)]]=-torch.inf
   vals,idx=torch.topk(sim,k,dim=1,sorted=True)
   assert torch.isfinite(vals).all(),'Insufficient distinct reference queries'
   ii.append(source[idx.cpu().numpy()]);cc.append(vals.cpu().numpy())
  return np.concatenate(ii),np.concatenate(cc)

def mean_ci(v,reps=1000):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 if not len(v):return dict(mean=None,ci95=None,n=0)
 rng=np.random.default_rng(814);boot=[]
 for _ in range(0,reps,20):boot.extend(v[rng.integers(0,len(v),size=(min(20,reps-len(boot)),len(v)))].mean(1))
 return dict(mean=float(v.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(v))

def expected_composition(y,query,source,nb,groups):
 """Expected error for random training rows with the same group counts as neighbours.
 Sampling independently with replacement; query never belongs to source.
 """
 target=y[query];mu=np.zeros_like(target);second=np.zeros_like(target);predvar=np.zeros_like(target);k=nb.shape[1]
 for group in np.unique(groups[nb]):
  pool=y[source[groups[source]==group]];assert len(pool)>0
  avg=pool.mean(0);sec=(pool*pool).mean(0);w=(groups[nb]==group).mean(1)[:,None]
  mu+=w*avg;second+=w*sec;predvar+=w*np.maximum(sec-avg*avg,0)/k
 pair=(target*target-2*target*mu+second).mean(1)
 meanerr=((target-mu)**2+predvar).mean(1)
 return pair,meanerr,mu

def outcomes(d,tr):
 r=d['reward'];c=d['cost'];best=int(r[tr].mean(0).argmax())
 return dict(mean_reward=r.mean(1)[:,None],solve_fraction=(r>=.5).mean(1)[:,None],strict_solve_fraction=(r>=1).mean(1)[:,None],best_source_model_reward=r[:,best,None],relative_reward=r-r.mean(1,keepdims=True),cost=c,relative_cost=c-c.mean(1,keepdims=True))

def pair_accuracy(a,b):
 u,v=np.triu_indices(a.shape[1],1);truth=b[:,u]-b[:,v];guess=a[:,u]-a[:,v];valid=np.abs(truth)>1e-12
 predicted_tie=np.abs(guess)<=1e-12
 score=((np.sign(truth)==np.sign(guess))&valid&~predicted_tie).sum(1)+.5*(predicted_tie&valid).sum(1)
 return np.divide(score,valid.sum(1),out=np.full(len(a),np.nan),where=valid.sum(1)>0),valid.sum(1)

def evaluate(d,tr,te,nb,cos,tag,mode,arrays,records,lambdas):
 fam=d['family'];topics=d['topic'];source_families=np.unique(fam[tr]);ys=outcomes(d,tr)
 assert not np.intersect1d(tr,te).size and np.isin(nb,tr).all()
 arrays[tag+'/query_rows']=te;arrays[tag+'/neighbors']=nb.astype(np.int32);arrays[tag+'/cosine']=cos
 for k in CONFIG['ks']:
  if nb.shape[1]<k:continue
  nn=nb[:,:k];shares=np.stack([(fam[nn]==f).mean(1) for f in source_families],1)
  known=(topics[te]!='');refknown=topics[nn]!='';topic_ok=known[:,None]&refknown
  rec=dict(block=tag,mode=mode,k=k,n_queries=len(te),n_reference=len(tr),same_family_share=float((fam[nn]==fam[te,None]).mean()),random_same_family_share=float(np.mean([(fam[tr]==f).mean() for f in fam[te]])),largest_family_share=float(shares.max(1).mean()),same_format_share=float((d['format'][nn]==d['format'][te,None]).mean()),random_same_format_share=float(np.mean([(d['format'][tr]==f).mean() for f in d['format'][te]])),mean_cosine=float(cos[:,:k].mean()),source_family_shares={str(f):float(shares[:,i].mean()) for i,f in enumerate(source_families)},topic_edge_coverage=float(topic_ok.mean()),topic_query_coverage=float(known.mean()),same_topic_share=float((topics[nn][topic_ok]==np.broadcast_to(topics[te,None],nn.shape)[topic_ok]).mean()) if topic_ok.any() else None,outcomes={})
  if mode=='ood':assert rec['same_family_share']==0
  for name,y in ys.items():
   individual=((y[nn]-y[te,None])**2).mean((1,2));prediction=((y[nn].mean(1)-y[te])**2).mean(1)
   stats=dict(neighbor_pair_mse=float(individual.mean()),neighbor_mean_mse=float(prediction.mean()),controls={})
   arrays[f'{tag}/k{k}/{name}/mean_mse']=prediction.astype(np.float32)
   for group_name,groups in [('family',fam),('native_subgroup',d['subgroup'])]:
    randompair,randommean,_=expected_composition(y,te,tr,nn,groups)
    diff=randompair-individual;gain=randommean-prediction
    c=dict(random_pair_mse=float(randompair.mean()),random_mean_mse=float(randommean.mean()),pair_mse_reduction=float(diff.mean()),prediction_mse_reduction=float(gain.mean()))
    if k in [20,100]:c['pair_reduction_uncertainty']=mean_ci(diff,CONFIG['bootstrap_repetitions'])
    stats['controls'][group_name]=c
    arrays[f'{tag}/k{k}/{name}/{group_name}_pair_reduction']=diff.astype(np.float32)
   rec['outcomes'][name]=stats
  rr=d['reward'][nn].mean(1);cc=d['cost'][nn].mean(1)
  rec['utility_ordering']=[]
  for lam in lambdas:
   acc,valid=pair_accuracy(rr-lam*cc,d['reward'][te]-lam*d['cost'][te])
   rec['utility_ordering'].append(dict(lambda_value=lam,pair_accuracy=float(np.nanmean(acc)) if np.isfinite(acc).any() else None,evaluable_queries=int(np.isfinite(acc).sum()),non_tied_pairs=int(valid.sum())))
   if k==100:arrays[f'{tag}/utility_ordering/lambda{lam:g}']=acc.astype(np.float32)
  records.append(rec)


def linear_probes(x,d,tr,te):
 """All fixed ridge strengths reported. No test-based model/alpha selection."""
 ys=outcomes(d,tr)
 for label in ['format','family']:
  if label=='family' and not set(d[label][te]).issubset(set(d[label][tr])):continue
  categories=np.unique(d[label]);ys[label+'_label']=np.stack([d[label]==v for v in categories],1).astype(float)
 names=list(ys);sizes=[ys[n].shape[1] for n in names];Y=np.concatenate([ys[n] for n in names],1)
 device='cuda' if torch.cuda.is_available() else 'cpu';a=torch.tensor(x[tr],device=device);q=torch.tensor(x[te],device=device)
 mu=a.mean(0);a-=mu;q-=mu;target=Y[tr];ym=target.mean(0);yc=target-ym
 gram=a.T@a; ev,V=torch.linalg.eigh(gram);ev=ev.clamp_min(0);rhs=V.T@(a.T@torch.tensor(yc,dtype=torch.float32,device=device));qv=q@V
 out=[]
 for alpha in CONFIG['ridge_alphas']:
  pred=(qv@(rhs/(ev[:,None]+alpha))).cpu().numpy()+ym;pos=0
  for name,size in zip(names,sizes):
   err=((pred[:,pos:pos+size]-Y[te,pos:pos+size])**2).mean(1);base=((ym[pos:pos+size]-Y[te,pos:pos+size])**2).mean(1)
   out.append(dict(target=name,alpha=alpha,mse=float(err.mean()),source_global_mean_mse=float(base.mean()),relative_mse_reduction=float(1-err.mean()/base.mean()) if base.mean()>0 else None))
   if name.endswith('_label'):
    out[-1]['accuracy']=float((pred[:,pos:pos+size].argmax(1)==Y[te,pos:pos+size].argmax(1)).mean())
   pos+=size
 return out


def main(rep,dataset,subset,fold):
 start=time.time();d,rows,original,meta=load_data(dataset,subset);path=ROOT/'artifacts/embeddings'/rep/dataset
 assert (path/'manifest.json').exists(),'Incomplete extraction'
 raw=np.asarray(np.load(path/'vectors.npy',mmap_mode='r')[original]);x=unit(raw);allrows=np.arange(len(x));f=d['family']
 if fold.startswith('iid_'):
  seed=int(fold[4:]);tr,te=train_test_split(allrows,train_size=.7,stratify=f,random_state=seed);tr=np.sort(tr);te=np.sort(te);mode='iid'
 else:
  families=sorted(set(f));fold_index=int(fold[4:])
  if fold_index>=len(families):
   dest=ROOT/'results/geometry'/rep/dataset/subset/fold
   write_json(dest/'not_applicable.json',dict(reason='No additional held-out family remains after the fixed context-length filter',available_families=families,absent_families=sorted(set(meta['family_counts'])-set(families)),encoder=rep,dataset=dataset,subset=subset,requested_fold=fold,panel_sha256=meta['panel_sha256']))
   print('NOT APPLICABLE',rep,dataset,subset,fold,flush=True);return
  family=families[fold_index];tr=np.flatnonzero(f!=family);te=np.flatnonzero(f==family);mode='ood'
 dest=ROOT/'results/geometry'/rep/dataset/subset/fold;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():print('already complete',dest);return
 np.savez_compressed(dest/'split.npz',train_rows=original[tr],test_rows=original[te],train_keys=d['keys'][tr],test_keys=d['keys'][te])
 ng=Neighbours(x);nb,cos=ng.get(te,tr);arrays={'original_rows':original,'test_rows':te};records=[]
 evaluate(d,tr,te,nb,cos,'unrestricted',mode,arrays,records,CONFIG['lambdas'][dataset])
 if mode=='iid':
  for group_name in ['family','subgroup']:
   for group in sorted(set(d[group_name][te])):
    a=tr[d[group_name][tr]==group];b=te[d[group_name][te]==group]
    if len(a)<5:continue
    ni,co=ng.get(b,a);evaluate(d,a,b,ni,co,group_name+'/'+str(group),mode,arrays,records,CONFIG['lambdas'][dataset])
 # OOD compare source-family concentration with usefulness of each source family's outcomes.
 transfer=[]
 if mode=='ood':
  relative=d['reward']-d['reward'].mean(1,keepdims=True)
  for family in sorted(set(f[tr])):
   a=tr[f[tr]==family];ni,co=ng.get(te,a,min(100,len(a)))
   err=((relative[ni].mean(1)-relative[te])**2).mean(1)
   transfer.append(dict(source_family=family,heldout_family=str(f[te[0]]),source_rows=len(a),mean_nearest_cosine=float(co[:,0].mean()),relative_reward_mse=float(err.mean()),retrieval_share_in_unrestricted_100=float((f[nb]==family).mean())))
   arrays['source_family/'+family+'/relative_reward_mse']=err.astype(np.float32)
 # Norms and direct linear readouts complement neighbourhood measurements.
 extra={}
 if mode=='ood':
  full,fc=ng.get(te,allrows)
  extra['isolated_query']=dict(original_same_family_share=float((f[full[:,:20]]==f[te,None]).mean()),original_mean_nearest_cosine=float(fc[:,0].mean()),removed_mean_nearest_cosine=float(cos[:,0].mean()),mean_nearest_cosine_change=float((cos[:,0]-fc[:,0]).mean()))
  arrays['isolated_query/original_neighbors']=full.astype(np.int32)
 for fmt in np.unique(d['format'][te]):
  a=tr[d['format'][tr]==fmt];b=te[d['format'][te]==fmt]
  if len(a)>=5:
   ni,co=ng.get(b,a);evaluate(d,a,b,ni,co,'same_format/'+str(fmt),mode,arrays,records,CONFIG['lambdas'][dataset])
 probes=linear_probes(x,d,tr,te)
 norms=np.linalg.norm(raw.astype(np.float32),axis=1);norm_correlations={}
 for key,y in outcomes(d,tr).items():
  if y.shape[1]==1:
   v=y[te,0];norm_correlations[key]=float(np.corrcoef(norms[te],v)[0,1]) if np.std(v)>0 and np.std(norms[te])>0 else None
 result=dict(encoder=rep,dataset=dataset,subset=subset,fold=fold,mode=mode,heldout_family=str(f[te[0]]) if mode=='ood' else None,n_train=len(tr),n_test=len(te),records=records,source_family_transfer=transfer,linear_probes=probes,extra_controls=extra,clustering={},norm_correlations=norm_correlations,code_sha256=sha(__file__),config_sha256=sha(ROOT/'configs/geometry.json'),panel_sha256=meta['panel_sha256'],embedding_sha256=json.loads((path/'manifest.json').read_text())['vector_sha256'],slurm_job=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.time()-start)
 np.savez_compressed(dest/'per_query.npz',**arrays);write_json(dest/'metrics.json',result)
 write_json(dest/'complete.json',dict(metrics_sha256=sha(dest/'metrics.json'),per_query_sha256=sha(dest/'per_query.npz')))
 print('COMPLETE',rep,dataset,subset,fold,round(time.time()-start),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);p.add_argument('--dataset',required=True);p.add_argument('--subset',default='primary');p.add_argument('--fold',required=True);a=p.parse_args();main(a.encoder,a.dataset,a.subset,a.fold)
