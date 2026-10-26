"""Predeclared descriptive projection candidates; preserve all views, never select on outcomes."""
import argparse,json,time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from common import ROOT,load,nearest,sha

def project(x,n=12):
 p=PCA(n_components=min(n,len(x)-1,x.shape[1]),svd_solver='randomized',random_state=20260923)
 return p.fit_transform(x),p

def quality(x,z):
 nn,_=nearest(x,x,20,exclude=np.arange(len(x)))
 dd=((z[:,None,:]-z[None,:,:])**2).sum(2);np.fill_diagonal(dd,np.inf)
 low=np.argsort(dd,axis=1,kind='stable')[:,:20]
 return dict(neighbor_overlap_k20=float(np.mean([len(set(a)&set(b))/20 for a,b in zip(nn,low)])),
  trustworthiness_k20=float(trustworthiness(x,z,n_neighbors=20,metric='cosine')))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',required=True);ap.add_argument('--rep',required=True);a=ap.parse_args()
 d=load(a.panel,a.rep);x=d['x'];f=d['fam'];rng=np.random.default_rng(20260923);dest=ROOT/'results/geometry'/a.panel/a.rep;dest.mkdir(parents=True,exist_ok=False)
 rows=np.concatenate([np.sort(rng.choice(np.flatnonzero(f==g),min(600,int((f==g).sum())),replace=False)) for g in np.unique(f)])
 xx=x[rows];coords,p=project(xx);meanr=d['r'].mean(1);solved=(d['r']>=.5).mean(1);lc=np.log10(np.maximum(d['c'].mean(1),1e-12))
 residual=meanr.copy()
 for g in np.unique(f):residual[f==g]-=meanr[f==g].mean()
 arrays=dict(rows=rows,query_ids=d['query_ids'][rows],fam=f[rows],topic=d['topic'][rows],pca=coords,mean_reward=meanr[rows],solve_fraction=solved[rows],mean_log10_cost=lc[rows],family_residual_reward=residual[rows])
 nn,_=nearest(xx,xx,20,exclude=np.arange(len(xx)))
 pred=KMeans(len(np.unique(f)),n_init=10,random_state=20260923).fit_predict(xx)
 info=dict(panel=a.panel,representation=a.rep,families=d['meta']['families'],sample_rows=len(rows),full_rows=len(x),seed=20260923,
  sample_rule='up to 600 random queries per family; outcome-independent; balanced descriptive sample',
  pca_explained_variance=p.explained_variance_ratio_.tolist(),family_kmeans_nmi=float(normalized_mutual_info_score(f[rows],pred)),
  family_purity_k20=float((f[rows[nn]]==f[rows,None]).mean()),
  same_family_random_chance=float(np.mean([(np.sum(f[rows]==g)-1)/(len(rows)-1) for g in f[rows]])),projections={})
 for axes in [(0,1),(0,2),(1,2)]:info['projections'][f'PC{axes[0]+1}_PC{axes[1]+1}']=quality(xx,coords[:,axes])
 # All fixed t-SNE views; no label fitting and no best-looking seed selection.
 tsne_input=PCA(n_components=min(50,len(xx)-1),svd_solver='randomized',random_state=20260923).fit_transform(xx)
 for perplexity in (30,50):
  z=TSNE(n_components=2,perplexity=perplexity,init='pca',learning_rate='auto',random_state=20260923).fit_transform(tsne_input)
  arrays[f'tsne_{perplexity}']=z;info['projections'][f'tsne_{perplexity}']=quality(xx,z)
 # Matrix ordering uses annotations solely for display.
 matrix_rows=np.concatenate([np.sort(rng.choice(np.flatnonzero(f==g),min(70,int((f==g).sum())),replace=False)) for g in np.unique(f)])
 matrix_rows=matrix_rows[np.lexsort((d['topic'][matrix_rows],f[matrix_rows]))]
 arrays.update(matrix_rows=matrix_rows,matrix_fam=f[matrix_rows],matrix_topic=d['topic'][matrix_rows],matrix_reward=meanr[matrix_rows],cosine=x[matrix_rows]@x[matrix_rows].T)
 info['within_family']={}
 for g in np.unique(f):
  rr=np.sort(rng.choice(np.flatnonzero(f==g),min(900,int((f==g).sum())),replace=False));z,pp=project(x[rr])
  arrays[f'within_{g}_rows']=rr;arrays[f'within_{g}_coords']=z;arrays[f'within_{g}_reward']=meanr[rr];arrays[f'within_{g}_topic']=d['topic'][rr]
  info['within_family'][str(g)]=dict(rows=len(rr),pca_explained_variance=pp.explained_variance_ratio_.tolist(),quality=quality(x[rr],z[:,:2]))
 np.savez_compressed(dest/'projections.npz',**arrays)
 info['code_sha256']=sha(__file__);(dest/'summary.json').write_text(json.dumps(info,indent=2));print('DONE',a.panel,a.rep,flush=True)
if __name__=='__main__':main()
