"""Fixed descriptive projections. No parameter or row selection based on separation."""
import argparse,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,load_data,unit,write_json,sha

def projection(x):
 z=PCA(n_components=min(50,x.shape[1],len(x)-1),random_state=42).fit_transform(x)
 return TSNE(n_components=2,perplexity=min(30,(len(x)-1)/3),init='pca',learning_rate='auto',random_state=42,max_iter=1000).fit_transform(z)
def choose(labels,maximum,rng):
 return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels==v),min(maximum,int((labels==v).sum())),replace=False) for v in np.unique(labels)]))
def main(rep,dataset):
 subset='shared_minilm' if rep=='minilm-l12' else 'primary';d,rows,ix,meta=load_data(dataset,subset)
 x=unit(np.asarray(np.load(ROOT/'artifacts/embeddings'/rep/dataset/'vectors.npy',mmap_mode='r')[ix]));rng=np.random.default_rng(42)
 family='mmlu' if dataset=='routerbench' else 'mmlupro';within=np.flatnonzero((d['family']==family)&(d['topic']!=''))
 topics,counts=np.unique(d['topic'][within],return_counts=True);ordered=sorted(zip(topics,counts),key=lambda v:(-v[1],v[0]));top=[v[0] for v in ordered[:5]]
 subject=top[0];mid=within[np.isin(d['topic'][within],top)];mid=mid[choose(d['topic'][mid],400,rng)]
 last=np.flatnonzero((d['family']==family)&(d['topic']==subject));last=np.sort(rng.choice(last,min(1200,len(last)),replace=False))
 sets=[choose(d['family'],600,rng),mid,last];ys=[d['family'][sets[0]],d['topic'][mid],(d['reward'][last]>=.5).mean(1)]
 dest=ROOT/'results/geometry'/rep/dataset/subset/'figures';dest.mkdir(parents=True,exist_ok=True)
 fig,ax=plt.subplots(1,3,figsize=(12,3.4),layout='constrained');saved={}
 for j,(a,ids,y) in enumerate(zip(ax,sets,ys)):
  z=projection(x[ids]);saved[f'panel{j}_original_rows']=ix[ids];saved[f'panel{j}_xy']=z
  if j<2:
   classes=np.unique(y);cmap=plt.get_cmap('tab10')
   for t,label in enumerate(classes):
    sel=y==label;a.scatter(z[sel,0],z[sel,1],s=5,alpha=.65,color=cmap(t),label=str(label).replace('grade-school-math','GSM8K'))
   a.legend(fontsize=6,markerscale=2,loc='upper center',bbox_to_anchor=(.5,-.02),ncol=2,frameon=False)
  else:
   h=a.scatter(z[:,0],z[:,1],c=y,cmap='viridis',vmin=0,vmax=1,s=8,alpha=.8);fig.colorbar(h,ax=a,shrink=.7,label='Fraction of models scoring ≥ 0.5')
  a.set_xticks([]);a.set_yticks([]);a.spines[['top','right','bottom','left']].set_visible(False)
 ax[0].set_title('Benchmark families',fontsize=11);ax[1].set_title('Subjects within '+family.upper(),fontsize=11);ax[2].set_title('Difficulty within '+subject.replace('-',' '),fontsize=10)
 fig.savefig(dest/'geometry_hierarchy.pdf',bbox_inches='tight');fig.savefig(dest/'geometry_hierarchy.png',dpi=180,bbox_inches='tight');plt.close(fig)
 np.savez_compressed(dest/'geometry_hierarchy_coordinates.npz',**saved)
 write_json(dest/'geometry_hierarchy.json',dict(dataset=dataset,encoder=rep,subset=subset,seed=42,perplexity=30,topic_selection='five largest native subjects by query count',chosen_subject=subject,source_only=False,note='Descriptive full-dataset projection, not an out-of-sample performance estimate; all tests use original vectors.',code_sha256=sha(__file__)))
 print('FIGURE',rep,dataset,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);a=p.parse_args()
 for d in ['routerbench','sprout']:main(a.encoder,d)
