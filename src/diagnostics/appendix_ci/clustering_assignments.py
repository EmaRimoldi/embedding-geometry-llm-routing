"""Recover original subject-clustering assignments; never alter old outputs."""
import sys,json,os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
from ci_common import ROOT,OUT,load_panel,write,track
def run(task):
 condition=task//10;dataset=['routerbench','sprout'][(task%10)//5];fold=f'iid_{task%5}'
 rep,subset=[('gte-qwen2-7b','primary'),('gte-qwen2-7b','shared_minilm'),('minilm-l12','shared_minilm')][condition]
 out=OUT/'clustering_assignments'/str(task)
 if (out/'report.json').exists():return
 p=ROOT/'results/geometry'/rep/dataset/subset
 j=json.loads((p/'additional_clustering_kmeans'/fold/'metrics.json').read_text());s=np.load(p/fold/'split.npz')
 d=load_panel(dataset);original=np.flatnonzero(d['minilm_keep']) if subset=='shared_minilm' else np.arange(len(d['reward']))
 tr=np.searchsorted(original,s['train_rows']);te=np.searchsorted(original,s['test_rows'])
 np.testing.assert_array_equal(original[tr],s['train_rows']);np.testing.assert_array_equal(original[te],s['test_rows'])
 x=np.asarray(np.load(ROOT/'artifacts/embeddings'/rep/dataset/'vectors.npy',mmap_mode='r')[original],np.float32)
 x=x/np.linalg.norm(x,axis=1,keepdims=True)
 fam=d['family'][original];topic=d['topic'][original];arrays={};records=[];prov={}
 for name in ['split.npz']:
  track(p/fold/name,prov)
 track(p/'additional_clustering_kmeans'/fold/'metrics.json',prov)
 track(ROOT/'artifacts/embeddings'/rep/dataset/'manifest.json',prov)
 for family in sorted(set(fam[tr])):
  key='topic/'+family
  if key not in j['clustering']:continue
  a=tr[(fam[tr]==family)&(topic[tr]!='')];b=te[(fam[te]==family)&(topic[te]!='')]
  k=len(np.unique(topic[a]));assert k==j['clustering'][key]['clusters']
  km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(x[a]);pred=km.predict(x[b])
  nmi=normalized_mutual_info_score(topic[b],pred);ari=adjusted_rand_score(topic[b],pred)
  assert abs(nmi-j['clustering'][key]['nmi'])<1e-10,(task,family,nmi,j['clustering'][key])
  assert abs(ari-j['clustering'][key]['ari'])<1e-10
  arrays[family+'|rows']=original[b];arrays[family+'|truth']=topic[b];arrays[family+'|pred']=pred
  records.append(dict(family=family,nmi=float(nmi),ari=float(ari),n=len(b),clusters=k))
  print('FIT_MATCH',task,family,nmi,ari,flush=True)
 out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'assignments.npz',**arrays)
 write(out/'report.json',dict(task=task,dataset=dataset,encoder=rep,subset=subset,fold=fold,records=records,original_scores_reproduced=True,source_hashes=prov,slurm_job=os.environ.get('SLURM_JOB_ID')))
if __name__=='__main__':run(int(sys.argv[1]))
