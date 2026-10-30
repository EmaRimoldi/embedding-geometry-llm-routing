"""Additional train-fitted clustering using ordinary k-means."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score

def cluster_recovery(x,d,tr,te):
 result={}
 for label in ['family','topic']:
  for family in ([None] if label=='family' else sorted(set(d['family'][tr]))):
   a=tr if family is None else tr[(d['family'][tr]==family)&(d['topic'][tr]!='')]
   b=te if family is None else te[(d['family'][te]==family)&(d['topic'][te]!='')]
   k=len(np.unique(d[label][a]))
   if k<2 or len(a)<k*2 or len(b)<2:continue
   km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(x[a]);pred=km.predict(x[b]);key=label if family is None else 'topic/'+family
   result[key]=dict(clusters=k,train_rows=len(a),test_rows=len(b),nmi=float(normalized_mutual_info_score(d[label][b],pred)),ari=float(adjusted_rand_score(d[label][b],pred)))
 return result

