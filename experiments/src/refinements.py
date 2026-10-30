"""Outcome-free fine groups fitted on training vectors only."""
import numpy as np
from sklearn.cluster import KMeans

def fine_groups(x,d,tr,te,requested,seed):
 groups=np.full(len(x),-1,np.int32);serial=0;details=[]
 for cell in sorted(set(d['subgroup'])):
  a=tr[d['subgroup'][tr]==cell];b=te[d['subgroup'][te]==cell]
  if not len(a):
   groups[b]=serial;details.append(dict(cell=str(cell),training=0,test=len(b),clusters=0));serial+=1;continue
  k=max(1,min(requested,len(a)//50))
  if k>1:
   km=KMeans(n_clusters=k,n_init=3,max_iter=100,random_state=seed).fit(x[a]);at=km.predict(x[a]);bt=km.predict(x[b]) if len(b) else np.array([],int)
   if np.bincount(at,minlength=k).min()<10:k=1
  if k==1:at=np.zeros(len(a),int);bt=np.zeros(len(b),int)
  groups[a]=serial+at;groups[b]=serial+bt;serial+=k;details.append(dict(cell=str(cell),training=len(a),test=len(b),clusters=k))
 assert (groups[np.r_[tr,te]]>=0).all()
 return groups,details
