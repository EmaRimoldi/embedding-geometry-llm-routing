"""RouteLLM-MF under held-out-family and preregistered held-out-cell splits."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
E4=Path(__file__).resolve().parents[2]; ISO=E4/"heldout_generalization_20260920"; sys.path[:0]=[str(E4/"src"),str(ISO/"scripts")]
from rb.data import load_panel
from rb.probes.sweep import LAMBDAS
from rb.reps import find_rep,load_rep
from heldout_cell_classical import labels
THRESHOLDS=np.round(np.linspace(0,1,101),2)
PAIRS={"routerbench-0shot":("gpt-4-1106-preview","mistralai/mistral-7b-chat"),"llmrouterbench-performance-cost":("gpt-5","qwen3-235b-a22b-2507")}
class MF(nn.Module):
 def __init__(self,n_models,text_dim,dim): super().__init__(); self.P=nn.Embedding(n_models,dim); self.text_proj=nn.Linear(text_dim,dim,bias=False); self.classifier=nn.Linear(dim,1,bias=False)
 def forward(self,win,loss,q):
  pw=nn.functional.normalize(self.P(win),p=2,dim=1); pl=nn.functional.normalize(self.P(loss),p=2,dim=1); return self.classifier((pw-pl)*self.text_proj(q)).squeeze(-1)
def train(xt,pairs,epochs,seed,device,dim=128,lr=3e-4,wd=1e-5,batch=64,val_pairs=None,patience=10):
 torch.manual_seed(seed); rng=np.random.default_rng(seed); net=MF(2,xt.shape[1],dim).to(device); opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd); lossf=nn.BCEWithLogitsLoss(); W=torch.as_tensor(pairs[:,1],device=device); L=torch.as_tensor(pairs[:,2],device=device); Q=torch.as_tensor(pairs[:,0],device=device); best=(float("inf"),0); bad=0
 for ep in range(1,epochs+1):
  net.train(); perm=torch.as_tensor(rng.permutation(len(pairs)),device=device)
  for s in range(0,len(pairs),batch):
   b=perm[s:s+batch]; loss=lossf(net(W[b],L[b],xt[Q[b]]),torch.ones(len(b),device=device)); opt.zero_grad(); loss.backward(); opt.step()
  if val_pairs is not None:
   net.eval()
   with torch.no_grad():
    v=torch.as_tensor(val_pairs,device=device); vl=float(lossf(net(v[:,1],v[:,2],xt[v[:,0]]),torch.ones(len(v),device=device)))
   if vl<best[0]: best=(vl,ep); bad=0
   else:
    bad+=1
    if bad>=patience: break
 return net,best
@torch.no_grad()
def winrate(net,xt,rows,device):
 net.eval(); out=[]
 for s in range(0,len(rows),8192):
  rr=torch.as_tensor(rows[s:s+8192],device=device); out.append(torch.sigmoid(net(torch.zeros(len(rr),dtype=torch.long,device=device),torch.ones(len(rr),dtype=torch.long,device=device),xt[rr])).cpu().numpy())
 return np.concatenate(out)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--panel",required=True); ap.add_argument("--manifest-tag",default=""); ap.add_argument("--split",choices=("family","cell"),required=True); ap.add_argument("--index",type=int,required=True); a=ap.parse_args(); assert os.environ.get("E4_PANEL")==a.panel
 p=load_panel(a.panel); fam=p.dataset_index(); spec=None
 if a.split=="family": te=np.flatnonzero(fam==a.index); split_name=str(p.datasets[a.index])
 else:
  _,broad=labels(p,a.panel); spec=json.load(open(ISO/"configs"/f"selected_cells{a.manifest_tag}_{a.panel}.json"))["selected"][a.index]; te=np.flatnonzero((fam==int(spec["family_index"]))&(broad==str(spec["topic"]))); assert len(te)==spec["n"]; split_name=f"{spec['family']}__{spec['topic']}"
 tr=np.setdiff1d(np.arange(p.n),te); models=list(map(str,p.models)); strong,weak=PAIRS[a.panel]; si,wi=models.index(strong),models.index(weak); r=np.nan_to_num(p.rewards.astype(float),nan=0); c=np.nan_to_num(p.costs.astype(float),nan=0); x=load_rep(find_rep(a.panel,"computed/gte-qwen2-7b@prompt"),p.n,p.keep).astype(np.float32)
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); assert device.type=="cuda"; xt=torch.as_tensor(x,device=device); diff=r[tr,si]!=r[tr,wi]; pr=tr[diff]; pairs=np.asarray([[q,0,1] if r[q,si]>r[q,wi] else [q,1,0] for q in pr],dtype=np.int64); assert len(pairs)>=100
 seed=20260920+a.index+(0 if a.split=="family" else 1000); rng=np.random.default_rng(seed); perm=rng.permutation(len(pairs)); nv=max(1,int(.05*len(pairs))); val,trn=pairs[perm[:nv]],pairs[perm[nv:]]; t0=time.time(); _,(vl,ep)=train(xt,trn,100,seed,device,val_pairs=val); assert ep>0; net,_=train(xt,pairs,ep,seed,device); wrtr=winrate(net,xt,tr,device); wrte=winrate(net,xt,te,device); ii=np.arange(len(te)); out={}; thresholds={}
 for lam in LAMBDAS:
  U=r-lam*c; vals=[U[tr][np.arange(len(tr)),np.where(wrtr>=th,si,wi)].mean() for th in THRESHOLDS]; th=float(THRESHOLDS[int(np.argmax(vals))]); ch=np.where(wrte>=th,si,wi); thresholds[f"{lam:g}"]=th; out[f"{lam:g}"]={"utility":float(U[te][ii,ch].mean()),"reward":float(r[te][ii,ch].mean()),"cost":float(c[te][ii,ch].mean()),"choices":ch.tolist()}
 mask=r[te,si]!=r[te,wi]; diag={"device":str(device),"strong":strong,"weak":weak,"train_pairs":len(pairs),"ties_dropped":int((~diff).sum()),"inner_val_pairs":len(val),"best_epoch":ep,"inner_val_bce":vl,"test_pair_accuracy_at_0.5":float(((wrte>=.5)==(r[te,si]>r[te,wi]))[mask].mean()) if mask.any() else None,"seconds":time.time()-t0}
 d=ISO/"results"/a.panel/f"{a.split}{a.manifest_tag}_routellm_mf"; d.mkdir(parents=True,exist_ok=True); rec={**p.provenance(),"split":a.split,"index":a.index,"name":split_name,"selection":spec,"n_train":len(tr),"n_test":len(te),"thresholds":thresholds,"diagnostics":diag,"results":out}; (d/f"{a.index:02d}_{split_name}.json").write_text(json.dumps(rec,indent=2)); print(json.dumps(rec,indent=2),flush=True)
if __name__=="__main__": main()
