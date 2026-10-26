"""exp92: nonlinear heads for the Section-5 recoverable-gap test at lambda=0.

Same outer fit/calibration/test protocol and family-topic residual targets as
exp90/91.  Replaces ridge with either a regularised MLP on the full frozen
representation or histogram gradient boosting on training-only PCA-128.  The
calibration split selects only the fraction allowed to leave the lookup.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(Path(__file__).resolve().parent))
from rb.data import load_panel  # noqa:E402
from rb.panelctx import PANEL,topic_str  # noqa:E402
from rb.probes.sweep import SEEDS,split  # noqa:E402
from common import REP_LABEL,base,load_rep,out_dir,save  # noqa:E402

EXP="exp92_nonlinear_gap_heads";MIN_CELL=50;QGRID=np.array([0,.01,.02,.05,.1,.2,.35,.5,.75,1.]);KLIST=(1,2,3)

def stats(Y,fam,key,F,K):
 g=Y.mean(0);fm=np.vstack([Y[fam==x].mean(0) if (fam==x).any() else g for x in range(F)]);cm=np.full((K,Y.shape[1]),np.nan)
 for k in np.unique(key):
  z=Y[key==k]
  if len(z)>=MIN_CELL:cm[k]=z.mean(0)
 return fm,cm
def apply(fam,key,fm,cm):
 z=cm[key].copy();bad=~np.isfinite(z).all(1);z[bad]=fm[fam[bad]];return z
def fit(method,X,Y,Xo,seed):
 sc=StandardScaler().fit(X);A=sc.transform(X);B=sc.transform(Xo)
 if method=="mlp":
  m=MLPRegressor(hidden_layer_sizes=(256,),alpha=1e-2,early_stopping=True,validation_fraction=.1,n_iter_no_change=12,max_iter=250,random_state=seed)
  return m.fit(A,Y).predict(B).astype(np.float32),{"n_iter":int(m.n_iter_)}
 p=PCA(128,svd_solver="randomized",random_state=seed).fit(A);A=p.transform(A);B=p.transform(B);P=np.zeros((len(B),Y.shape[1]),np.float32);iters=[]
 for j in range(Y.shape[1]):
  m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=20,early_stopping=True,random_state=seed).fit(A,Y[:,j]);P[:,j]=m.predict(B);iters.append(int(m.n_iter_))
 return P,{"pca_components":128,"n_iter":iters}
def rankalt(S,ch):S=S.copy();S[np.arange(len(S)),ch]=-np.inf;return np.argsort(-S,axis=1)
def agg(v,f):return {"per_query":float(v.mean()),"per_family_equal_weight":float(np.mean([v[f==g].mean() for g in np.unique(f)]))}
def boot(d,f,B=2000):
 rng=np.random.default_rng(0);groups=[np.flatnonzero(f==g) for g in np.unique(f)];pq=[];ef=[]
 for _ in range(B):
  ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);pq.append(d[ix].mean());ef.append(np.mean([d[rng.choice(g,len(g),replace=True)].mean() for g in groups]))
 return {"point":float(d.mean()),"ci95":np.percentile(pq,[2.5,97.5]).tolist(),"equal_family":float(np.mean([d[g].mean() for g in groups])),"equal_family_ci95":np.percentile(ef,[2.5,97.5]).tolist()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--method",choices=("mlp","hgb"),required=True);ap.add_argument("--rep",default="gte",choices=("gte","gte_l21","reader_l21"));a=ap.parse_args();t0=time.time()
 panel=load_panel(PANEL);X=load_rep(a.rep,panel);R=np.nan_to_num(panel.rewards.astype(float),nan=0.0);fam=panel.dataset_index().astype(int);_,key=np.unique(np.asarray(topic_str(panel),str),return_inverse=True);F=len(np.unique(fam));K=key.max()+1;M=panel.m
 runs=[];boots={}
 for seed in SEEDS:
  outer,te=split(panel.n,fam,seed);fi,ca=split(len(outer),fam[outer],1000+seed);fitix,cal=outer[fi],outer[ca]
  # Paper lookup uses all outer training rows; head uses fit only.
  fma,cma=stats(R[outer],fam[outer],key[outer],F,K);base_te=apply(fam[te],key[te],fma,cma);choice_base=base_te.argmax(1)
  fm,cm=stats(R[fitix],fam[fitix],key[fitix],F,K);bf=apply(fam[fitix],key[fitix],fm,cm);bc=apply(fam[cal],key[cal],fm,cm);bt=apply(fam[te],key[te],fm,cm)
  P,meta=fit(a.method,X[fitix],R[fitix]-bf,np.vstack([X[cal],X[te]]),seed);nc=len(cal);pc=bc+P[:nc];pt=bt+P[nc:]
  cb_cal=bc.argmax(1);cp_cal=pc.argmax(1);cp_te=pt.argmax(1);adv_c=pc.max(1)-pc[np.arange(nc),cb_cal];adv_t=pt.max(1)-pt[np.arange(len(te)),choice_base]
  order=np.argsort(-adv_c);scores=[]
  for q in QGRID:
   ch=cb_cal.copy();n=int(round(q*nc));ch[order[:n]]=cp_cal[order[:n]];scores.append(R[cal][np.arange(nc),ch].mean())
  q=float(QGRID[int(np.argmax(scores))]);ch=choice_base.copy();n=int(round(q*len(te)));rows=np.argsort(-adv_t)[:n];ch[rows]=cp_te[rows]
  ub=R[te][np.arange(len(te)),choice_base];ud=R[te][np.arange(len(te)),cp_te];uh=R[te][np.arange(len(te)),ch];uo=R[te].max(1);den=uo.mean()-ub.mean()
  rb=rankalt(base_te,choice_base);rp=rankalt(pt,choice_base);short={}
  for k in KLIST:
   uc=np.take_along_axis(R[te],np.column_stack([choice_base,rb[:,:k]]),axis=1).max(1);up=np.take_along_axis(R[te],np.column_stack([choice_base,rp[:,:k]]),axis=1).max(1)
   short[str(k)]={"predicted_fraction_gap":float((up.mean()-ub.mean())/den),"cell_fraction_gap":float((uc.mean()-ub.mean())/den),"excess_over_cell":float(up.mean()-uc.mean())}
  if seed==0:
   boots["calibrated_switch-cell"]=boot(uh-ub,fam[te]);
   for k in KLIST:
    uc=np.take_along_axis(R[te],np.column_stack([choice_base,rb[:,:k]]),axis=1).max(1);up=np.take_along_axis(R[te],np.column_stack([choice_base,rp[:,:k]]),axis=1).max(1);boots[f"shortlist{k}_predicted-cellranked"]=boot(up-uc,fam[te])
  runs.append({"seed":seed,"selected_switch_fraction":q,"cell":agg(ub,fam[te]),"direct":agg(ud,fam[te]),"switch":agg(uh,fam[te]),"oracle":agg(uo,fam[te]),"fraction_gap_recovered":float((uh.mean()-ub.mean())/den),"shortlist":short,"fit_meta":meta});print(f"seed {seed} done ({time.time()-t0:.0f}s)",flush=True)
 def mean(path):
  vals=[]
  for z in runs:
   x=z
   for p in path:x=x[p]
   vals.append(float(x))
  return float(np.mean(vals))
 summary={"cell":mean(["cell","per_query"]),"direct":mean(["direct","per_query"]),"switch":mean(["switch","per_query"]),"oracle":mean(["oracle","per_query"]),"fraction_gap_recovered":mean(["fraction_gap_recovered"]),"switch_fraction":mean(["selected_switch_fraction"]),"shortlist":{str(k):{x:mean(["shortlist",str(k),x]) for x in ("predicted_fraction_gap","cell_fraction_gap","excess_over_cell")} for k in KLIST}}
 result=base(panel,{"experiment":EXP,"method":a.method,"representation":a.rep,"representation_label":REP_LABEL[a.rep],"lambda":0,"protocol":"outer fit/calibration/test; cell residual targets; nonlinear head","runs":runs,"summary":summary,"bootstrap_split0":boots,"seconds":time.time()-t0})
 md=[f"# {EXP} — {a.method} on {REP_LABEL[a.rep]} — {panel.name}","",f"Cell {summary['cell']:.4f}; direct {summary['direct']:.4f}; calibrated switch {summary['switch']:.4f}; oracle {summary['oracle']:.4f}; gap recovered {summary['fraction_gap_recovered']:.1%}; switch fraction {summary['switch_fraction']:.1%}.","","k | predicted fraction gap | cell-ranked fraction gap | utility excess over cell-ranked","---|---:|---:|---:"]
 for k,z in summary["shortlist"].items():md.append(f"{k} | {z['predicted_fraction_gap']:.1%} | {z['cell_fraction_gap']:.1%} | {z['excess_over_cell']:+.4f}")
 md += ["","## Split-0 paired bootstrap","","comparison | per query [95% CI] | equal family [95% CI]","---|---|---"]
 for k,z in boots.items():md.append(f"{k} | {z['point']:+.4f} [{z['ci95'][0]:+.4f}, {z['ci95'][1]:+.4f}] | {z['equal_family']:+.4f} [{z['equal_family_ci95'][0]:+.4f}, {z['equal_family_ci95'][1]:+.4f}]")
 save(out_dir(f"{EXP}_{a.rep}_{a.method}"),result,md)
if __name__=="__main__":main()
