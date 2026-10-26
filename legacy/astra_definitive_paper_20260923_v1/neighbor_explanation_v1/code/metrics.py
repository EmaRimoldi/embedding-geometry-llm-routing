"""Outcome evaluation only; never used for neighbour selection or tuning."""
import numpy as np

def profiles(r,c,rr,cc):
 return dict(reward_mae=np.abs(rr-r).mean(1),reward_mse=((rr-r)**2).mean(1),
             cost_mae_usd=np.abs(cc-c).mean(1),cost_mse_usd2=((cc-c)**2).mean(1),
             centered_reward_mse=((rr-rr.mean(1,keepdims=True)-r+r.mean(1,keepdims=True))**2).mean(1),
             centered_cost_mse_usd2=((cc-cc.mean(1,keepdims=True)-c+c.mean(1,keepdims=True))**2).mean(1))

def routing(r,c,rr,cc,lam):
 u=r-lam*c;p=rr-lam*cc;ch=p.argmax(1);obs=u[np.arange(len(u)),ch]
 a,b=np.triu_indices(u.shape[1],1);true=u[:,a]-u[:,b];pred=p[:,a]-p[:,b]
 informative=np.abs(true)>1e-12;predtie=np.abs(pred)<=1e-12
 hit=np.where(predtie,.5,(np.sign(true)==np.sign(pred)).astype(float));num=informative.sum(1)
 acc=np.divide((hit*informative).sum(1),num,out=np.full(len(u),np.nan),where=num>0)
 weights=np.where(informative,np.abs(true),0);den=weights.sum(1)
 weighted=np.divide((hit*weights).sum(1),den,out=np.full(len(u),np.nan),where=den>0)
 return dict(utility=obs,regret=u.max(1)-obs,pair_accuracy=acc,margin_weighted_pair_accuracy=weighted),ch

def summarize(arrays,families,population,boot=1000,seed=20260923):
 """Query weighted, fixed-family stratified paired query bootstrap.
 Arrays contain contrasts computed BEFORE resampling, preserving pairing.
 Rows may be a uniform sample within each family; population contains true N_f.
 Undefined pair accuracy is reported conditionally on informative queries.
 """
 names=list(arrays);x=np.column_stack([arrays[k] for k in names]);assert len(x)==len(families)
 weights=np.array([population[int(g)]/np.sum(families==g) for g in families],float)
 finite=np.isfinite(x);tot=(weights[:,None]*finite).sum(0)
 point=np.divide((weights[:,None]*np.nan_to_num(x)).sum(0),tot,out=np.full(len(names),np.nan),where=tot>0)
 rng=np.random.default_rng(seed);numer=np.zeros((boot,len(names)));denom=np.zeros_like(numer)
 for g in np.unique(families):
  ix=np.flatnonzero(families==g);n=len(ix);v=x[ix];valid=np.isfinite(v).astype(float);v=np.nan_to_num(v);pop=population[int(g)]
  for start in range(0,boot,64):
   b=min(64,boot-start);counts=rng.multinomial(n,np.full(n,1/n),size=b).astype(float)
   numer[start:start+b]+=pop/n*(counts@v);denom[start:start+b]+=pop/n*(counts@valid)
 draws=np.divide(numer,denom,out=np.full_like(numer,np.nan),where=denom>0)
 result={}
 for j,name in enumerate(names):
  ok=np.isfinite(draws[:,j]);result[name]=dict(mean=float(point[j]) if tot[j]>0 else None,ci95=np.quantile(draws[ok,j],[.025,.975]).tolist() if ok.any() else [None,None],n_valid=int(finite[:,j].sum()),population_coverage=float(tot[j]/sum(population.values())))
 return result
