"""Held-out test of MMLU topic information about relative model utility."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/baselines'))
from common import load,split_path,sha
PANEL='routerbench-0shot'
OUT=ROOT/'results/geometry/_diagnostics/mmlu_topic_preference'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 d=load(PANEL);s=json.loads(split_path(PANEL).read_text())
 cfg=json.loads((ROOT/'configs/baselines.json').read_text());lams=cfg['lambdas'][PANEL]
 family=d['meta']['families'].index('mmlu');is_mmlu=d['fam']==family
 ids=np.flatnonzero(is_mmlu);topics=np.unique(d['topic'][ids]);m=d['r'].shape[1]
 assert len(topics)==57 and not any(t in ['','unknown','None'] for t in topics)
 splits=[];pred=[];subject=[]
 for i in range(5):
  sp=f'iid_{i}';tr=np.asarray(s[sp]['train']);te=np.asarray(s[sp]['test'])
  tr=tr[is_mmlu[tr]];te=te[is_mmlu[te]];assert not np.intersect1d(tr,te).size
  r=d['r'][tr].mean(0);c=d['c'][tr].mean(0)
  rt=np.empty((len(te),m));ct=np.empty_like(rt)
  for t in topics:
   ti=tr[d['topic'][tr]==t];mask=d['topic'][te]==t;assert len(ti)>0
   rt[mask]=d['r'][ti].mean(0);ct[mask]=d['c'][ti].mean(0)
   for lam in lams:
    u=d['r'][ti]-lam*d['c'][ti];ch=int(u.mean(0).argmax())
    subject.append(dict(split=sp,topic=str(t),lambda_value=lam,n_train=len(ti),chosen_model=d['meta']['models'][ch],family_chosen_model=d['meta']['models'][int((r-lam*c).argmax())]))
  for lam in lams:
   actual=d['r'][te]-lam*d['c'][te];pf=np.broadcast_to(r-lam*c,actual.shape);pt=rt-lam*ct
   actual_rel=actual-actual.mean(1,keepdims=True);pfrel=pf-pf.mean(1,keepdims=True);ptrel=pt-pt.mean(1,keepdims=True)
   cf=pf.argmax(1);ctop=pt.argmax(1);index=np.arange(len(te))
   ef=((pfrel-actual_rel)**2).mean(1);et=((ptrel-actual_rel)**2).mean(1)
   uf=actual[index,cf];ut=actual[index,ctop];disagree=cf!=ctop
   vals=dict(relative_mse_family=float(ef.mean()),relative_mse_topic=float(et.mean()),relative_mse_reduction_fraction=float(1-et.mean()/ef.mean()),utility_family=float(uf.mean()),utility_topic=float(ut.mean()),utility_gain=float((ut-uf).mean()),decision_disagreement_fraction=float(disagree.mean()),test_queries=len(te))
   splits.append(dict(split=sp,lambda_value=lam,**vals));pred.append((lam,te,ef,et,uf,ut,disagree))
 # Repeated test appearances of a query are grouped in the bootstrap; not independent samples.
 pooled=[];save={}
 for lam in lams:
  subset=[x for x in pred if x[0]==lam];rows=np.concatenate([x[1] for x in subset]);unique,ix=np.unique(rows,return_inverse=True);counts=np.bincount(ix)
  values=np.column_stack([np.concatenate([x[j] for x in subset]) for j in range(2,7)])
  sums=np.zeros((len(unique),5));np.add.at(sums,ix,values)
  actual=sums.sum(0)/counts.sum();boot=[];rng=np.random.default_rng(42)
  for k in range(1000):
   draw=rng.integers(len(unique),size=len(unique));a=sums[draw].sum(0)/counts[draw].sum()
   boot.append([a[0]-a[1],a[3]-a[2]])
  ci=np.quantile(boot,[.025,.975],axis=0)
  values_by_split=[x for x in splits if x['lambda_value']==lam]
  rec=dict(lambda_value=lam,relative_mse_family=float(actual[0]),relative_mse_topic=float(actual[1]),relative_mse_reduction_fraction=float(1-actual[1]/actual[0]),mse_improvement_ci95=ci[:,0].tolist(),utility_family=float(actual[2]),utility_topic=float(actual[3]),utility_gain=float(actual[3]-actual[2]),utility_gain_ci95=ci[:,1].tolist(),disagreement_fraction=float(actual[4]),unique_test_queries=len(unique),test_appearances=len(rows),split_utility_gains=[x['utility_gain'] for x in values_by_split],split_mse_reductions=[x['relative_mse_reduction_fraction'] for x in values_by_split])
  pooled.append(rec)
  save[f'{lam}|rows']=rows;save[f'{lam}|metrics']=values
 report=dict(protocol='MMLU-only family-wide versus native subject mean utilities; identical five frozen IID splits and all penalties; predictions fit only on training queries. Row-centering utility across models isolates relative preference from overall query reward/cost level.',uncertainty='Pointwise 95% paired query-cluster bootstrap, 1000 resamples; repeated test appearances resampled together. Conditional on fixed splits and fitted predictors; not simultaneous intervals or fresh-training uncertainty.',n_mmlu=len(ids),n_topics=len(topics),model_names=d['meta']['models'],pooled=pooled,by_split=splits,topic_training_choices=subject,provenance={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'datasets/routerbench/prepared/panel.npz',split_path(PANEL),ROOT/'configs/baselines.json']})
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(OUT/'paired_metrics.npz',**save)
 lines=['# Does MMLU topic predict model preference?','',report['protocol'],'',report['uncertainty'],'','| Lambda | Relative MSE reduction (%) | Utility gain | 95% paired interval | Different routing choices (%) |','|---:|---:|---:|---|---:|']
 for a in pooled:lines.append(f"| {a['lambda_value']} | {100*a['relative_mse_reduction_fraction']:.2f} | {a['utility_gain']:+.5f} | [{a['utility_gain_ci95'][0]:+.5f}, {a['utility_gain_ci95'][1]:+.5f}] | {100*a['disagreement_fraction']:.1f} |")
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
