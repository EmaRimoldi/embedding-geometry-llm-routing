"""Select (architecture, alpha, early_stopping) per penalty on inner source validation and score the test choices.

For each (panel, outer split, representation) with all 6 configurations complete, writes
results/mlp_tuned/<panel>/<split>/<rep>/result.json in the definitive study's format:
  results[str(lambda)]['mlp_tuned']          selected over every configuration (the new baseline)
  results[str(lambda)]['mlp_published']      alpha=1e-4, no early stopping, architecture selected (old protocol;
                                             should track astra_definitive .../results/mlp_gpu)
  results[str(lambda)]['best_single_train']  reference
and results/mlp_tuned/SUMMARY.md: IID = mean over the five splits, OOD = pooled over held-out families
(query weighted), alongside the stored old MLP where it exists.
"""
import json
from collections import defaultdict
import numpy as np
from common_mlp import DEF,PANELS,PROTOCOL,ROOT,evaluate,load,split_names,splits

def main():
 cfgs=[(ia,es) for ia in range(len(PROTOCOL['architectures'])) for es in PROTOCOL['early_stopping']];alphas=PROTOCOL['alphas']
 table=defaultdict(dict);md=['# Tuned MLP baseline\n',f"alphas {alphas}; architectures {PROTOCOL['architectures']}; early stopping {PROTOCOL['early_stopping']}\n"]
 for panel in PANELS:
  lams=PROTOCOL['lambdas'][panel];S=splits(panel);cache={}
  for rep in PROTOCOL['representations']:
   for sp in split_names(panel):
    runs={}
    for ia,es in cfgs:
     f=ROOT/'results/runs'/panel/sp/rep/f'arch{ia}_es{int(es)}'/'result.json'
     if f.exists():runs[(ia,es)]=(json.loads(f.read_text()),np.load(f.with_name('choices.npz'))['choice'])
    if len(runs)<len(cfgs):continue
    if panel not in cache:cache[panel]=load(panel,None)
    d=cache[panel];r,c,fam,miss=d['r'],d['c'],d['fam'],d['missing'];te=np.array(S[sp]['test']);tr=np.array(S[sp]['train'])
    out={};sel={}
    for il,lam in enumerate(lams):
     cand=[(ia,es,ja) for ia,es in cfgs for ja in range(len(alphas))]
     crit=lambda k:runs[k[:2]][0]['inner_sum'][il][k[2]]/runs[k[:2]][0]['inner_n']
     best=max(cand,key=crit);pub=max([k for k in cand if k[1] is False and k[2]==0],key=crit)
     ch=lambda k:runs[k[:2]][1][:,il,k[2]].astype(int)
     single=np.full(len(te),int(np.argmax((r[tr]-lam*c[tr]).mean(0))))
     out[str(lam)]=dict(mlp_tuned=evaluate(ch(best),r[te],c[te],fam[te],miss[te],lam),mlp_published=evaluate(ch(pub),r[te],c[te],fam[te],miss[te],lam),
                        best_single_train=evaluate(single,r[te],c[te],fam[te],miss[te],lam))
     sel[str(lam)]=dict(architecture=PROTOCOL['architectures'][best[0]],early_stopping=best[1],alpha=alphas[best[2]],inner_utility=crit(best),
                        published_architecture=PROTOCOL['architectures'][pub[0]])
    dest=ROOT/'results/mlp_tuned'/panel/sp/rep;dest.mkdir(parents=True,exist_ok=True)
    (dest/'result.json').write_text(json.dumps(dict(panel=panel,split=sp,representation=rep,n_test=len(te),results=out,selections=sel,
      sources=[str(ROOT/'results/runs'/panel/sp/rep/f'arch{ia}_es{int(es)}') for ia,es in cfgs]),indent=1))
    table[(panel,rep)][sp]=(len(te),out,sel)
  # summary
  for rep in PROTOCOL['representations']:
   T=table.get((panel,rep),{});md.append(f'\n## {panel} · {rep}\n')
   for kind in ('iid','ood'):
    names=[s for s in split_names(panel) if s.startswith(kind)]
    if not all(s in T for s in names):md.append(f'- {kind}: {sum(s in T for s in names)}/{len(names)} splits complete');continue
    def agg(m):
     if kind=='iid':return [np.mean([T[s][1][str(l)][m]['utility'] for s in names]) for l in lams]
     n=sum(T[s][0] for s in names);return [sum(T[s][1][str(l)][m]['utility']*T[s][0] for s in names)/n for l in lams]
    rows={'tuned MLP':agg('mlp_tuned'),'published-default MLP (rerun)':agg('mlp_published'),'best single (train)':agg('best_single_train')}
    old=DEF/'results/mlp_gpu'/panel
    if rep=='gte' and old.exists():
     try:
      vals=[]
      for l in lams:
       per=[(json.loads((old/s/'gte'/f'lambda_{l:g}'/'result.json').read_text()),T[s][0]) for s in names]
       u=[p[0]['results'][str(l) if str(l) in p[0]['results'] else f'{l:g}']['mlp_utility_validated']['utility'] for p in per]
       vals.append(np.mean(u) if kind=='iid' else sum(x*p[1] for x,p in zip(u,per))/sum(p[1] for p in per))
      rows['stored MLP (definitive run)']=vals
     except (FileNotFoundError,KeyError):rows['stored MLP (definitive run)']=None
    md.append(f'\n### {kind.upper()}\n\n| method | '+' | '.join(f'λ={l:g}' for l in lams)+' |\n|'+'---|'*(len(lams)+1))
    for k,v in rows.items():md.append(f'| {k} | '+(' | '.join(f'{x:.4f}' for x in v) if v is not None else 'incomplete')+' |')
    choice=defaultdict(int)
    for s in names:
     for l in lams:q=T[s][2][str(l)];choice[(tuple(q['architecture']),q['early_stopping'],q['alpha'])]+=1
    md.append('\nselected (architecture, early stopping, alpha) over splits × penalties: '+', '.join(f'{k}: {v}' for k,v in sorted(choice.items(),key=lambda kv:-kv[1])))
 (ROOT/'results/mlp_tuned').mkdir(parents=True,exist_ok=True);(ROOT/'results/mlp_tuned/SUMMARY.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))

if __name__=='__main__':main()
