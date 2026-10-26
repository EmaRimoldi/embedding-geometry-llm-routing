"""Exact reward/cost swaps and donor-family composition controls."""
import json,sys,time
import numpy as np
from common import *

def main():
    panel,g=FOLDS[int(sys.argv[1])];t=time.time();d=load(panel);r,c,f=d['r'],d['c'],d['fam']
    tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g);assert not np.any(f[tr]==g)
    nf=len(d['meta']['families']);rt=np.zeros((nf,r.shape[1]));ct=np.zeros_like(rt)
    for h in np.unique(f[tr]):rt[h]=r[tr[f[tr]==h]].mean(0);ct[h]=c[tr[f[tr]==h]].mean(0)
    pred={};arrays={'test_rows':te,'query_ids':d['ids'][te]};results={};all_choices={}
    for rep in REPS:
        x=load(panel,rep)['x'];ii,dd=nearest(x[te],x[tr]);nb=tr[ii]
        arrays[rep+'|neighbors']=nb;arrays[rep+'|distance']=dd
        for k in (10,40,100):
            jj=nb[:,:k];rn=r[jj].mean(1);cn=c[jj].mean(1);rf=rt[f[jj]].mean(1);cf=ct[f[jj]].mean(1)
            pred[rep,k]=(rn,cn,rf,cf)
        print(panel,g,rep,'retrieval done',round(time.time()-t),flush=True)
    for lam in LAMS[panel]:
        choices={};key=f'{lam:g}'
        for (rep,k),(rn,cn,rf,cf) in pred.items():
            for name,rr,cc in [('knn',rn,cn),('family_composition',rf,cf),('neighbor_reward',rn,cf),('neighbor_cost',rf,cn)]:
                choices[f'{rep}/k{k}/{name}']=(rr-lam*cc).argmax(1)
        for a in REPS:
            for b in REPS:
                choices[f'swap_reward={a}/cost={b}']=(pred[a,100][0]-lam*pred[b,100][1]).argmax(1)
        choices['best_single_source']=np.full(len(te),(r[tr]-lam*c[tr]).mean(0).argmax())
        ev={name:evaluate(ch,r[te],c[te],d['missing'][te],lam) for name,ch in choices.items()}
        contrasts={}
        for rep in REPS:
            for k in (10,40,100):
                for name in ('knn','neighbor_reward','neighbor_cost'):
                    a=f'{rep}/k{k}/{name}';b=f'{rep}/k{k}/family_composition'
                    contrasts[a+'_vs_composition']=contrast(choices[a],choices[b],r[te],c[te],lam)
        contrasts['reader_vs_gte_knn100']=contrast(choices['reader/k100/knn'],choices['gte/k100/knn'],r[te],c[te],lam)
        # Two-factor Shapley decomposition of the total GTE-to-Qwen utility change.
        ugg=ev['swap_reward=gte/cost=gte']['utility'];uqg=ev['swap_reward=reader/cost=gte']['utility']
        ugq=ev['swap_reward=gte/cost=reader']['utility'];uqq=ev['swap_reward=reader/cost=reader']['utility']
        sr=.5*((uqg-ugg)+(uqq-ugq));sc=.5*((ugq-ugg)+(uqq-uqg));assert abs(sr+sc-(uqq-ugg))<1e-10
        results[key]=dict(methods=ev,contrasts=contrasts,reward_cost_swap_decomposition=dict(reward=sr,cost=sc,total=uqq-ugg,
            interpretation='Algebraic attribution over four predicted-head combinations; not a causal claim about the encoder'))
        for name,ch in choices.items():arrays[key+'|'+name]=ch.astype(np.int16)
    out=dict(panel=panel,family_index=g,family=d['meta']['families'][g],rows=len(te),results=results,
        snapshot_sha256=d['meta']['snapshot_sha256'],source_sha256=code_sha(__file__),common_sha256=code_sha(ROOT/'common.py'),seconds=time.time()-t)
    dest=ROOT/'results/donors'/panel/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'paired.npz',**arrays);(dest/'result.json').write_text(json.dumps(out,indent=2))
    print('DONE',panel,g,round(time.time()-t),flush=True)

if __name__=='__main__':main()
