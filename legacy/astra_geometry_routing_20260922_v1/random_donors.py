"""Match the neighbor sample size as well as family/length composition."""
import sys,json,time
import numpy as np
from common import *

def main():
    panel,g=FOLDS[int(sys.argv[1])];d=load(panel);f=d['fam'];tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g);r,c=d['r'],d['c'];t=time.time()
    src=ROOT/'results/donors'/panel/f'fold_{g:02d}';saved=np.load(src/'paired.npz');assert np.array_equal(saved['test_rows'],te)
    edges=np.unique(np.quantile(np.log1p(d['tokens'][tr]),[.2,.4,.6,.8]));bins=np.digitize(np.log1p(d['tokens']),edges)
    results={};arrays={'test_rows':te};draws=10
    for rep in REPS:
        nb=saved[rep+'|neighbors'][:,:100]
        for reference,groups in [('family',f),('family_length',10*f+bins)]:
            pools={int(h):tr[groups[tr]==h] for h in np.unique(groups[tr])}
            musR={h:r[ids].mean(0) for h,ids in pools.items()};musC={h:c[ids].mean(0) for h,ids in pools.items()}
            nr=np.zeros((draws,len(te),r.shape[1]),np.float32);nc=np.zeros_like(nr)
            er=np.zeros((len(te),r.shape[1]));ec=np.zeros_like(er)
            rng=np.random.default_rng(44821+g)
            for j,neighbors in enumerate(nb):
                labs,counts=np.unique(groups[neighbors],return_counts=True)
                for h,n in zip(labs,counts):er[j]+=n/100*musR[int(h)];ec[j]+=n/100*musC[int(h)]
                for b in range(draws):
                    donors=np.concatenate([rng.choice(pools[int(h)],int(n),replace=False) for h,n in zip(labs,counts)])
                    assert len(donors)==100 and len(np.unique(donors))==100
                    nr[b,j]=r[donors].mean(0);nc[b,j]=c[donors].mean(0)
            rec={}
            for lam in LAMS[panel]:
                ch=(nr-lam*nc).argmax(2);idx=np.arange(len(te));uu=(r[te]-lam*c[te])[idx[None,:],ch]
                smooth=(er-lam*ec).argmax(1);base=saved[f'{lam:g}|{rep}/k100/knn'];baseu=(r[te]-lam*c[te])[idx,base]
                meanu=uu.mean(0)
                rec[f'{lam:g}']=dict(random_mean_utility=float(meanu.mean()),random_replication_utilities=uu.mean(1).tolist(),
                    knn_mean_utility=float(baseu.mean()),knn_minus_random=float((baseu-meanu).mean()),
                    smoothed=evaluate(smooth,r[te],c[te],d['missing'][te],lam))
                arrays[f'{rep}|{reference}|{lam:g}|random_mean_utility']=meanu
                arrays[f'{rep}|{reference}|{lam:g}|smoothed_choice']=smooth.astype(np.int16)
            results[rep+'/'+reference]=rec;print(panel,g,rep,reference,round(time.time()-t),flush=True)
    dest=ROOT/'results/random_donors'/panel/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'paired.npz',**arrays)
    (dest/'result.json').write_text(json.dumps(dict(panel=panel,family_index=g,rows=len(te),draws=draws,results=results,
        snapshot_sha256=d['meta']['snapshot_sha256'],source_sha256=code_sha(__file__),seconds=time.time()-t),indent=2))

if __name__=='__main__':main()
