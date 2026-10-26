"""Descriptive global and within-family similarity of the two native spaces."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
from geometry import ROOT,FROZEN,unit

def cka(a,b):
    a=a-a.mean(0);b=b-b.mean(0);ka=a@a.T;kb=b@b.T
    return float((ka*kb).sum()/np.sqrt((ka*ka).sum()*(kb*kb).sum()))

def rsa(a,b):
    ix=np.triu_indices(len(a),1)
    da=(1-unit(a)@unit(a).T)[ix];db=(1-unit(b)@unit(b).T)[ix]
    return float(spearmanr(da,db).statistic)

def main():
    out={}
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        src=FROZEN/panel;z=np.load(src/'panel.npz');f=z['fam'];meta=json.loads((src/'manifest.json').read_text())
        x={k:unit(np.load(src/f'{k}.npy',mmap_mode='r')) for k in ('gte','reader')};rng=np.random.default_rng(9022)
        rows=np.concatenate([rng.choice(np.flatnonzero(f==g),min(192,np.sum(f==g)),replace=False) for g in np.unique(f)])
        a,b=x['gte'][rows],x['reader'][rows]
        r=dict(scope='descriptive pooled geometry; never used to fit OOD transforms',rows=len(rows),
            linear_cka=cka(a,b),pairwise_distance_spearman=rsa(a,b),per_family={})
        ac=a.copy();bc=b.copy()
        for g in np.unique(f):
            ix=f[rows]==g;aa=a[ix];bb=b[ix]
            ac[ix]-=aa.mean(0);bc[ix]-=bb.mean(0)
            r['per_family'][meta['families'][int(g)]]=dict(rows=int(ix.sum()),linear_cka=cka(aa,bb),pairwise_distance_spearman=rsa(aa,bb))
        r['after_within_family_centering']=dict(linear_cka=cka(ac,bc),pairwise_distance_spearman=rsa(ac,bc))
        out[panel]=r
    dest=ROOT/'results/global';dest.mkdir(parents=True,exist_ok=False)
    (dest/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
