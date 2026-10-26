"""Paired bootstrap summary of frozen GTE emitted vs L21 neighborhood results."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'exp65_l21'/'summary.json'
if OUT.exists():raise FileExistsError(OUT)
summary={}
for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
    if panel.startswith('router'):
        old=ROOT/'results'/'difficulty_neighborhoods'/'exp52_conditional'
        new=ROOT/'results'/'exp65_l21'/panel/'difficulty_l21'/'exp65_conditional'
    else:
        old=ROOT/'results'/'exp65_l21'/panel/'difficulty_filtered'/'emitted'
        new=ROOT/'results'/'exp65_l21'/panel/'difficulty_filtered'/'l21'
    a=np.load(old/'query_metrics.npy');b=np.load(new/'query_metrics.npy')
    assert all(np.array_equal(a[z],b[z]) for z in ('seed','query','proxy','k','random'))
    summary[panel]={'emitted':str(old),'l21':str(new),'metrics':{}}
    for v,name in enumerate(('solve_fraction','mean_reward','reward_profile')):
        aa=a[(a['proxy']==v)&(a['k']==10)];bb=b[(b['proxy']==v)&(b['k']==10)]
        n=int(aa['query'].max())+1;ct=np.bincount(aa['query'],minlength=n);q=np.flatnonzero(ct)
        den=np.bincount(aa['query'],weights=aa['random'],minlength=n)/np.maximum(ct,1)
        near_a=np.bincount(aa['query'],weights=aa['near'],minlength=n)/np.maximum(ct,1)
        near_b=np.bincount(bb['query'],weights=bb['near'],minlength=n)/np.maximum(ct,1)
        fam=np.bincount(aa['query'],weights=aa['family'],minlength=n)/np.maximum(ct,1)
        groups=[q[fam[q]==f] for f in np.unique(fam[q])]
        rng=np.random.default_rng(650+v);boot=np.empty(1000)
        for j in range(1000):
            z=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            boot[j]=(near_a[z].mean()-near_b[z].mean())/den[z].mean()
        d=den[q].mean()
        summary[panel]['metrics'][name]={'n_unique_queries':len(q),'emitted_reduction':float(1-near_a[q].mean()/d),
            'l21_reduction':float(1-near_b[q].mean()/d),'l21_minus_emitted':float((near_a[q].mean()-near_b[q].mean())/d),
            'paired_ci95':np.quantile(boot,[.025,.975]).tolist()}
OUT.write_text(json.dumps(summary,indent=2))
print(OUT)
