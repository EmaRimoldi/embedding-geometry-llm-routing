"""Predefined paired contrasts and held-out-domain probes for factorial prompts."""
from pathlib import Path
import hashlib,itertools,json,time
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from geometry import ROOT,unit

FACTORS=('content','level','mode','position','sham','terminal')

def ci_domain(values,domains):
    means=np.array([np.mean(values[domains==d]) for d in np.unique(domains)])
    rng=np.random.default_rng(9022); boot=means[rng.integers(0,len(means),(5000,len(means)))].mean(1)
    return dict(mean=float(means.mean()),ci95_domains=np.quantile(boot,[.025,.975]).tolist(),
                per_domain=means.tolist(),pairs=len(values))

def main():
    rows=[json.loads(s) for s in (ROOT/'inputs/controlled.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['kind']=='controlled']; n=len(rows); t0=time.time()
    lookup={tuple(r[f] for f in FACTORS):i for i,r in enumerate(rows)}
    assert len(lookup)==n
    domains=np.array([r['domain'] for r in rows]); contents=np.array([r['content'] for r in rows])
    target=np.array([r['mode'] for r in rows]); wording=np.array([r['wording'] for r in rows])
    terminal=np.array([r['terminal'] for r in rows]); position=np.array([r['position'] for r in rows]); sham=np.array([r['sham'] for r in rows])
    lexical=HashingVectorizer(n_features=4096,alternate_sign=False,ngram_range=(1,2),norm='l2').transform([r['text'] for r in rows]).toarray().astype(np.float32)
    sets=[('gte','primary'),('reader','primary'),('reader','mean'),('reader','last8'),('lexical','hashed_words')]
    out=dict(rows=n,content_blocks=96,domains=8,primary_subset='prefix instruction; common terminal marker; neutral difficulty label',encoders={})
    arrays={}; contrasts=[('mode',a,b) for a,b in itertools.combinations(('answer','explain','python','json'),2)]
    contrasts += [('level',0,1),('sham',0,1),('sham',0,2),('sham',1,2),('position',0,1),('terminal',0,1)]
    for rep,pool in sets:
        if rep=='lexical':raw=lexical; tokens=np.array([len(r['text'].split()) for r in rows])
        else:
            manifest=json.loads((ROOT/'encoded'/rep/'manifest.json').read_text());assert manifest['canary_pass']
            raw=np.load(ROOT/'encoded'/rep/f'{pool}.npy',mmap_mode='r')[:n];tokens=np.load(ROOT/'encoded'/rep/'tokens.npy')[:n]
        x=unit(raw); name=f'{rep}/{pool}'; res=dict(contrasts={},format_probe={})
        # Content moves, holding format, wording version, demand and nuisance settings constant.
        content_pairs=[]
        for i,r in enumerate(rows):
            other=dict(r);other['content']=r['domain']*12+(r['seed']+3)%12
            j=lookup[tuple(other[f] for f in FACTORS)];content_pairs.append(j)
        content_distance=((x-x[content_pairs])**2).sum(1)
        res['within_domain_content_distance']=ci_domain(content_distance,domains)
        for factor,a,b in contrasts:
            base=[]; other=[]
            for i,r in enumerate(rows):
                if r[factor]!=a:continue
                s=dict(r);s[factor]=b;j=lookup[tuple(s[f] for f in FACTORS)]
                base.append(i);other.append(j)
            base=np.array(base);other=np.array(other);delta=x[other]-x[base];mag=(delta*delta).sum(1)
            tag=f'{factor}:{a}->{b}'; d=ci_domain(mag,domains[base]); dt=tokens[other]-tokens[base]
            d['mean_token_difference']=float(dt.mean());d['mean_absolute_token_difference']=float(np.abs(dt).mean())
            for tol in (0,2):
                ok=np.abs(dt)<=tol
                if ok.any():d[f'length_difference_le_{tol}']=ci_domain(mag[ok],domains[base][ok])
            if factor in ('mode','level'):
                clean=(position[base]==0)&(terminal[base]==1)&(sham[base]==0)
                d['primary_clean_subset']=ci_domain(mag[clean],domains[base][clean])
            dirs=unit(np.stack([delta[domains[base]==h].mean(0) for h in range(8)]))
            align=dirs@dirs.T;d['direction_cosine_across_domains']=float(align[np.triu_indices(8,1)].mean())
            res['contrasts'][tag]=d
            arrays[f'{name}|{tag}|content']=contents[base];arrays[f'{name}|{tag}|squared_distance']=mag
        # Same problem in another output mode versus another problem in the same mode.
        prototypes=[]; keys=[]
        for core,level,mode in itertools.product(range(96),range(2),('answer','explain','python','json')):
            ids=[lookup[(core,level,mode,p,s,t)] for p,s,t in itertools.product(range(2),range(3),range(2))]
            prototypes.append(x[ids].mean(0));keys.append((core,level,mode))
        proto=unit(np.asarray(prototypes)); pmap={k:i for i,k in enumerate(keys)}
        win=[]; pdom=[]; gaps=[]
        for i,(core,level,mode) in enumerate(keys):
            if mode!='answer':continue
            twins=[pmap[core,level,m] for m in ('explain','python','json')]
            # Match constructional demand and phrasing version, compare all other domains.
            rivals=[pmap[d*12+core%12,level,mode] for d in range(8) if d!=core//12]
            twin_d=((proto[twins]-proto[i])**2).sum(1)
            rival_d=((proto[rivals]-proto[i])**2).sum(1)
            win.append(float((twin_d[:,None]<rival_d[None,:]).mean()));gaps.append(float(rival_d.mean()-twin_d.mean()));pdom.append(core//12)
        res['same_problem_cross_format_nearer_than_other_domain_same_format']=ci_domain(np.asarray(win),np.asarray(pdom))
        res['triplet_margin']=ci_domain(np.asarray(gaps),np.asarray(pdom))
        arrays[f'{name}|prototypes']=proto; arrays[f'{name}|prototype_keys']=np.asarray(['|'.join(map(str,k)) for k in keys])
        # Test subjects AND instruction wording absent from training.
        # Dimensions reduced by an outcome-free random projection to limit logistic cost equally.
        rng=np.random.default_rng(9022); proj=rng.normal(size=(x.shape[1],256)).astype(np.float32)/np.sqrt(256)
        xp=x@proj
        for h in range(8):
            tr=(domains!=h)&(wording!=2);te=(domains==h)&(wording==2)
            sc=StandardScaler().fit(xp[tr]);clf=LogisticRegression(C=1.,max_iter=1000).fit(sc.transform(xp[tr]),target[tr])
            pr=clf.predict(sc.transform(xp[te]));res['format_probe'][str(h)]=dict(accuracy=float((pr==target[te]).mean()),n=int(te.sum()),
                majority=.25,feature_control='fixed 256-dimensional random projection, then source-only scaling')
        out['encoders'][name]=res;print('DONE',name,round(time.time()-t0),flush=True)
    out['seconds']=time.time()-t0;out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest=ROOT/'results/controlled';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'paired.npz',**arrays);(dest/'result.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
