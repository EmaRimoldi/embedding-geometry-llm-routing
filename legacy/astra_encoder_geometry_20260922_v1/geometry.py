"""Source-fitted geometry, true family removal, and matched donor controls."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parent
E4=Path(os.environ.get('E4_ROOT',ROOT.parent))
FROZEN=E4/'astra_reward_transfer_20260922_v1/inputs'
TASKS=[(p,g) for p,n in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)] for g in range(n)]
KS=(5,20,100,500)
def unit(x):
    # Cast and validate before accumulating squared values for the norm.
    # Cached pooling arrays can be float16 with norms exceeding its square range.
    x=np.asarray(x,dtype=np.float32)
    assert np.isfinite(x).all(), 'Nonfinite input representation'
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.isfinite(norms).all(), 'Nonfinite representation norm'
    return x/np.maximum(norms,1e-12)

def nearest(q,x,k=500,query_ids=None,source_ids=None):
    k=min(k,len(x)-(1 if query_ids is not None else 0)); out=[]; ds=[]
    sq=(x*x).sum(1)
    for st in range(0,len(q),128):
        a=q[st:st+128]; d=(a*a).sum(1)[:,None]+sq-2*a@x.T; np.maximum(d,0,out=d)
        if query_ids is not None:
            lookup={int(v):i for i,v in enumerate(source_ids)}
            for i,qid in enumerate(query_ids[st:st+len(a)]):
                if int(qid) in lookup:d[i,lookup[int(qid)]]=np.inf
        ix=np.argpartition(d,k-1,axis=1)[:,:k]; dv=np.take_along_axis(d,ix,axis=1)
        order=np.argsort(dv,axis=1); out.append(np.take_along_axis(ix,order,axis=1)); ds.append(np.take_along_axis(dv,order,axis=1))
    return np.concatenate(out),np.concatenate(ds)

def transforms(raw,fam,train,fmt,seed):
    x=unit(raw); rng=np.random.default_rng(seed)
    cal=np.concatenate([rng.choice(train[fam[train]==f],min(512,np.sum(fam[train]==f)),replace=False) for f in np.unique(fam[train])])
    _,fi,co=np.unique(fam[cal],return_inverse=True,return_counts=True); w=1/co[fi]; w/=w.sum()
    z=x[cal].astype(float); mu=np.average(z,axis=0,weights=w); z-=mu
    cov=z.T@(w[:,None]*z); trace=float(np.trace(cov)); pr=trace**2/float((cov*cov).sum())
    k=min(128,len(cal)-1,x.shape[1]); ev,V=eigh(cov,subset_by_index=[len(mu)-k,len(mu)-1],check_finite=False)
    ev=ev[::-1]; V=V[:,::-1]; centered=x-mu.astype(np.float32)
    axes={}; variance={}; format_valid=np.unique(fmt[cal])
    if len(format_valid)==2 and min(format_valid)>=0:
        d=np.average(x[cal][fmt[cal]==1],axis=0,weights=w[fmt[cal]==1])-np.average(x[cal][fmt[cal]==0],axis=0,weights=w[fmt[cal]==0])
        d=unit(d[None])[0]; axes['format_removed']=d
        target=float(d@cov@d); variance['format_removed']=target/trace
        low=np.flatnonzero(ev<=target)
        for j in range(3):
            if not len(low):break
            t=int(rng.choice(low)); frac=np.clip((target-ev[t])/max(ev[0]-ev[t],1e-12),0,1)
            u=np.sqrt(frac)*V[:,0]+(-1 if j%2 else 1)*np.sqrt(1-frac)*V[:,t]
            u=unit(u[None])[0]; axes[f'variance_matched_removed_{j}']=u
            variance[f'variance_matched_removed_{j}']=float(u@cov@u)/trace
    def get(mode):
        if mode=='raw_euclidean':return np.asarray(raw,np.float32)
        if mode=='native_cosine':return x
        if mode=='centered_cosine':return unit(centered)
        if mode.startswith('pc') and mode.endswith('_removed'):
            n=int(mode[2:].split('_')[0]); v=V[:,:n].astype(np.float32)
            return unit(centered-(centered@v)@v.T)
        if mode.startswith('pca128'):
            a=centered@V.astype(np.float32)
            if mode.endswith('white'):a=a/np.sqrt(np.maximum(ev,0)+.01*ev.mean()).astype(np.float32)
            return unit(a)
        u=axes[mode];return unit(centered-(centered@u)[:,None]*u)
    modes=['native_cosine','raw_euclidean','centered_cosine','pc1_removed','pc10_removed','pca128','pca128_white']+list(axes)
    meta=dict(calibration_rows=len(cal),calibration_families=np.unique(fam[cal]).tolist(),
        participation_ratio=pr,top1_variance=float(ev[0]/trace),top10_variance=float(ev[:10].sum()/trace),
        top128_variance=float(ev.sum()/trace),axis_removed_variance_fraction=variance,
        norm_quantiles=np.quantile(np.linalg.norm(raw[train],axis=1),[0,.1,.5,.9,1]).tolist())
    return modes,get,meta

def targets(r,c):
    good=np.all(c>0,axis=1); logc=np.zeros_like(c); logc[good]=np.log(c[good])
    return {'reward_level':(r.mean(1)[:,None],np.ones(len(r),bool)),
        'solve_fraction':((r>=.5).mean(1)[:,None],np.ones(len(r),bool)),
        'reward_profile':(r-r.mean(1)[:,None],np.ones(len(r),bool)),
        'logcost_level':(logc.mean(1)[:,None],good),
        'logcost_profile':(logc-logc.mean(1)[:,None],good)}

def behavior(nb,qi,train,Y,good,fam,bins,arrays,prefix):
    result={}; validq=good[qi]; groups=fam*10+bins
    source=train[good[train]]
    if not len(source):return result
    mus={int(g):Y[source[groups[source]==g]].mean(0) for g in np.unique(groups[source])}
    ss={int(g):(Y[source[groups[source]==g]]**2).mean(0) for g in np.unique(groups[source])}
    sizes={int(g):int(np.sum(groups[source]==g)) for g in np.unique(groups[source])}
    # Filter donors BEFORE truncating to requested k for positive-cost-only diagnostics.
    candidates=[row[good[row]] for row in nb]
    for k in KS:
        keep=np.array([validq[i] and len(v)>=k for i,v in enumerate(candidates)])
        if not np.any(keep):continue
        nn=np.stack([v[:k] for v,ok in zip(candidates,keep) if ok]); y=Y[qi[keep]]
        nd=Y[nn]; mse=((nd.mean(1)-y)**2).mean(1); pair=((nd-y[:,None])**2).mean((1,2))
        group=groups[nn]; expected=np.zeros_like(y); second=np.zeros_like(y)
        own_group=groups[qi[keep]]; self_in_pool=np.isin(qi[keep],source)
        for g in np.unique(group):
            weight=(group==g).mean(1)
            mu=np.broadcast_to(mus[int(g)],y.shape).copy(); s2=np.broadcast_to(ss[int(g)],y.shape).copy()
            remove=self_in_pool&(own_group==g); n=sizes[int(g)]
            if n>1:
                mu[remove]=(n*mu[remove]-y[remove])/(n-1)
                s2[remove]=(n*s2[remove]-y[remove]**2)/(n-1)
            else:assert np.all(weight[remove]==0),'Self-only stratum cannot supply a neighbor'
            expected+=weight[:,None]*mu; second+=weight[:,None]*s2
        rand=(y*y-2*y*expected+second).mean(1); matched_mse=((expected-y)**2).mean(1)
        gain=rand-pair
        arrays[f'{prefix}|k{k}|rows']=qi[keep]; arrays[f'{prefix}|k{k}|distance_gain']=gain
        result[str(k)]=dict(queries=int(keep.sum()),nearest_pair_mse=float(pair.mean()),
            matched_random_pair_mse=float(rand.mean()),relative_distance_gain=float(1-pair.mean()/rand.mean()) if rand.mean()>1e-12 else None,
            prediction_mse=float(mse.mean()),matched_composition_prediction_mse=float(matched_mse.mean()),
            donor_condition='source family and source-fitted input-length quintile',
            complete_positive_cost_only=not bool(good.all()))
    return result

def composition(nb,d,qi,fam,train,fmt,topic):
    out={}; F=int(fam.max())+1
    for k in KS:
        if nb.shape[1]<k:continue
        nn=nb[:,:k]; shares=np.stack([(fam[nn]==f).mean(1) for f in range(F)],axis=1)
        entropy=-(shares*np.log(np.maximum(shares,1e-30))).sum(1)
        count=np.bincount(nn.ravel(),minlength=len(fam))[train]; top=max(1,int(np.ceil(.01*len(train))))
        record=dict(source_shares=shares.mean(0).tolist(),mean_largest_source_share=float(shares.max(1).mean()),
            effective_source_families=float(np.exp(entropy).mean()),
            mean_squared_distance=float(d[:,:k].mean()),nearest_squared_distance=float(d[:,0].mean()),
            top_one_percent_source_retrieval_share=float(np.sort(count)[-top:].sum()/max(count.sum(),1)),
            same_family_share=float((fam[nn]==fam[qi,None]).mean()))
        if np.all(fmt[qi]>=0):
            record['same_format_share']=float((fmt[nn]==fmt[qi,None]).mean())
            record['source_format_chance']=float(np.mean([(fmt[train]==f).mean() for f in fmt[qi]]))
        known=topic[qi]!=''
        if known.any():
            eligible=topic[nn[known]]!=''; matched=topic[nn[known]]==topic[qi[known],None]
            if eligible.any():record['same_topic_among_labeled_donors']=float(matched[eligible].mean())
        out[str(k)]=record
    return out

def main():
    task=int(sys.argv[1]); panel,g=TASKS[task]; t0=time.time()
    src=FROZEN/panel; z=np.load(src/'panel.npz'); meta=json.loads((src/'manifest.json').read_text())
    labels=np.load(ROOT/'inputs'/f'{panel}_metadata.npz'); fam=z['fam']; r=z['r'].astype(float); c=z['c'].astype(float)
    fmt,topic=labels['format'],labels['topic']; train=np.flatnonzero(fam!=g); test=np.flatnonzero(fam==g)
    assert not np.any(fam[train]==g)
    rng=np.random.default_rng(9022+g); qi=np.sort(rng.choice(test,min(256,len(test)),replace=False))
    edges=np.unique(np.quantile(np.log1p(z['reader_tokens'][train]),[.2,.4,.6,.8])); bins=np.digitize(np.log1p(z['reader_tokens']),edges)
    ys=targets(r,c); arrays={'query_rows':qi,'panel_rows':z['panel_rows'][qi]}
    result=dict(panel=panel,family_index=g,family=meta['families'][g],n_test=len(test),n_queries=len(qi),
        source_families=np.unique(fam[train]).tolist(),snapshot_sha256=meta['snapshot_sha256'],encoders={})
    native_neighbors={}
    for rep in ('gte','reader'):
        raw=np.load(src/f'{rep}.npy',mmap_mode='r'); modes,get,struct=transforms(raw,fam,train,fmt,9022+g)
        info=dict(structure=struct,modes={})
        for mode in modes:
            x=get(mode); ii,dd=nearest(x[qi],x[train]); nb=train[ii]
            assert not np.any(fam[nb]==g)
            stats=composition(nb,dd,qi,fam,train,fmt,topic); beh={}
            for name,(Y,good) in ys.items():beh[name]=behavior(nb,qi,train,Y,good,fam,bins,arrays,f'{rep}|{mode}|{name}')
            info['modes'][mode]=dict(neighbors=stats,behavior=beh)
            if mode=='native_cosine':
                native_neighbors[rep]=nb; arrays[f'{rep}|native_neighbors']=nb
                allrows=np.arange(len(fam)); ai,ad=nearest(x[qi],x,query_ids=qi,source_ids=allrows)
                info['ordinary_neighbors']=composition(ai,ad,qi,fam,allrows,fmt,topic)
                # Explicitly reproduce the old conditional-outsiders statistic, with denominators.
                old=ai[:,:100]; outsider=fam[old]!=g
                info['conditional_outsiders']=dict(outside_edges=int(outsider.sum()),total_edges=int(old.size),
                    fraction=float(outsider.mean()),same_format_fraction=float((fmt[old][outsider]==np.broadcast_to(fmt[qi,None],old.shape)[outsider]).mean()) if outsider.any() and np.all(fmt[qi]>=0) else None)
            if mode in ('native_cosine','pca128_white'):
                ii2,dd2=nearest(x[qi],x[test],k=100,query_ids=qi,source_ids=test); nn2=test[ii2]
                info['modes'][mode]['within_family']=composition(nn2,dd2,qi,fam,test,fmt,topic)
                info['modes'][mode]['ood_to_within_nearest_distance_ratio']=float(np.mean(dd[:,0]/np.maximum(dd2[:,0],1e-10)))
                info['modes'][mode]['within_family_behavior']={name:behavior(nn2,qi,test,Y,good,fam,bins,arrays,f'{rep}|{mode}|within|{name}') for name,(Y,good) in ys.items()}
            print(panel,g,rep,mode,round(time.time()-t0),flush=True)
            del x
        result['encoders'][rep]=info
    overlap={}
    for k in KS:
        a,b=native_neighbors['gte'][:,:k],native_neighbors['reader'][:,:k]
        values=np.array([len(set(aa)&set(bb))/k for aa,bb in zip(a,b)])
        overlap[str(k)]=float(values.mean());arrays[f'neighbor_overlap_k{k}']=values
    result['neighbor_overlap']=overlap; result['seconds']=time.time()-t0
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest=ROOT/'results/atlas'/panel;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'fold_{g:02d}_paired.npz',**arrays)
    (dest/f'fold_{g:02d}.json').write_text(json.dumps(result,indent=2));print('DONE',panel,g,flush=True)

if __name__=='__main__':main()
