"""Matched IID/OOD accessibility and source-only cross-space mapping."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from geometry import ROOT,FROZEN,TASKS,unit,nearest
sys.path.insert(0,str(ROOT/'vendor'))
from reward_core_snapshot import ridge_many

ALPHAS=(10.,100.,1000.)

def pca_source(x,fam,fit,seed):
    rng=np.random.default_rng(seed)
    rows=np.concatenate([rng.choice(fit[fam[fit]==g],min(512,np.sum(fam[fit]==g)),replace=False) for g in np.unique(fam[fit])])
    _,fi,ct=np.unique(fam[rows],return_inverse=True,return_counts=True);w=1/ct[fi];w/=w.sum()
    mu=np.average(x[rows],axis=0,weights=w)
    z=(x[rows]-mu)*np.sqrt(w[:,None])
    _,s,v=randomized_svd(z,n_components=min(128,len(rows)-1,x.shape[1]),n_iter=4,random_state=seed)
    return mu,v.T,float((s*s).sum()/(z*z).sum())

def scores(y,p,source_mean):
    mse=float(((p-y)**2).mean()); baseline=float(((y-source_mean)**2).mean())
    var=float(((y-y.mean(0))**2).mean())
    return dict(mse=mse,source_mean_mse=baseline,r2_vs_source_mean=1-mse/baseline if baseline>1e-12 else None,
                r2_within_test=1-mse/var if var>1e-12 else None,rows=len(y))

def main():
    panel,outer=TASKS[int(sys.argv[1])]; t0=time.time();src=FROZEN/panel
    z=np.load(src/'panel.npz');fam=z['fam'];r=z['r'].astype(float);c=z['c'].astype(float)
    meta=json.loads((src/'manifest.json').read_text());labs=np.load(ROOT/'inputs'/f'{panel}_metadata.npz')
    rng=np.random.default_rng(9022+outer);source=np.flatnonzero(fam!=outer);outerrows=np.flatnonzero(fam==outer)
    val=np.concatenate([rng.choice(source[fam[source]==g],max(1,int(.2*np.sum(fam[source]==g))),replace=False) for g in np.unique(fam[source])])
    fit=np.setdiff1d(source,val)
    iid=np.concatenate([rng.choice(val[fam[val]==g],min(128,np.sum(fam[val]==g)),replace=False) for g in np.unique(fam[val])])
    ood=np.sort(rng.choice(outerrows,min(256,len(outerrows)),replace=False));test=np.r_[iid,ood]
    assert not np.any(fam[fit]==outer) and not np.intersect1d(fit,test).size
    raw={k:np.load(src/f'{k}.npy',mmap_mode='r') for k in ('gte','reader')};xu={k:unit(v) for k,v in raw.items()}
    length=np.column_stack([np.log1p(z['reader_tokens']),np.log1p(labs['chars']),np.log1p(labs['lines'])])
    result=dict(panel=panel,family_index=outer,family=meta['families'][outer],n_fit=len(fit),n_iid=len(iid),n_ood=len(ood),
        primary_alpha=100.,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snapshot_sha256=meta['snapshot_sha256'],encoders={},cost_probes={})
    arrays={'iid_rows':iid,'ood_rows':ood,'source_fit_rows':fit}
    fields={'reward_level':slice(0,1),'solve_fraction':slice(1,2),'reward_profile':slice(2,2+r.shape[1])}
    behavior=np.c_[r.mean(1),(r>=.5).mean(1),r-r.mean(1,keepdims=True)]
    if panel=='routerbench-0shot':
        fields['format']=slice(behavior.shape[1],behavior.shape[1]+1);behavior=np.c_[behavior,labs['format']]
    _,fi,ct=np.unique(fam[fit],return_inverse=True,return_counts=True);w=len(fit)/(len(ct)*ct[fi])
    for rep in ('gte','reader'):
        other='reader' if rep=='gte' else 'gte'
        mu,v,share=pca_source(xu[other],fam,fit,9022+outer)
        mapcoords=(xu[other]-mu)@v;Y=np.c_[behavior,mapcoords]
        controls=np.c_[length,np.log(np.maximum(np.linalg.norm(raw[rep],axis=1),1e-12))]
        features={'length':length,'norm_length':controls,'unit':xu[rep],
                  'unit_controls':np.c_[xu[rep],controls],'raw':np.asarray(raw[rep])}
        res=dict(features={},map_target=other,map_target_pca_variance=share)
        for name,x in features.items():
            pred=ridge_many(x[fit],Y[fit],w,x[test],alphas=ALPHAS)
            source_mean=np.average(Y[fit],axis=0,weights=w);rec={}
            for alpha,(p,_) in pred.items():
                rr={}
                for split,sl,ids in [('iid',slice(0,len(iid)),iid),('ood',slice(len(iid),len(test)),ood)]:
                    rr[split]={f:scores(Y[ids,cols],p[sl,cols],source_mean[cols]) for f,cols in fields.items()}
                    if 'format' in fields:
                        rr[split]['format']['accuracy']=float(((p[sl,fields['format']][:,0]>=.5)==labs['format'][ids]).mean())
                    rr[split]['cross_space']=scores(mapcoords[ids],p[sl,behavior.shape[1]:],source_mean[behavior.shape[1]:])
                rec[str(alpha)]=rr
                if alpha==100.:
                    arrays[f'{rep}|{name}|predicted_behavior']=p[:,:behavior.shape[1]].astype(np.float32)
                    if name in ('unit','unit_controls'):
                        projected=p[len(iid):,behavior.shape[1]:].astype(np.float32)
                        ni,_=nearest(projected,mapcoords[fit].astype(np.float32),k=20)
                        true,_=nearest(mapcoords[ood].astype(np.float32),mapcoords[fit].astype(np.float32),k=20)
                        rr['ood']['mapped_neighbor_overlap_k20']=float(np.mean([len(set(a)&set(b))/20 for a,b in zip(ni,true)]))
                        arrays[f'{rep}|{name}|mapped_neighbors']=fit[ni]
                        arrays[f'{rep}|{name}|target_neighbors']=fit[true]
            res['features'][name]=rec; print(panel,outer,rep,name,round(time.time()-t0),flush=True)
        result['encoders'][rep]=res
    # Cost targets evaluated only on rows where every model has strictly positive recorded cost.
    good=(c>0).all(1);cf=fit[good[fit]];cv=iid[good[iid]];co=ood[good[ood]];ce=np.r_[cv,co]
    if len(cf)>200 and len(ce):
        logc=np.log(c[good]);Y=np.zeros((len(c),1+c.shape[1]))
        Y[good]=np.c_[logc.mean(1),logc-logc.mean(1,keepdims=True)]
        _,fi,ct=np.unique(fam[cf],return_inverse=True,return_counts=True);w=len(cf)/(len(ct)*ct[fi])
        for rep in ('gte','reader'):
            controls=np.c_[length,np.log(np.maximum(np.linalg.norm(raw[rep],axis=1),1e-12))]
            for name,x in [('norm_length',controls),('unit',xu[rep]),('unit_controls',np.c_[xu[rep],controls])]:
                pred=ridge_many(x[cf],Y[cf],w,x[ce],alphas=ALPHAS);base=np.average(Y[cf],axis=0,weights=w);rec={}
                for a,(p,_) in pred.items():
                    rec[str(a)]={}
                    for split,sl,ids in [('iid',slice(0,len(cv)),cv),('ood',slice(len(cv),len(ce)),co)]:
                        if len(ids):rec[str(a)][split]={'logcost_level':scores(Y[ids,:1],p[sl,:1],base[:1]),'logcost_profile':scores(Y[ids,1:],p[sl,1:],base[1:])}
                    if a==100.:arrays[f'cost|{rep}|{name}|predictions']=p.astype(np.float32)
                result['cost_probes'][rep+'/'+name]=rec
        arrays['cost_iid_rows']=cv;arrays['cost_ood_rows']=co
    result['seconds']=time.time()-t0;dest=ROOT/'results/probes'/panel;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'fold_{outer:02d}.npz',**arrays)
    (dest/f'fold_{outer:02d}.json').write_text(json.dumps(result,indent=2));print('DONE',panel,outer,flush=True)

if __name__=='__main__':main()
