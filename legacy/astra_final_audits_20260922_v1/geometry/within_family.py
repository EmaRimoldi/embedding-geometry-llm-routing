"""Within-source-family feature association, with source mean preserved before renormalization."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E4=Path(os.environ['E4_ROOT']); OLD=E4/'astra_geometry_routing_20260922_v1'
sys.path.insert(0,str(OLD))
from common import load,nearest,evaluate,FOLDS,LAMS,unit

def basis_fit(x,z,w):
    mu=np.average(z,axis=0,weights=w); sd=np.sqrt(np.average((z-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1
    a=(z-mu)/sd;xm=np.average(x,axis=0,weights=w)
    b=np.linalg.solve((a*w[:,None]).T@a+np.eye(a.shape[1]),(a*w[:,None]).T@(x-xm))
    _,s,v=np.linalg.svd(b,full_matrices=False);rank=int((s>max(s[0]*1e-6,1e-8)).sum())
    return v[:rank].T.astype(np.float32),dict(rank=rank,singular_values=s.tolist())

def run(task):
    panel,g=FOLDS[task];d=load(panel);f=d['fam'];tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g)
    z=np.load(OLD/'inputs/surface'/f'{panel}.npy');names=json.loads((OLD/'inputs/surface'/f'{panel}.json').read_text())['feature_names']
    # Equal total weight to each source family; no target rows/outcomes enter fitting.
    w=1/np.bincount(f[tr])[f[tr]];w=w/w.mean()
    mu=np.average(z[tr],axis=0,weights=w);sd=np.sqrt(np.average((z[tr]-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1
    zs=(z-mu)/sd
    fr={h:d['r'][f==h].mean(0) for h in np.unique(f[tr])};fc={h:d['c'][f==h].mean(0) for h in np.unique(f[tr])}
    rn=np.stack([fr[h] for h in f[tr]]);cn=np.stack([fc[h] for h in f[tr]])
    cs=np.maximum(d['c'][tr].mean(0),1e-8)
    result=dict(panel=panel,family_index=g,family=d['meta']['families'][g],rows=len(te),feature_names=names,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),snapshot_sha256=d['meta']['snapshot_sha256'],representations={})
    arrays={'test_rows':te,'query_ids':d['ids'][te]}
    for rep in ['gte','reader','reader_mean']:
        x=load(panel,rep)['x'];xm=np.average(x[tr],axis=0,weights=w).astype(np.float32);original=None;repout={}
        xfit=x[tr].copy();zfit=z[tr].copy()
        for h in np.unique(f[tr]):
            mask=f[tr]==h;xfit[mask]-=xfit[mask].mean(0);zfit[mask]-=zfit[mask].mean(0)
        configurations=[('original',None)]
        for label,cols in [('length',[0]),('format',list(range(2,14))+[15]),('all_surface',list(range(z.shape[1])))]:
            b,meta=basis_fit(xfit,zfit[:,cols],w);configurations.append(('remove_'+label,(b,meta)))
        ranks=sorted(set(q[1][0].shape[1] for q in configurations[1:]))
        for rank in ranks:
            b=np.linalg.qr(np.random.default_rng(911+g+rank).normal(size=(x.shape[1],rank)))[0].astype(np.float32)
            configurations.append((f'remove_random_rank{rank}',(b,dict(rank=rank))))
        # Surface-only Euclidean retrieval is a feature control, not a semantic encoder.
        if rep=='gte':configurations.append(('surface_only',None))
        for label,binfo in configurations:
            if label=='surface_only':xx=zs.astype(np.float32)
            elif binfo is None:xx=x
            else:
                b,_=binfo;xx=unit(x-((x-xm)@b)@b.T)
            ix,_=nearest(xx[te],xx[tr],100);ii=tr[ix]
            if original is None:original=ix
            overlap=np.array([len(set(a)&set(b))/100 for a,b in zip(original,ix)])
            counts=np.stack([(f[ii]==h).mean(1) for h in np.unique(f[tr])],1)
            pr=d['r'][ii].mean(1);pc=d['c'][ii].mean(1);pfr=rn[ix].mean(1);pfc=cn[ix].mean(1)
            dr=pr-d['r'][te];dc=pc-d['c'][te]
            rz=zs[ii].mean(1);zd=rz-zs[te]
            metrics=dict(neighbor_overlap=float(overlap.mean()),dominant_source_share=float(counts.max(1).mean()),
                donor_family_mean_shares=counts.mean(0).tolist(),source_families=[d['meta']['families'][h] for h in np.unique(f[tr])],
                feature_mean_squared_error=(zd**2).mean(0).tolist(),reward_mse=float((dr**2).mean()),
                relative_reward_mse=float(((dr-dr.mean(1,keepdims=True))**2).mean()),
                cost_mse=float((dc**2).mean()),cost_scaled_mse=float(((dc/cs)**2).mean()),
                removed=binfo[1] if binfo else None,routing={})
            for lam in LAMS[panel]:
                ch=(pr-lam*pc).argmax(1);cf=(pfr-lam*pfc).argmax(1)
                metrics['routing'][f'{lam:g}']={'knn':evaluate(ch,d['r'][te],d['c'][te],d['missing'][te],lam),
                    'family_composition':evaluate(cf,d['r'][te],d['c'][te],d['missing'][te],lam)}
                arrays[f'{rep}|{label}|{lam:g}']=ch.astype(np.int16)
            repout[label]=metrics
            print(panel,g,rep,label,metrics['neighbor_overlap'],flush=True)
        result['representations'][rep]=repout
    dest=ROOT/'geometry/within_family_results'/panel/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'choices.npz',**arrays);(dest/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':run(int(sys.argv[1]))
