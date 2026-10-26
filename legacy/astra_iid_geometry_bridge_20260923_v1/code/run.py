"""Read-only reuse of fixed experiments; new outputs in this directory only."""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
E4=Path(os.environ.get('E4_ROOT',ROOT.parent))
DEF=E4/'astra_definitive_paper_20260923_v1'
DIFF=E4/'astra_sprout_difficulty_cells_20260923_v1'
sys.path.insert(0,str(DEF/'code'))
from common import load,sha,nearest
spec=importlib.util.spec_from_file_location('frozen_difficulty',DIFF/'code/run.py')
controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)

def donor_pool(tr,q,labels,fam):
    pool=tr[labels[tr]==labels[q]]
    return pool if len(pool) else tr[fam[tr]==fam[q]]

def means(tr,te,labels,fam,r,c):
    rr=np.empty_like(r[te]);cc=np.empty_like(c[te])
    for lab in np.unique(labels[te]):
        pos=np.flatnonzero(labels[te]==lab);pool=donor_pool(tr,te[pos[0]],labels,fam)
        assert len(pool) and len(np.unique(fam[te[pos]]))==1
        rr[pos]=r[pool].mean(0);cc[pos]=c[pool].mean(0)
    return rr,cc

def grouped_neighbors(x,tr,te,labels,fam):
    nn=np.full((len(te),100),-1,np.int32)
    for lab in np.unique(labels[te]):
        pos=np.flatnonzero(labels[te]==lab);pool=donor_pool(tr,te[pos[0]],labels,fam)
        ix,_=nearest(x[te[pos]],x[pool],min(100,len(pool)))
        nn[pos,:ix.shape[1]]=pool[ix]
    return nn

def neighbor_mean(nn,y,k):
    ids=nn[:,:k];valid=ids>=0;assert valid.any(1).all()
    return (y[np.maximum(ids,0)]*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]

def model_pair_accuracy(pred,actual):
    i,j=np.triu_indices(actual.shape[1],1)
    truth=actual[:,i]-actual[:,j];guess=pred[:,i]-pred[:,j]
    valid=np.abs(truth)>1e-12
    correct=(truth*guess>0).astype(float);correct[np.abs(guess)<=1e-12]=.5
    return np.divide((correct*valid).sum(1),valid.sum(1),out=np.full(len(actual),np.nan),where=valid.sum(1)>0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True,type=int);a=ap.parse_args()
    assert a.task in range(10);panel=['sprout-sixfam','routerbench-0shot'][a.task//5];seed=a.task%5
    dest=ROOT/'results'/panel/f'iid_{seed}';dest.mkdir(parents=True,exist_ok=False);t=time.time()
    lockpath=DEF/'config/SPROUT_CANONICAL_PROTOCOL.json'
    if panel=='sprout-sixfam':
        for file,h in json.loads(lockpath.read_text())['file_sha256'].items():
            if Path(file).name.endswith('manifest.json') or file.endswith('.npy') or not Path(file).exists():continue  # supplementary package
            assert sha(file)==h,file
    d=load(panel,'gte');x,r,c,f=d['x'],d['r'],d['c'],d['fam']
    frozen=np.load(DEF/'inputs'/panel/'panel.npz')
    for key in ['query_ids','r','c','fam']:assert np.array_equal(d[key],frozen[key]),key
    cellpath=(DEF/'inputs/sprout_cells_native_openhermes_v1/cells.npz' if panel=='sprout-sixfam' else DEF/'cell_lookup_v2'/panel/'cells.npz')
    cz=np.load(cellpath);assert np.array_equal(cz['query_ids'],d['query_ids']);cells=cz['cells']
    splitpath=DEF/'inputs'/panel/'splits.json';ss=json.loads(splitpath.read_text())[f'iid_{seed}'];tr=np.array(ss['train']);te=np.array(ss['test']);n=len(te);idx=np.arange(n)
    assert not np.intersect1d(tr,te).size
    cfg=json.loads((DEF/'config/protocol.json').read_text());lams=cfg['lambdas'][panel]
    globalpath=DEF/'results/knn'/panel/f'iid_{seed}/gte/predictions.npz';old=np.load(globalpath)
    assert np.array_equal(old['test_rows'],te) and np.array_equal(old['query_ids'],d['query_ids'][te])
    pred={'family_mean':means(tr,te,f,f,r,c),'cell_mean':means(tr,te,cells,f,r,c)}
    pred.update({f'global_knn{k}':(old[f'knn{k}|r'],old[f'knn{k}|c']) for k in [10,100]})
    assets={};sourcepaths=[cellpath,splitpath,globalpath,DEF/'inputs'/panel/'panel.npz']
    if panel=='sprout-sixfam':
        cp=DIFF/'results'/f'seed{seed}'/'query_metrics.npz';cached=np.load(cp)
        assert np.array_equal(cached['test_rows'],te) and np.array_equal(cached['query_ids'],d['query_ids'][te])
        nn=cached['approved_cell__nn100'];sourcepaths.append(cp)
    else:nn=grouped_neighbors(x,tr,te,cells,f)
    assets['cell_neighbors100']=nn
    for k in [10,100]:pred[f'cell_knn{k}']=(neighbor_mean(nn,r,k),neighbor_mean(nn,c,k))
    for groups in [2,4,8]:
        if panel=='sprout-sixfam':labels=cached[f'cell_clusters{groups}__assignments']
        else:labels,_=controls.refine(x,tr,te,cells,groups,20260923+seed)
        pred[f'group{groups}_mean']=means(tr,te,labels,f,r,c);assets[f'groups{groups}']=labels
        if groups==8:
            nn=cached['cell_clusters8__nn100'] if panel=='sprout-sixfam' else grouped_neighbors(x,tr,te,labels,f)
            assets['group8_neighbors100']=nn
            for k in [10,100]:pred[f'group8_knn{k}']=(neighbor_mean(nn,r,k),neighbor_mean(nn,c,k))
    print('PREDICTIONS READY',panel,seed,time.time()-t,flush=True)
    rr0,cc0=pred['cell_mean'];arrays={'test_rows':te,'query_ids':d['query_ids'][te],'family':f[te],**assets};results=[]
    # Verify reconstructed baseline choices against frozen lookup outputs.
    lookup=np.load(DEF/'cell_lookup_v2'/panel/f'iid_{seed}_predictions.npz')
    assert np.array_equal(lookup['test_rows'],te)
    for lam in lams:assert np.array_equal((rr0-lam*cc0).argmax(1),lookup[f'1|{lam:g}'])
    for name,(rr,cc) in pred.items():
        arrays[f'{name}|r']=rr;arrays[f'{name}|c']=cc
        arrays[f'{name}|mean_reward_mse']=(rr.mean(1)-r[te].mean(1))**2
        arrays[f'{name}|centered_reward_mse']=((rr-rr.mean(1,keepdims=True)-(r[te]-r[te].mean(1,keepdims=True)))**2).mean(1)
        for lam in lams:
            actual=r[te]-lam*c[te];base=(rr0-lam*cc0).argmax(1);choice=(rr-lam*cc).argmax(1)
            br=(rr-lam*cc0).argmax(1);bc=(rr0-lam*cc).argmax(1)
            u0=actual[idx,base];u=actual[idx,choice];ur=actual[idx,br];uc=actual[idx,bc]
            reward_part=.5*((ur-u0)+(u-uc));cost_part=.5*((uc-u0)+(u-ur))
            assert np.allclose(reward_part+cost_part,u-u0,atol=1e-12)
            key=f'{name}|{lam:g}';same=choice==base
            arrays[key+'|choice']=choice.astype(np.int16);arrays[key+'|utility']=u
            arrays[key+'|gain']=u-u0;arrays[key+'|agreement']=same.astype(float)
            arrays[key+'|reward_part']=reward_part;arrays[key+'|cost_part']=cost_part
            arrays[key+'|pair_accuracy']=model_pair_accuracy(rr-lam*cc,actual)
            results.append(dict(method=name,lambda_value=lam,utility=float(u.mean()),reward=float(r[te][idx,choice].mean()),cost=float(c[te][idx,choice].mean()),gain_vs_cell=float((u-u0).mean()),agreement=float(same.mean()),conditional_disagreement_gain=float((u-u0)[~same].mean()) if (~same).any() else None,
                 reward_part=float(reward_part.mean()),cost_part=float(cost_part.mean()),pair_accuracy=float(np.nanmean(arrays[key+'|pair_accuracy'])),
                 per_family={d['meta']['families'][g]:dict(utility=float(u[f[te]==g].mean()),gain=float((u-u0)[f[te]==g].mean()),agreement=float(same[f[te]==g].mean())) for g in np.unique(f)}))
    # Independent finite-donor random reference: route each draw separately.
    rng=np.random.default_rng(61000+seed);random_u=np.zeros((len(lams),n));random_r=np.zeros_like(random_u);random_c=np.zeros_like(random_u);random_agree=np.zeros_like(random_u)
    for draw in range(20):
        rr=np.empty_like(r[te]);cc=np.empty_like(c[te])
        for lab in np.unique(cells[te]):
            pos=np.flatnonzero(cells[te]==lab);pool=donor_pool(tr,te[pos[0]],cells,f);k=min(100,len(pool))
            for st in range(0,len(pos),128):
                ip=pos[st:st+128];nb=np.array([rng.choice(pool,k,replace=False) for _ in ip])
                rr[ip]=r[nb].mean(1);cc[ip]=c[nb].mean(1)
        for j,lam in enumerate(lams):
            ch=(rr-lam*cc).argmax(1);base=(rr0-lam*cc0).argmax(1)
            random_r[j]+=r[te][idx,ch]/20;random_c[j]+=c[te][idx,ch]/20
            random_u[j]+=(r[te]-lam*c[te])[idx,ch]/20;random_agree[j]+=(ch==base)/20
            arrays[f'random_cell100_draw{draw}|{lam:g}|choice']=ch.astype(np.int16)
        if draw%5==0:print('RANDOM',panel,seed,draw,time.time()-t,flush=True)
    for j,lam in enumerate(lams):
        key=f'random_cell100|{lam:g}';baseu=arrays[f'cell_mean|{lam:g}|utility'];arrays[key+'|utility']=random_u[j];arrays[key+'|gain']=random_u[j]-baseu;arrays[key+'|agreement']=random_agree[j]
        results.append(dict(method='random_cell100',lambda_value=lam,utility=float(random_u[j].mean()),reward=float(random_r[j].mean()),cost=float(random_c[j].mean()),gain_vs_cell=float((random_u[j]-baseu).mean()),agreement=float(random_agree[j].mean()),per_family={d['meta']['families'][g]:dict(utility=float(random_u[j][f[te]==g].mean()),gain=float((random_u[j]-baseu)[f[te]==g].mean())) for g in np.unique(f)}))
    # Verify unrestricted primary router exactly matches prior published run.
    previous=json.loads(globalpath.with_name('result.json').read_text())
    for rec in results:
        if rec['method']=='global_knn100':assert abs(rec['utility']-previous['results'][str(rec['lambda_value'])]['knn100']['utility'])<1e-10
    np.savez_compressed(dest/'predictions.npz',**arrays)
    report=dict(panel=panel,seed=seed,lambdas=lams,families=d['meta']['families'],n_test=n,results=results,
                input_hashes={str(p):sha(p) for p in sourcepaths},code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL.md'),geometry_code_sha256=sha(DIFF/'code/run.py'),
                canonical_sprout_lock_sha256=sha(lockpath) if panel=='sprout-sixfam' else None,
                checks=['frozen cohort row identity','split alignment','cell choices reproduced at every lambda','global kNN-100 utilities reproduced','reward/cost attribution identity'],seconds=time.time()-t)
    (dest/'result.json').write_text(json.dumps(report,indent=2));print('DONE',panel,seed,time.time()-t,flush=True)

if __name__=='__main__':main()
