"""Nested OOD evaluation with exact fallback and matched architectures."""
import argparse,json,time
import numpy as np
from learning_common import *
from predictors import kernel_fit,mlp_fit,prior

def options(names):
    # Exact fallback first; ties resolve toward smaller corrections.
    out=[('prior',0.,0.)]
    for name in names:
        for br in (0.,.5,1.):
            for bc in (0.,.5,1.):
                if br or bc:out.append((name,br,bc))
    return out

def decide(pred,prior_r,prior_c,cfg,lam):
    name,br,bc=cfg
    if name=='prior':rr,cc=prior_r,prior_c
    else:
        rr=(1-br)*prior_r+br*pred[name][0];cc=(1-bc)*prior_c+bc*pred[name][1]
    return (rr-lam*cc).argmax(1)

def select(inner,lams):
    names=list(next(iter(inner.values()))['pred']);opts=options(names);selections={};scores={}
    for lam in lams:
        vals=[]
        for cfg in opts:
            vv=[]
            for z in inner.values():
                ch=decide(z['pred'],z['prior_r'],z['prior_c'],cfg,lam)
                vv.append(float((z['r']-lam*z['c'])[np.arange(len(ch)),ch].mean()))
            vals.append(vv)
        vals=np.asarray(vals);mean=vals.mean(1)
        selected={}
        sets={'selected_blend':list(range(len(opts))),
              'direct': [i for i,o in enumerate(opts) if o[1:]==(1.,1.)],
              'reward_only':[i for i,o in enumerate(opts) if o[2]==0.],
              'cost_only':[i for i,o in enumerate(opts) if o[1]==0.]}
        if any(n.startswith('rbf') for n in names):
            sets['direct_geometry']=[i for i,o in enumerate(opts) if o[0].startswith('rbf') and not o[0].endswith('_controls') and o[1:]==(1.,1.)]
            sets['direct_controls']=[i for i,o in enumerate(opts) if o[0].startswith('controls') and o[1:]==(1.,1.)]
        for label,allowed in sets.items():
            best=max(allowed,key=lambda i:(mean[i],-sum(opts[i][1:])))
            selected[label]=opts[best]
        # A fixed paired one-standard-error safeguard, without outer outcomes.
        best=int(np.argmax(mean));delta=vals[best]-vals[0]
        se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        selected['one_se_blend']=opts[0] if delta.mean()<=se else opts[best]
        selections[f'{lam:g}']=selected;scores[f'{lam:g}']=dict(options=opts,per_inner_family=vals.tolist(),family_order=list(inner),paired_se_vs_prior=se)
    return selections,scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--kind',choices=['kernel','mlp'],required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    reps=REPS if a.kind=='kernel' else ('gte','reader')
    panel,g=FOLDS[a.task//len(reps)];rep=reps[a.task%len(reps)];t=time.time();d=load(panel,rep)
    f=d['fam'];tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g);x=d['x'];ctrl=d['controls'];r,c=d['r'],d['c']
    dest=ROOT/'results'/a.kind/f'seed_{a.seed}'/panel/rep/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    (dest/'started.json').write_text(json.dumps(dict(panel=panel,rep=rep,family=g,kind=a.kind,seed=a.seed)))
    fit_fn=kernel_fit if a.kind=='kernel' else mlp_fit
    inner={};settings={};oof_arrays={'source_rows':tr,'query_ids':d['ids'][tr]}
    for h in np.unique(f[tr]):
        fit=np.flatnonzero((f!=g)&(f!=h));va=np.flatnonzero(f==h)
        assert not np.any(np.isin(f[fit],[g,h]))
        pred,info=fit_fn(x[fit],ctrl[fit],r[fit],c[fit],f[fit],x[va],ctrl[va],a.seed)
        info.pop('checkpoint',None)
        pr,pc=prior(x[fit],r[fit],c[fit],f[fit],x[va],a.seed)
        inner[int(h)]=dict(pred=pred,prior_r=pr,prior_c=pc,r=r[va],c=c[va]);settings[str(h)]=info
        loc=np.searchsorted(tr,va)
        for name,(rr,cc) in {'prior':(pr,pc),**pred}.items():
            for what,arr in [('r',rr),('c',cc)]:
                key=name+'|'+what
                if key not in oof_arrays:oof_arrays[key]=np.full((len(tr),r.shape[1]),np.nan,np.float32)
                oof_arrays[key][loc]=arr
        print(panel,rep,g,'inner',h,round(time.time()-t),flush=True)
    selected,scores=select(inner,LAMS[panel])
    (dest/'selection.json').write_text(json.dumps(dict(selected=selected,scores=scores,inner_fit_settings=settings),indent=2))
    for k,v in oof_arrays.items():
        if k.endswith('|r') or k.endswith('|c'):assert np.isfinite(v).all()
    np.savez_compressed(dest/'inner_predictions.npz',**oof_arrays);del inner,oof_arrays
    pred,info=fit_fn(x[tr],ctrl[tr],r[tr],c[tr],f[tr],x[te],ctrl[te],a.seed)
    state=info.pop('checkpoint',None)
    if state:
        flat={k:v for k,v in state.items() if k!='state'};flat.update({'net|'+k:v for k,v in state['state'].items()});np.savez_compressed(dest/'checkpoint.npz',**flat)
    pr,pc=prior(x[tr],r[tr],c[tr],f[tr],x[te],a.seed)
    arrays={'test_rows':te,'query_ids':d['ids'][te]};results={}
    for name,(rr,cc) in {'prior':(pr,pc),**pred}.items():arrays[name+'|r']=rr.astype(np.float32);arrays[name+'|c']=cc.astype(np.float32)
    for lam in LAMS[panel]:
        key=f'{lam:g}';choices={'prior':decide(pred,pr,pc,('prior',0.,0.),lam)}
        choices.update({name:decide(pred,pr,pc,cfg,lam) for name,cfg in selected[key].items()})
        results[key]={name:evaluate(ch,r[te],c[te],d['missing'][te],lam) for name,ch in choices.items()}
        for name,ch in choices.items():arrays[key+'|choice|'+name]=ch.astype(np.int16)
    np.savez_compressed(dest/'predictions.npz',**arrays)
    result=dict(panel=panel,representation=rep,family_index=g,family=d['meta']['families'][g],rows=len(te),kind=a.kind,seed=a.seed,
        selected=selected,settings=info,results=results,snapshot_sha256=d['meta']['snapshot_sha256'],
        source_sha256=code_sha(__file__),predictor_sha256=code_sha(ROOT/'predictors.py'),common_sha256=code_sha(ROOT/'learning_common.py'),seconds=time.time()-t)
    (dest/'result.json').write_text(json.dumps(result,indent=2));print('DONE',a.kind,panel,rep,g,round(time.time()-t),flush=True)

if __name__=='__main__':main()
