"""Cheap observable-text controls, using identical outer families and source tuning."""
import argparse,json,re,sys,time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from learning_common import *
from predictors import weights,standardized,ridge
from nonlinear import select,decide

def prepare():
    sys.path.insert(0,str(OLD/'vendor'))
    from rb.data import load_panel
    dest=ROOT/'inputs/surface';dest.mkdir(parents=True,exist_ok=False)
    names=['log_tokens','log_chars','log_lines','digit_fraction','punctuation_fraction','log_question_marks',
           'log_line_choices','log_parenthesized_choices','log_code_fences','log_code_keywords',
           'log_question_headers','log_answer_headers','log_explanation_words','log_option_words','log_words','log_equals']
    for panel in LAMS:
        d=load(panel);p=load_panel(panel);ids=d['panel_rows'];assert np.array_equal(p.query_ids[ids].astype(str),d['ids'])
        data=[]
        for text,tokens in zip(p.prompts[ids],d['tokens']):
            s=str(text);n=max(1,len(s));low=s.lower()
            data.append([np.log1p(tokens),np.log1p(len(s)),np.log1p(s.count('\n')+1),sum(ch.isdigit() for ch in s)/n,
                sum(ch in '.,;:!?()[]{}' for ch in s)/n,np.log1p(s.count('?')),
                np.log1p(len(re.findall(r'^\s*[A-Ea-e][.)]\s',s,re.M))),np.log1p(len(re.findall(r'\([A-Ea-e]\)',s))),
                np.log1p(s.count('```')),np.log1p(len(re.findall(r'\b(?:def|import|return|class|print)\b',s))),
                np.log1p(low.count('question:')),np.log1p(low.count('answer:')),
                np.log1p(len(re.findall(r'\b(?:explain|reasoning|justify)\b|step by step',low))),
                np.log1p(len(re.findall(r'\b(?:choices|options)\b|multiple.choice',low))),np.log1p(len(s.split())),np.log1p(s.count('='))])
        x=np.asarray(data,np.float32);assert np.isfinite(x).all()
        np.save(dest/f'{panel}.npy',x)
        (dest/f'{panel}.json').write_text(json.dumps(dict(feature_names=names,rows=len(x),source_sha256=code_sha(__file__),snapshot_sha256=d['meta']['snapshot_sha256']),indent=2))
    print('Surface features prepared',flush=True)

def fit_predict(x,r,c,f,xt):
    w=weights(f);z,zt,_,_=standardized(x,xt,w);y=np.c_[r,c];m=r.shape[1];out={}
    for label,cols in [('length',[0]),('surface',list(range(x.shape[1])))]:
        for a,p in ridge(z[:,cols],y,w,zt[:,cols],alphas=(1.,100.,10000.)).items():
            out[f'{label}_a{a:g}']=(np.clip(p[:,:m],0,1),np.maximum(p[:,m:],0))
    mu=np.average(y,axis=0,weights=w);sd=np.sqrt(np.average((y-mu)**2,axis=0,weights=w));sd[sd<1e-8]=1
    model=ExtraTreesRegressor(n_estimators=100,max_depth=8,min_samples_leaf=30,random_state=42,n_jobs=2)
    model.fit(z,(y-mu)/sd,sample_weight=w);p=model.predict(zt)*sd+mu
    out['surface_trees']=(np.clip(p[:,:m],0,1),np.maximum(p[:,m:],0))
    return out

def run(task):
    panel,g=FOLDS[task];d=load(panel);f=d['fam'];tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g)
    x=np.load(ROOT/'inputs/surface'/f'{panel}.npy');r,c=d['r'],d['c'];src=ROOT/'results/kernel/seed_42'/panel/'gte'/f'fold_{g:02d}'
    pi=np.load(src/'inner_predictions.npz');po=np.load(src/'predictions.npz');assert np.array_equal(pi['source_rows'],tr) and np.array_equal(po['test_rows'],te)
    inner={};oof={'source_rows':tr,'query_ids':d['ids'][tr]}
    for h in np.unique(f[tr]):
        ti=np.flatnonzero((f!=g)&(f!=h));va=np.flatnonzero(f==h);loc=np.searchsorted(tr,va)
        pred=fit_predict(x[ti],r[ti],c[ti],f[ti],x[va]);pr=pi['prior|r'][loc];pc=pi['prior|c'][loc]
        inner[int(h)]=dict(pred=pred,prior_r=pr,prior_c=pc,r=r[va],c=c[va])
        for name,(rr,cc) in {'prior':(pr,pc),**pred}.items():
            for target,arr in [('r',rr),('c',cc)]:
                key=name+'|'+target
                if key not in oof:oof[key]=np.full((len(tr),r.shape[1]),np.nan,np.float32)
                oof[key][loc]=arr
    selected,scores=select(inner,LAMS[panel]);dest=ROOT/'results/surface/seed_42'/panel/'text'/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    (dest/'selection.json').write_text(json.dumps(dict(selected=selected,scores=scores),indent=2))
    pred=fit_predict(x[tr],r[tr],c[tr],f[tr],x[te]);pr,pc=po['prior|r'],po['prior|c'];arrays={'test_rows':te,'query_ids':d['ids'][te]};results={}
    for name,(rr,cc) in {'prior':(pr,pc),**pred}.items():arrays[name+'|r']=rr;arrays[name+'|c']=cc
    for lam in LAMS[panel]:
        ch={name:decide(pred,pr,pc,cfg,lam) for name,cfg in selected[f'{lam:g}'].items()};ch['prior']=(pr-lam*pc).argmax(1)
        results[f'{lam:g}']={name:evaluate(v,r[te],c[te],d['missing'][te],lam) for name,v in ch.items()}
        for name,v in ch.items():arrays[f'{lam:g}|choice|{name}']=v
    np.savez_compressed(dest/'predictions.npz',**arrays);np.savez_compressed(dest/'inner_predictions.npz',**oof)
    (dest/'result.json').write_text(json.dumps(dict(panel=panel,representation='text',family_index=g,rows=len(te),seed=42,kind='surface',results=results,
        selected=selected,snapshot_sha256=d['meta']['snapshot_sha256'],source_sha256=code_sha(__file__)),indent=2));print('DONE surface',panel,g,flush=True)

if __name__=='__main__':
    prepare() if sys.argv[1]=='prepare' else run(int(sys.argv[1]))
