"""Sensitivity checks on actual prompts (no prototype averaging) and real-data pooling."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from controlled import ci_domain
from geometry import ROOT,E4,FROZEN,unit,nearest,targets,behavior,composition

def literal_triplets():
    rows=[json.loads(s) for s in (ROOT/'inputs/controlled.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['kind']=='controlled'];n=len(rows)
    lookup={(r['content'],r['level'],r['mode'],r['position'],r['sham'],r['terminal']):i for i,r in enumerate(rows)}
    result={}
    for rep,pool in [('gte','primary'),('reader','primary'),('reader','mean'),('reader','last8')]:
        x=unit(np.load(ROOT/'encoded'/rep/f'{pool}.npy',mmap_mode='r')[:n]);out={}
        for position in (0,1):
            for terminal in (0,1):
                vals=[];margins=[];dom=[];answer_explain=[]
                for core in range(96):
                    for level in (0,1):
                        i=lookup[core,level,'answer',position,0,terminal]
                        twins=[lookup[core,level,m,position,0,terminal] for m in ('explain','python','json')]
                        rivals=[lookup[d*12+core%12,level,'answer',position,0,terminal] for d in range(8) if d!=core//12]
                        dt=((x[twins]-x[i])**2).sum(1);dr=((x[rivals]-x[i])**2).sum(1)
                        vals.append(float((dt[:,None]<dr[None]).mean()));margins.append(float(dr.mean()-dt.mean()));dom.append(core//12)
                        answer_explain.append(float((dt[0]<dr).mean()))
                out[f'position={position},terminal={terminal}']=dict(
                    same_problem_cross_format_win=ci_domain(np.array(vals),np.array(dom)),
                    answer_vs_explanation_only=ci_domain(np.array(answer_explain),np.array(dom)),
                    margin=ci_domain(np.array(margins),np.array(dom)),
                    primary=(position==0 and terminal==1),prototype_averaging=False)
        result[rep+'/'+pool]=out
    return result

def actual_pooling(panel):
    src=FROZEN/panel;z=np.load(src/'panel.npz');meta=json.loads((src/'manifest.json').read_text())
    labels=np.load(ROOT/'inputs'/f'{panel}_metadata.npz');fam=z['fam'];fmt=labels['format'];topic=labels['topic']
    ys=targets(z['r'].astype(float),z['c'].astype(float));out={}
    for pool in ('mean','last8'):
        path=E4/'features'/panel/'prompt/hidden/qwen2.5-7b-instruct'/f'L21_{pool}.npy'
        x=np.load(path,mmap_mode='r');hm=json.loads((path.parent/'meta.json').read_text())
        assert hm['rows']==meta['original_rows'] and len(x)==meta['original_rows']
        xu=unit(x[z['panel_rows']]);rep={}
        for g in np.unique(fam):
            train=np.flatnonzero(fam!=g);test=np.flatnonzero(fam==g)
            qi=np.sort(np.random.default_rng(9022+g).choice(test,min(256,len(test)),replace=False))
            edges=np.unique(np.quantile(np.log1p(z['reader_tokens'][train]),[.2,.4,.6,.8]));bins=np.digitize(np.log1p(z['reader_tokens']),edges)
            ii,d=nearest(xu[qi],xu[train]);nb=train[ii];arrays={}
            rep[str(g)]=dict(family=meta['families'][int(g)],neighbors=composition(nb,d,qi,fam,train,fmt,topic),
                behavior={name:behavior(nb,qi,train,Y,good,fam,bins,arrays,'pool/'+name) for name,(Y,good) in ys.items()})
        out[pool]=rep
    return out

def main():
    t=time.time();out={'literal_prompts':literal_triplets(),'real_pooling':{}}
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        out['real_pooling'][panel]=actual_pooling(panel);print('pooling done',panel,flush=True)
    out['seconds']=time.time()-t
    out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest=ROOT/'results/confirmation';dest.mkdir(parents=True,exist_ok=False)
    (dest/'result.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
