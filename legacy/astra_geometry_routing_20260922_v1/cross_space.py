"""Source-selected reward/cost representation combinations; no outer tuning."""
import argparse,json,time
import numpy as np
from learning_common import *

def centered(x):return x-x.mean(1,keepdims=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--kind',choices=['kernel','mlp'],required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    panel,g=FOLDS[a.task];d=load(panel);f=d['fam'];tr=np.flatnonzero(f!=g);te=np.flatnonzero(f==g)
    reps=REPS if a.kind=='kernel' else ('gte','reader');innerR={};innerC={};testR={};testC={};settings={}
    for rep in reps:
        src=ROOT/'results'/a.kind/f'seed_{a.seed}'/panel/rep/f'fold_{g:02d}'
        audit=json.loads((src/'result.json').read_text());assert audit['snapshot_sha256']==d['meta']['snapshot_sha256']
        inn=np.load(src/'inner_predictions.npz');out=np.load(src/'predictions.npz')
        assert np.array_equal(inn['source_rows'],tr) and np.array_equal(out['test_rows'],te)
        assert np.array_equal(inn['query_ids'],d['ids'][tr]) and np.array_equal(out['query_ids'],d['ids'][te])
        names=[k[:-2] for k in inn.files if k.endswith('|r') and k!='prior|r']
        losses={}
        for name in names:
            rr=[];cc=[]
            for h in np.unique(f[tr]):
                v=f[tr]==h;s=(f!=g)&(f!=h);scale=np.maximum(d['c'][s].mean(0),1e-6)
                rr.append(float(((centered(inn[name+'|r'][v])-centered(d['r'][tr][v]))**2).mean()))
                cc.append(float((((inn[name+'|c'][v]-d['c'][tr][v])/scale)**2).mean()))
            losses[name]=(float(np.mean(rr)),float(np.mean(cc)))
        rn=min(names,key=lambda n:losses[n][0]);cn=min(names,key=lambda n:losses[n][1]);settings[rep]=dict(reward_head=rn,cost_head=cn,losses=losses)
        for label,ra,ca in [('prior','prior','prior'),('fit',rn,cn)]:
            tag=rep+'/'+label;innerR[tag]=inn[ra+'|r'];innerC[tag]=inn[ca+'|c'];testR[tag]=out[ra+'|r'];testC[tag]=out[ca+'|c']
    opts=[('gte/prior','gte/prior')]+[(rr,cc) for rr in innerR for cc in innerC if (rr,cc)!=('gte/prior','gte/prior')]
    results={};arrays={'test_rows':te,'query_ids':d['ids'][te]};selected={};score_tables={}
    for lam in LAMS[panel]:
        scores=[]
        for rr,cc in opts:
            ch=(innerR[rr]-lam*innerC[cc]).argmax(1);u=(d['r'][tr]-lam*d['c'][tr])[np.arange(len(tr)),ch]
            scores.append([float(u[f[tr]==h].mean()) for h in np.unique(f[tr])])
        scores=np.asarray(scores);mean=scores.mean(1);best=int(np.argmax(mean));delta=scores[best]-scores[0]
        se=delta.std(ddof=1)/np.sqrt(len(delta));conservative=0 if delta.mean()<=se else best
        selections={'cross_selected':opts[best],'cross_one_se':opts[conservative]}
        for rep in reps:
            allowed=[i for i,(rr,cc) in enumerate(opts) if rr.split('/')[0]==cc.split('/')[0]==rep]
            selections['same_space_selected/'+rep]=opts[max(allowed,key=lambda i:mean[i])]
        selected[f'{lam:g}']=selections;score_tables[f'{lam:g}']=scores.tolist()
        choices={f'fixed_R={rr}/C={cc}':(testR[rr]-lam*testC[cc]).argmax(1) for rr,cc in opts}
        choices.update({name:(testR[rr]-lam*testC[cc]).argmax(1) for name,(rr,cc) in selections.items()})
        results[f'{lam:g}']={name:evaluate(ch,d['r'][te],d['c'][te],d['missing'][te],lam) for name,ch in choices.items()}
        for name,ch in choices.items():arrays[f'{lam:g}|{name}']=ch.astype(np.int16)
    dest=ROOT/'results/cross_space'/a.kind/f'seed_{a.seed}'/panel/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    (dest/'selection.json').write_text(json.dumps(dict(heads=settings,choices=selected,options=opts,source_family_scores=score_tables),indent=2))
    np.savez_compressed(dest/'paired.npz',**arrays)
    (dest/'result.json').write_text(json.dumps(dict(panel=panel,family_index=g,rows=len(te),kind=a.kind,seed=a.seed,
        results=results,snapshot_sha256=d['meta']['snapshot_sha256'],source_sha256=code_sha(__file__)),indent=2))
    print('DONE cross-space',a.kind,panel,g,flush=True)

if __name__=='__main__':main()
