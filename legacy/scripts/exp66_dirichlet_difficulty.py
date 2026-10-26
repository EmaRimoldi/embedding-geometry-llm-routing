"""ProbeDirichlet-style difficulty regression on frozen full-layer mean states.

Adaptation of Wu et al. 2026: global learned Dirichlet layer weights, sampled
in training, expected weights in evaluation, linear prediction head. Their
binary weak-model correctness target is replaced by benchmark solve fraction.
No released prompt over the model window is admitted. Frozen states and panel
are read only; all outputs are isolated under results/exp66_dirichlet/.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rb.data import SUCCESS_THRESHOLD,load_panel
from rb.panelctx import topic_str, _rb_record_prefix, _HS
from rb.probes.sweep import split

def safe_topic_str(panel):
    if not panel.name.startswith("routerbench"):
        return topic_str(panel)
    pre=_rb_record_prefix(panel)
    src=np.array(["wikihow" if _HS.search(str(t)) else "activitynet" for t in panel.prompts])
    hs=np.char.add("hellaswag/",src)
    return np.where(panel.dataset_ids=="mmlu",pre,np.where(panel.dataset_ids=="hellaswag",hs,panel.dataset_ids))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',required=True)
    ap.add_argument('--reader',choices=['qwen2.5-7b-instruct','gte-qwen2-7b'],required=True)
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    torch.manual_seed(6200+a.seed);np.random.seed(6200+a.seed)
    torch.set_num_threads(8)
    p=load_panel(a.panel);fam=p.dataset_index()
    xdir=ROOT/'features'/a.panel/'prompt'/'hidden'/a.reader
    meta=json.loads((xdir/'meta.json').read_text())
    layers=list(range(meta['layers']))
    paths=[xdir/f'L{l}_mean.npy' for l in layers]
    assert all(z.exists() for z in paths)
    lengths=np.load(ROOT/'features'/a.panel/'prompt'/'lengths.npz')
    key='qwen_tokens' if a.panel.startswith('routerbench') else 'gte_tokens'
    if key not in lengths.files:
        raise KeyError(f'{key} absent from lengths.npz; refusing to silently admit truncated inputs')
    lens=lengths[key]; lens=lens[p.keep] if p.keep is not None and len(lens)==len(p.keep) else lens
    assert len(lens)==p.n
    valid=np.flatnonzero(lens<8000)
    assert int((lens>=8192).sum())==meta['texts_truncated'], 'Token counts do not reproduce extraction metadata'
    r=np.nan_to_num(p.rewards,nan=0.0)
    y=((r>=SUCCESS_THRESHOLD).mean(1)).astype(np.float32)
    _,cell=np.unique(np.asarray(safe_topic_str(p),dtype=str),return_inverse=True)
    tr0,te0=split(p.n,fam,a.seed)
    tr=np.intersect1d(tr0,valid);te=np.intersect1d(te0,valid)
    fit,val=train_test_split(tr,test_size=.15,random_state=6200+a.seed,stratify=fam[tr])
    out=ROOT/'results'/'exp66_dirichlet'/a.panel/a.reader/f'seed{a.seed}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    # Load once, retaining float16 on GPU. The 29 frozen layers are <= 8 GB on
    # RouterBench, and no model weights or gradients are needed for their states.
    t0=time.time();mm=[np.load(z,mmap_mode='r') for z in paths]
    assert all(z.shape[0] in (p.n,len(p.keep) if p.keep is not None else p.n) for z in mm)
    if p.keep is not None and len(mm[0])==len(p.keep):
        arr=np.stack([z[p.keep] for z in mm],axis=1)
    else:arr=np.stack(mm,axis=1)
    assert arr.shape[:2]==(p.n,len(layers))
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    H=torch.from_numpy(arr).to(dev)
    del arr,mm
    yy=torch.from_numpy(y).to(dev)
    floor=np.full(p.n,float(y[tr].mean()),dtype=np.float32)
    for c in np.unique(cell[te]):
        ids=tr[cell[tr]==c]
        if len(ids)>=50:floor[te[cell[te]==c]]=y[ids].mean()
    D=H.shape[-1]; L=H.shape[1]
    results={}
    for kind in ('mean','dirichlet'):
        torch.manual_seed(6200+a.seed)
        head=torch.nn.Linear(D,1).to(dev)
        theta=torch.nn.Parameter(torch.zeros(L,device=dev)) if kind=='dirichlet' else None
        logmass=torch.nn.Parameter(torch.tensor(float(np.log(L)),device=dev)) if kind=='dirichlet' else None
        params=list(head.parameters())+([theta,logmass] if theta is not None else [])
        opt=torch.optim.Adam(params,lr=1e-4)
        best=np.inf;best_state=None;history=[];bad=0
        for epoch in range(a.epochs):
            head.train();order=np.random.permutation(fit)
            for batch in np.array_split(order,max(1,(len(order)+127)//128)):
                h=H[batch].float()
                if kind=='mean':w=torch.full((len(batch),L),1/L,device=dev)
                else:
                    alpha=torch.softmax(theta,0)*torch.exp(logmass)
                    w=torch.distributions.Dirichlet(alpha).rsample((len(batch),))
                fused=(h*w[:,:,None]).sum(1)
                pred=head(fused).squeeze(1)
                loss=torch.nn.functional.mse_loss(pred,yy[batch])
                opt.zero_grad();loss.backward();opt.step()
            head.eval()
            with torch.no_grad():
                w=torch.softmax(theta,0) if kind=='dirichlet' else torch.full((L,),1/L,device=dev)
                pred=head((H[val].float()*w[None,:,None]).sum(1)).squeeze(1)
                vl=float(torch.nn.functional.mse_loss(pred,yy[val]).item())
            history.append(vl)
            if vl<best-1e-6:
                best=vl;bad=0
                best_state=([z.detach().cpu().clone() for z in head.parameters()],w.detach().cpu().numpy().copy(),epoch)
            else:bad+=1
            print(a.panel,a.reader,kind,'epoch',epoch,'val_mse',round(vl,5),flush=True)
            if bad>=8:break
        with torch.no_grad():
            head.weight.copy_(best_state[0][0].to(dev));head.bias.copy_(best_state[0][1].to(dev))
            w=torch.tensor(best_state[1],device=dev)
            q=head((H[te].float()*w[None,:,None]).sum(1)).squeeze(1).cpu().numpy()
        den=float(np.sum((y[te]-floor[te])**2));ss=float(np.sum((y[te]-q)**2))
        results[kind]={'r2_beyond_cell':1-ss/den,'mse':ss/len(te),'cell_mse':den/len(te),
                       'mae':float(np.mean(np.abs(y[te]-q))),'best_epoch':best_state[2],
                       'validation_mse':best,'layer_weights':best_state[1].tolist(),'val_history':history}
        np.save(out/f'{kind}_test_predictions.npy',np.column_stack([te,y[te],floor[te],q]).astype(np.float32))
        print('RESULT',a.panel,a.reader,kind,results[kind]['r2_beyond_cell'],flush=True)
    report={**p.provenance(),'method':'ProbeDirichlet-style global layer weights plus linear head; MSE solve-fraction adaptation; mean-pool all 29 layers; same 70/30 split for controls','reader':a.reader,'reader_checkpoint':meta['path'],'truncated_in_meta':meta['texts_truncated'],'valid_rows':len(valid),'train_rows':len(fit),'validation_rows':len(val),'test_rows':len(te),'seed':a.seed,'source_meta_sha256':hashlib.sha256((xdir/'meta.json').read_bytes()).hexdigest(),'results':results,'seconds':time.time()-t0}
    (out/'result.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
