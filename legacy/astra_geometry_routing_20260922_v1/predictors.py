"""Identical nonlinear learners in each space; no target-family data in fitting."""
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression

ALPHAS=(.1,1.,10.,100.)

def weights(f):
    _,i,n=np.unique(f,return_inverse=True,return_counts=True)
    return len(f)/(len(n)*n[i])

def standardized(x,xt,w):
    mu=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w));sd[sd<1e-7]=1
    return ((x-mu)/sd).astype(np.float32),((xt-mu)/sd).astype(np.float32),mu,sd

def prior(x,r,c,f,xt,seed):
    families=np.unique(f)
    clf=LogisticRegression(C=1.,class_weight='balanced',max_iter=500,random_state=seed)
    clf.fit(x,f);assert np.array_equal(clf.classes_,families)
    post=clf.predict_proba(xt)
    return post@np.stack([r[f==h].mean(0) for h in families]),post@np.stack([c[f==h].mean(0) for h in families])

def ridge(z,y,w,zt,alphas=ALPHAS):
    z=np.asarray(z,np.float64);zt=np.asarray(zt,np.float64)
    mu=np.average(z,axis=0,weights=w);ym=np.average(y,axis=0,weights=w)
    a=z-mu;b=zt-mu
    ev,v=eigh(a.T@(w[:,None]*a),check_finite=False);ev=np.maximum(ev,0)
    rhs=v.T@(a.T@(w[:,None]*(y-ym)))
    return {alpha:b@(v@(rhs/(ev[:,None]+alpha)))+ym for alpha in alphas}

def kernel_fit(x,ctrl,r,c,f,xt,ctrlt,seed):
    w=weights(f);rng=np.random.default_rng(seed);families=np.unique(f)
    cap=max(1,256//len(families))
    anchors=np.concatenate([rng.choice(np.flatnonzero(f==g),min(cap,np.sum(f==g)),replace=False) for g in families])
    a=x[anchors];ad=np.maximum(((a*a).sum(1)[:,None]+(a*a).sum(1)[None]-2*a@a.T),0)
    upper=ad[np.triu_indices(len(a),1)];scale=float(np.median(upper[upper>1e-8]));assert scale>0
    ds=np.maximum((x*x).sum(1)[:,None]+(a*a).sum(1)[None]-2*x@a.T,0)
    dt=np.maximum((xt*xt).sum(1)[:,None]+(a*a).sum(1)[None]-2*xt@a.T,0)
    y=np.c_[r,c];m=r.shape[1];out={};cs,ct,_,_=standardized(ctrl,ctrlt,w)
    for alpha,p in ridge(cs,y,w,ct,alphas=(1.,100.)).items():
        out[f'controls_a{alpha:g}']=(np.clip(p[:,:m],0,1),np.maximum(p[:,m:],0))
    for gamma in (.5,2.):
        ka=np.exp(-gamma*ad/scale).astype(np.float64)
        ev,v=eigh(ka,check_finite=False);transform=v/np.sqrt(np.maximum(ev,1e-6))[None,:]
        phi=(np.exp(-gamma*ds/scale)@transform).astype(np.float32)
        phit=(np.exp(-gamma*dt/scale)@transform).astype(np.float32)
        for aug in (False,True):
            z=np.c_[phi,cs] if aug else phi;zt=np.c_[phit,ct] if aug else phit
            for alpha,p in ridge(z,y,w,zt).items():
                name=f'rbf_g{gamma:g}_a{alpha:g}'+('_controls' if aug else '')
                out[name]=(np.clip(p[:,:m],0,1),np.maximum(p[:,m:],0))
    return out,dict(anchors=anchors.tolist(),median_squared_distance=scale,source_families=families.tolist())

def mlp_fit(x,ctrl,r,c,f,xt,ctrlt,seed):
    import torch
    import torch.nn as nn
    assert torch.cuda.is_available()
    device='cuda';w=weights(f)
    # Scalars are supplied identically; GTE norm is explicitly constant.
    xx=np.c_[x,ctrl];xxte=np.c_[xt,ctrlt];y=np.c_[r,c];m=r.shape[1]
    rng=np.random.default_rng(seed)
    val=np.concatenate([rng.choice(np.flatnonzero(f==g),max(1,int(.15*np.sum(f==g))),replace=False) for g in np.unique(f)])
    fit=np.setdiff1d(np.arange(len(f)),val)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.net=nn.Sequential(nn.Linear(xx.shape[1],256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,64),nn.GELU(),nn.Linear(64,y.shape[1]))
        def forward(self,z):return self.net(z)
    def run(ids,epochs,early=False):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        ww=weights(f[ids]);z,zt,xmu,xsd=standardized(xx[ids],xxte,ww)
        ym=np.average(y[ids],axis=0,weights=ww);ys=np.sqrt(np.average((y[ids]-ym)**2,axis=0,weights=ww));ys[ys<1e-7]=1
        X=torch.tensor(z,device=device);Y=torch.tensor((y[ids]-ym)/ys,dtype=torch.float32,device=device);W=torch.tensor(ww,dtype=torch.float32,device=device)
        net=Net().to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
        best=float('inf');bestep=1;stale=0
        if early:
            xv=torch.tensor((xx[val]-xmu)/xsd,dtype=torch.float32,device=device)
            yv=torch.tensor((y[val]-ym)/ys,dtype=torch.float32,device=device)
            wv=torch.tensor(weights(f[val]),dtype=torch.float32,device=device)
        for ep in range(1,epochs+1):
            net.train();perm=torch.randperm(len(ids),device=device)
            for st in range(0,len(ids),512):
                ii=perm[st:st+512];p=net(X[ii]);loss=((p-Y[ii]).square().mean(1)*W[ii]).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            if early:
                net.eval()
                with torch.inference_mode():vl=float((((net(xv)-yv).square().mean(1))*wv).mean().cpu())
                if vl<best-1e-5:best=vl;bestep=ep;stale=0
                else:stale+=1
                if stale>=8:break
        if early:return bestep,best
        net.eval();pred=[]
        with torch.inference_mode():
            for st in range(0,len(zt),512):pred.append(net(torch.tensor(zt[st:st+512],device=device)).cpu().numpy())
        p=np.concatenate(pred)*ys+ym
        return p,dict(epochs=epochs,source_x_mean=xmu,source_x_scale=xsd,target_mean=ym,target_scale=ys,
            state={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    ep,vl=run(fit,60,True);p,state=run(np.arange(len(f)),ep)
    return {'mlp':(np.clip(p[:,:m],0,1),np.maximum(p[:,m:],0))},dict(epochs=ep,internal_validation_loss=vl,seed=seed,checkpoint=state)
