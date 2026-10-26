"""Same frozen MF reproduction, with explicit input-scale controls."""
import ast,json,sys,time
import numpy as np
from common import *

def main():
    task=int(sys.argv[1]);panel,g=FOLDS[task//4];rep=('gte','reader')[(task%4)//2];mode=('unit','source_rms')[task%2]
    d=load(panel,rep);tr=np.flatnonzero(d['fam']!=g);te=np.flatnonzero(d['fam']==g);x=d['x'];scale=None
    if mode=='source_rms':
        raw=np.asarray(np.load(OLD/'inputs'/panel/f'{rep}.npy',mmap_mode='r'),np.float32)
        scale=float(np.sqrt(np.mean(np.sum(raw[tr].astype(np.float64)**2,axis=1))))
        assert np.isfinite(scale) and scale>0;x=raw/scale
    p=ROOT/'snapshots/matched_baselines.py';tree=ast.parse(p.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('functions','mf')]
    ns={'np':np,'Path':Path,'ast':ast,'ROOT':ROOT};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    t=time.time();out,settings=ns['mf'](x,d['r'],d['c'],d['fam'],g,LAMS[panel],d['meta']['models'],panel)
    settings.update(input_mode=mode,source_rms_scale=scale)
    arrays={'test_rows':te,'query_ids':d['ids'][te]};results={}
    for lam,methods in out.items():
        results[f'{lam:g}']={name:evaluate(ch,d['r'][te],d['c'][te],d['missing'][te],lam) for name,ch in methods.items()}
        for name,ch in methods.items():arrays[f'{lam:g}|{name}']=ch.astype(np.int16)
    dest=ROOT/'results/mf_control'/mode/panel/rep/f'fold_{g:02d}';dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'predictions.npz',**arrays)
    (dest/'result.json').write_text(json.dumps(dict(panel=panel,representation=rep,family_index=g,rows=len(te),settings=settings,results=results,
        snapshot_sha256=d['meta']['snapshot_sha256'],source_sha256=code_sha(__file__),reproduction_sha256=code_sha(p),
        core_sha256=code_sha(ROOT/'baseline_source/heldout_routellm_mf.py'),seconds=time.time()-t),indent=2))
    print('DONE MF scale control',panel,g,rep,mode,round(time.time()-t),flush=True)

if __name__=='__main__':main()
