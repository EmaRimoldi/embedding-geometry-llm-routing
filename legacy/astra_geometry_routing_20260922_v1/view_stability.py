"""Change only the router's view; keep the downstream benchmark query fixed."""
import json
import numpy as np
from common import *

VARIANTS=('original','trailing_space','trailing_newlines','response_marker','answer_marker','question_header')

def main():
    src=GEO/'real_presentation/results';arrays=np.load(src/'paired.npz');checks=json.loads((src/'result.json').read_text())
    out={};saved={}
    for panel,nf in [('routerbench-0shot',7),('llmrouterbench-performance-cost',10)]:
        d=load(panel);r,c,f=d['r'],d['c'],d['fam'];out[panel]={}
        for rep,pool in [('gte','primary'),('reader','primary'),('reader','mean'),('reader','last8')]:
            label=rep+'/'+pool;rec={}
            assert min(v['reencoding_cosine_to_cache'] for v in checks[panel][label].values())>.995
            for g in range(nf):
                key=f'{panel}|{rep}|{pool}|{g}';basekey=key+'|original';qi=arrays[basekey+'|ids'];pred={}
                for v in VARIANTS:
                    assert np.array_equal(qi,arrays[key+'|'+v+'|ids'])
                    nb=arrays[key+'|'+v+'|neighbors'];assert (f[nb]!=g).all() and (f[qi]==g).all()
                    pred[v]=(r[nb].mean(1),c[nb].mean(1))
                # Fixed equal weighting, three encoder views; no utility-based selection.
                views=('original','answer_marker','question_header')
                pred['three_view_average']=tuple(np.mean([pred[v][i] for v in views],axis=0) for i in (0,1))
                famres={}
                for lam in LAMS[panel]:
                    orig=(pred['original'][0]-lam*pred['original'][1]).argmax(1);rr={}
                    for v,(rp,cp) in pred.items():
                        ch=(rp-lam*cp).argmax(1)
                        rr[v]=dict(score=evaluate(ch,r[qi],c[qi],d['missing'][qi],lam),
                            vs_original=contrast(ch,orig,r[qi],c[qi],lam))
                        saved[key+'|'+v+f'|{lam:g}']=ch
                    famres[f'{lam:g}']=rr
                rec[str(g)]=dict(rows=len(qi),family=d['meta']['families'][g],results=famres)
            out[panel][label]=rec
    dest=ROOT/'results/view_stability';dest.mkdir(parents=True,exist_ok=False)
    (dest/'result.json').write_text(json.dumps(dict(results=out,protocol='Only encoder/router input is changed. The query actually sent to candidate models remains the original benchmark query; recorded rewards/costs are therefore evaluated for that original task. This is not an estimate of model outcomes for rewritten prompts.',
        warnings=['Outcome-independent short-query sample, not the full benchmark.','GTE strips trailing whitespace; marker/header controls change both token streams.','Three-view averaging uses three encoding passes; no total serving-cost claim.','All variants reported; no outer-utility selection.'],source_sha256=code_sha(__file__)),indent=2))
    np.savez_compressed(dest/'choices.npz',**saved);print('Routing-view stability done',flush=True)

if __name__=='__main__':main()
