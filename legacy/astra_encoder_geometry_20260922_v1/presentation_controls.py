"""Real-query presentation sensitivity. Never treat perturbed prompts as new outcome labels."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
from geometry import ROOT,FROZEN,unit,nearest

DEST=ROOT/'real_presentation'
VARIANTS=('original','trailing_space','trailing_newlines','response_marker','answer_marker','question_header')

def mutate(s,v):
    return {'original':s,'trailing_space':s+' ','trailing_newlines':s+'\n\n',
            'response_marker':s+'\n\nResponse:','answer_marker':s+'\n\nAnswer:',
            'question_header':'Question:\n'+s}[v]

def prepare():
    sys.path.insert(0,str(ROOT/'vendor'))
    from rb.data import load_panel
    DEST.mkdir(exist_ok=False);(DEST/'inputs').mkdir();(DEST/'vendor').mkdir()
    shutil.copy2(ROOT/'encode.py',DEST/'encode.py')
    shutil.copy2(ROOT/'vendor/embed_gpu_snapshot.py',DEST/'vendor/embed_gpu_snapshot.py')
    corpus=[];cohorts={}
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        p=load_panel(panel);z=np.load(FROZEN/panel/'panel.npz');meta=json.loads((FROZEN/panel/'manifest.json').read_text())
        assert np.array_equal(p.query_ids[z['panel_rows']].astype(str),z['query_ids'])
        fam=z['fam'];eligible=np.flatnonzero(z['reader_tokens']<=1536);chosen=[]
        for g in np.unique(fam):
            candidates=eligible[fam[eligible]==g]
            ix=np.sort(np.random.default_rng(19342+int(g)).choice(candidates,min(24,len(candidates)),replace=False))
            chosen.extend(ix.tolist())
            for i in ix:
                s=str(p.prompts[z['panel_rows'][i]])
                for v in VARIANTS:
                    corpus.append(dict(kind='presentation',panel=panel,row=int(i),family=int(g),variant=v,text=mutate(s,v)))
        cohorts[panel]=dict(rows=chosen,families=meta['families'],max_original_tokens=1536)
    # Independent canaries reproduce the existing encoding check. Original variants also
    # quantify re-encoding discrepancy for every query in this new study.
    old=[json.loads(s) for s in (ROOT/'inputs/controlled.jsonl').read_text().splitlines()]
    corpus.extend(r for r in old if r['kind']=='canary')
    path=DEST/'inputs/controlled.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in corpus))
    (DEST/'protocol.json').write_text(json.dumps(dict(cohorts=cohorts,variants=VARIANTS,seed=19342,
        scope='Formatting perturbations on real query text; no changed-prompt rewards are observed or inferred.',
        primary='trailing whitespace; markers and header are separately reported because they add framing',
        corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
    print('prepared',len(corpus),'texts',flush=True)

def analyze():
    rows=[json.loads(s) for s in (DEST/'inputs/controlled.jsonl').read_text().splitlines()]
    out={};saved={}
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        z=np.load(FROZEN/panel/'panel.npz');fam=z['fam'];meta=json.loads((FROZEN/panel/'manifest.json').read_text());panelout={}
        for rep in ('gte','reader'):
            pools=['primary'] if rep=='gte' else ['primary','mean','last8']
            for pool in pools:
                em=json.loads((DEST/'encoded'/rep/'manifest.json').read_text());assert em['canary_pass'] and em['truncated']==0
                new=unit(np.load(DEST/'encoded'/rep/f'{pool}.npy'))
                if pool=='primary':cached=unit(np.load(FROZEN/panel/f'{rep}.npy'))
                else:
                    from geometry import E4
                    cached=unit(np.load(E4/'features'/panel/'prompt/hidden/qwen2.5-7b-instruct'/f'L21_{pool}.npy',mmap_mode='r')[z['panel_rows']])
                res={}
                for g in np.unique(fam):
                    ix=[i for i,r in enumerate(rows) if r['kind']=='presentation' and r['panel']==panel and r['family']==g and r['variant']=='original']
                    ids=np.array([rows[i]['row'] for i in ix]);source=np.flatnonzero(fam!=g)
                    if not len(ids):continue
                    assert not np.any(fam[source]==g)
                    base=new[ix];nn0,_=nearest(base,cached[source],k=20);nn0=source[nn0]
                    records={}
                    for v in VARIANTS:
                        vi=[i for i,r in enumerate(rows) if r['kind']=='presentation' and r['panel']==panel and r['family']==g and r['variant']==v]
                        assert [rows[i]['row'] for i in vi]==ids.tolist()
                        q=new[vi];nn,_=nearest(q,cached[source],k=20);nn=source[nn]
                        overlap=np.array([len(set(a)&set(b))/20 for a,b in zip(nn0,nn)])
                        shares0=np.stack([(fam[nn0]==h).mean(1) for h in np.unique(fam)],axis=1)
                        shares=np.stack([(fam[nn]==h).mean(1) for h in np.unique(fam)],axis=1)
                        tv=.5*np.abs(shares-shares0).sum(1)
                        # Retrieval identity against all original benchmark queries, allowing self.
                        closest,_=nearest(q,cached,k=1)
                        displacement=((q-base)**2).sum(1)
                        records[v]=dict(rows=len(ids),mean_cosine_to_reencoded_original=float((q*base).sum(1).mean()),
                            mean_squared_displacement=float(displacement.mean()),neighbor_overlap_k20=float(overlap.mean()),
                            source_family_total_variation=float(tv.mean()),identity_top1=float((closest[:,0]==ids).mean()))
                        key=f'{panel}|{rep}|{pool}|{g}|{v}'
                        saved[key+'|ids']=ids;saved[key+'|neighbors']=nn;saved[key+'|displacement']=displacement;saved[key+'|overlap']=overlap;saved[key+'|family_tv']=tv
                    records['reencoding_cosine_to_cache']=float((base*cached[ids]).sum(1).mean())
                    assert records['reencoding_cosine_to_cache']>.995, (panel,rep,pool,g,'Re-encoding mismatch')
                    res[meta['families'][int(g)]]=records
                panelout[rep+'/'+pool]=res;print(panel,rep,pool,'done',flush=True)
        out[panel]=panelout
    dest=DEST/'results';dest.mkdir(exist_ok=False)
    (dest/'result.json').write_text(json.dumps(out,indent=2));np.savez_compressed(dest/'paired.npz',**saved)
    md=['# Real-query presentation controls','',
        'Same query IDs in both spaces; at most 24 queries per family, original length ≤1536 tokens. No outcome-guided selection. Neighbor overlap compares changed and re-encoded original queries against all cached source queries, excluding the held-out family. Means weight families equally. No changed-prompt rewards are assumed.',
        'GTE strips trailing whitespace before tokenization: both whitespace variants have identical original token sequences in all 386 cases. Qwen preserves raw text. Therefore whitespace compares deployed preprocessing pipelines, not learned encoder invariance. Markers and headers change tokens in both pipelines. All cached arrays are cast to float32 before normalization; earlier float16-overflow results are archived and invalid.','',
        'Benchmark | Representation | Change | Original-query top1 | OOD neighbor overlap@20 | Source-family TV','---|---|---|---:|---:|---:']
    for panel,reps in out.items():
        for rep,folds in reps.items():
            for v in VARIANTS:
                a=[d[v] for d in folds.values()]
                md.append(f'{panel}|{rep}|{v}|'+ '|'.join(f'{np.mean([d[k] for d in a]):.3f}' for k in ('identity_top1','neighbor_overlap_k20','source_family_total_variation')))
    md+=['','Whitespace is the closest control to unchanged content. Markers and headers introduce framing, so they are interpreted separately. Pooling is part of the representation. This sensitivity test does not measure changed-prompt model outcomes or routing performance.']
    (dest/'summary.md').write_text('\n'.join(md)+'\n')

if __name__=='__main__':
    (prepare if sys.argv[1]=='prepare' else analyze)()
