"""Freeze row alignment and exclude every prompt exceeding either encoder context."""
from pathlib import Path
import os,json,hashlib
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;E4=Path(os.environ['E4_ROOT']);SP=E4/'sprout'
MODEL=E4.parent/'artifacts/models/Qwen2.5-7B-Instruct__a09a3545'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    dest=ROOT/'inputs';dest.mkdir(exist_ok=False)
    rows=[json.loads(l) for l in open(SP/'panel/rows.jsonl')];meta=json.load(open(SP/'panel/panel.json'))
    mask=np.load(SP/'features/gte-qwen2-7b_rows.npz');gtekeep=mask['keep'].astype(bool)
    assert len(rows)==meta['rows'] and np.array_equal(mask['key'],np.asarray([r['key'] for r in rows]))
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    lengths=[]
    for a in range(0,len(rows),128):
        batch=tok([r['prompt'] for r in rows[a:a+128]],add_special_tokens=False,truncation=False,return_length=True)
        lengths.extend(batch['length'])
    lengths=np.asarray(lengths);qkeep=(lengths>0)&(lengths<=8192);keep=gtekeep&qkeep
    ids=[];seen=set()
    for i in np.flatnonzero(keep):
        s=rows[i]['prompt']
        if s in seen:continue
        seen.add(s);ids.append(i)
    ids=np.asarray(ids);gteids=np.full(len(rows),-1,dtype=np.int64);gteids[gtekeep]=np.arange(gtekeep.sum())
    z=np.load(SP/'panel/sprout.npz');r=z['rewards'][ids];c=z['costs'][ids]
    assert r.shape==c.shape==(len(ids),len(meta['models']))
    assert np.isfinite(r).all() and np.isfinite(c).all() and (c>=0).all() and (r>=0).all() and (r<=1).all()
    fn=sorted(set(rows[i]['family'] for i in ids));assert fn==['gpqa','math','mmlupro','musr','openhermes','ragbench']
    fam=np.asarray([fn.index(rows[i]['family']) for i in ids]);keys=np.asarray([str(rows[i]['key']) for i in ids])
    assert len(set(keys))==len(ids)
    readerdir=E4/'ood_cost_geometry_20260921/features/sprout/hidden/qwen2.5-7b-instruct'
    reps={'gte':SP/'features/gte-qwen2-7b.npy','reader':readerdir/'L21_last.npy','reader_mean':readerdir/'L21_mean.npy'}
    refs={}
    for name,p in reps.items():
        a=np.load(p,mmap_mode='r');assert a.shape[0]==(gtekeep.sum() if name=='gte' else len(rows))
        sel=gteids[ids] if name=='gte' else ids
        for start in range(0,len(sel),2048):
            v=np.asarray(a[sel[start:start+2048]],np.float32);norm=np.linalg.norm(v,axis=1)
            assert np.isfinite(v).all() and np.isfinite(norm).all() and (norm>0).all()
        st=p.stat();refs[name]=dict(path=str(p),shape=list(a.shape),dtype=str(a.dtype),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p))
    np.savez_compressed(dest/'panel.npz',r=r,c=c,fam=fam,query_ids=keys,panel_rows=ids,gte_rows=gteids[ids],reader_tokens=lengths[ids])
    # Preserve prompts for extraction checks only; no responses or answers.
    with open(dest/'queries.jsonl','w') as f:
        for i in ids:f.write(json.dumps({'row':int(i),'key':str(rows[i]['key']),'family':rows[i]['family'],'prompt':rows[i]['prompt']})+'\n')
    split_counts={s:{f:sum(rows[i]['split']==s and rows[i]['family']==f for i in ids) for f in fn} for s in ['train','validation','test']}
    info=dict(panel='sprout-sixfam',rows=len(ids),original_rows=len(rows),families=fn,models=meta['models'],
        family_counts={f:int((fam==g).sum()) for g,f in enumerate(fn)},representations=refs,
        excluded_gte=int((~gtekeep).sum()),excluded_reader=int((~qkeep).sum()),additional_reader_exclusions=int((gtekeep&~qkeep).sum()),
        duplicate_prompts_removed=int(keep.sum()-len(ids)),release_split_family_counts=split_counts,
        reward_source_sha256=meta['reward_sha256'],cost_source_sha256=meta['cost_sha256'],
        reward_definition='Original released score, unchanged; no thresholding for utility',cost_definition=meta['cost_formula'],
        source_rows_sha256=sha(SP/'panel/rows.jsonl'),snapshot_sha256=sha(dest/'panel.npz'),
        contexts='GTE frozen keep mask AND Qwen raw prompt tokens <=8192; no truncated row retained',
        reader_extraction_meta=json.load(open(readerdir/'meta.json')))
    (dest/'manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:v for k,v in info.items() if k not in ['representations','models','reader_extraction_meta']},indent=2),flush=True)
if __name__=='__main__':main()
