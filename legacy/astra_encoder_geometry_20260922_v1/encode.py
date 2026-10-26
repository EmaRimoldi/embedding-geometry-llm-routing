"""Encode factorial controls using local frozen models, with no truncation."""
from pathlib import Path
import ast, hashlib, json, os, sys, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent; E4=Path(os.environ['E4_ROOT'])
FROZEN=E4/'astra_reward_transfer_20260922_v1/inputs'
MODELS=E4.parent/'artifacts/models'

def main():
    rep=sys.argv[1]; rows=[json.loads(l) for l in (ROOT/'inputs/controlled.jsonl').read_text().splitlines()]
    texts=[r['text'] for r in rows]; dest=ROOT/'encoded'/rep; dest.mkdir(parents=True,exist_ok=False)
    t=time.time(); model_path=MODELS/('gte-Qwen2-7B-instruct' if rep=='gte' else 'Qwen2.5-7B-Instruct__a09a3545')
    if rep=='gte':
        src=ROOT/'vendor/embed_gpu_snapshot.py'; tree=ast.parse(src.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_register_optional_flash_attention_stub')
        ns={'sys':sys}; exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
        ns['_register_optional_flash_attention_stub']()
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(str(model_path),trust_remote_code=True,device='cuda',
            model_kwargs={'torch_dtype':torch.bfloat16,'attn_implementation':'sdpa'})
        model.max_seq_length=8192
        lengths=np.array([len(model.tokenizer(s,add_special_tokens=True,truncation=False)['input_ids']) for s in texts])
        assert np.all(lengths+1<=8192), 'Never truncate controlled queries'
        out=model.encode(texts,batch_size=16,show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=False)
        np.save(dest/'primary.npy',np.asarray(out,np.float32))
    else:
        from transformers import AutoTokenizer,AutoModel
        tok=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        tok.padding_side='left'; tok.pad_token=tok.pad_token or tok.eos_token
        ids=tok(texts,truncation=False,add_special_tokens=False)['input_ids']
        lengths=np.asarray(list(map(len,ids))); assert np.all(lengths<=8192)
        model=AutoModel.from_pretrained(str(model_path),local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        dim=model.config.hidden_size; n=len(rows)
        mm={p:np.lib.format.open_memmap(dest/f'{p}.npy',mode='w+',dtype=np.float32,shape=(n,dim))
            for p in ('primary','mean','last8')}
        order=np.argsort(lengths)
        for start in range(0,n,16):
            ix=order[start:start+16]; L=max(lengths[ix]); batch=[ids[i] for i in ix]
            inp=torch.tensor([[tok.pad_token_id]*(L-len(s))+s for s in batch],device='cuda')
            mask=torch.tensor([[0]*(L-len(s))+[1]*len(s) for s in batch],device='cuda')
            with torch.inference_mode(): hs=model(input_ids=inp,attention_mask=mask,output_hidden_states=True,use_cache=False).hidden_states[21].float()
            m=mask.unsqueeze(-1).float(); last8=m.clone(); last8[:,:-8]=0
            mm['primary'][ix]=hs[:,-1].cpu().numpy()
            mm['mean'][ix]=((hs*m).sum(1)/m.sum(1)).cpu().numpy()
            mm['last8'][ix]=((hs*last8).sum(1)/last8.sum(1)).cpu().numpy()
            if start%512==0: print(rep,start,n,round(time.time()-t),flush=True)
        for v in mm.values():v.flush()
        out=np.load(dest/'primary.npy',mmap_mode='r')
    np.save(dest/'tokens.npy',lengths)
    checks={}
    for panel in ('routerbench-0shot','llmrouterbench-performance-cost'):
        ix=[i for i,r in enumerate(rows) if r['kind']=='canary' and r['panel']==panel]
        src=np.load(FROZEN/panel/f'{rep}.npy',mmap_mode='r')
        a=np.asarray(out[ix]); b=np.asarray(src[[rows[i]['row'] for i in ix]])
        cos=(a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        checks[panel]=dict(mean_cosine=float(cos.mean()),min_cosine=float(cos.min()),rows=len(ix))
    passed=all(v['mean_cosine']>.995 and v['min_cosine']>.98 for v in checks.values())
    meta=dict(rep=rep,model=str(model_path),layer=21 if rep=='reader' else 'emitted',
        rows=len(rows),truncated=0,max_tokens=int(lengths.max()),canary_checks=checks,
        canary_pass=passed,seconds=time.time()-t,
        corpus_sha256=hashlib.sha256((ROOT/'inputs/controlled.jsonl').read_bytes()).hexdigest())
    (dest/'manifest.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2),flush=True)
    assert passed,'Encoding canaries differ from the frozen benchmark protocol; inspect before analysis'

if __name__=='__main__':main()
