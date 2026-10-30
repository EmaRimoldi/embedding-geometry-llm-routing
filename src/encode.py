"""Frozen representations of complete prompts. Resumable; no outcomes read."""
import argparse,json,time
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel,AutoConfig
import importlib.util,sys
from common import ROOT,CONFIG,sha,write_json

def main(enc,dataset):
 torch.set_num_threads(4)
 assert torch.cuda.is_available(),'Encoder extraction requires an allocated GPU'
 modelpath=ROOT/'embedders'/enc
 tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,trust_remote_code=enc=='gte-qwen2-7b')
 tok.padding_side='left';tok.pad_token=tok.pad_token or tok.eos_token
 model_class=AutoModel
 if enc=='gte-qwen2-7b':
  spec=importlib.util.spec_from_file_location('published_gte_model',modelpath/'modeling_qwen.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);model_class=module.Qwen2Model
 model=model_class.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32 if enc=='minilm-l12' else torch.bfloat16,attn_implementation='sdpa').cuda().eval()
 if enc=='gte-qwen2-7b':assert model.config.is_causal is False,'GTE must be bidirectional'
 layers=CONFIG['qwen_layers'] if enc=='qwen2.5-7b-instruct' else [None]
 rows=[json.loads(s) for s in (ROOT/'datasets'/dataset/'prepared/queries.jsonl').open()]
 n=len(rows);dim=model.config.hidden_size;seqs=[]
 for start in range(0,n,128):
  seqs+=tok([r['prompt'] for r in rows[start:start+128]],add_special_tokens=enc=='minilm-l12',truncation=False)['input_ids']
 if enc=='gte-qwen2-7b':seqs=[s if s and s[-1]==tok.eos_token_id else s+[tok.eos_token_id] for s in seqs]
 lengths=np.array([len(s) for s in seqs]);limit=512 if enc=='minilm-l12' else 8192;eligible=(lengths>0)&(lengths<=limit)
 if enc!='minilm-l12':assert eligible.all()
 targets={};outroot=ROOT/'artifacts/embeddings'/enc
 for layer in layers:
  p=(outroot/f'layer_{layer}'/dataset) if layer is not None else (outroot/dataset)
  p.mkdir(parents=True,exist_ok=True)
  if (p/'manifest.json').exists():continue
  vpath=p/'vectors.npy';donepath=p/'done.npy'
  if vpath.exists():
   vectors=np.load(vpath,mmap_mode='r+');done=np.load(donepath);assert vectors.shape==(n,dim)
  else:
   vectors=np.lib.format.open_memmap(vpath,mode='w+',dtype=np.float32,shape=(n,dim));vectors[:]=np.nan;vectors.flush();done=np.zeros(n,bool);np.save(donepath,done)
  targets[layer]=(p,vectors,done)
 if not targets:print('already complete',enc,dataset);return
 todo=np.flatnonzero(eligible & ~np.logical_and.reduce([v[2] for v in targets.values()]))
 order=todo[np.argsort(lengths[todo],kind='stable')];start=0;t0=time.time();batches=0
 while start<len(order):
  end=start+1
  while end<len(order) and end-start<16 and (end-start+1)*lengths[order[end]]<=8192:end+=1
  ix=order[start:end];batch=tok.pad({'input_ids':[seqs[i] for i in ix]},padding=True,return_tensors='pt');batch={k:v.cuda() for k,v in batch.items()}
  with torch.inference_mode():
   out=model(**batch,output_hidden_states=enc=='qwen2.5-7b-instruct',return_dict=True,**({'use_cache':False} if enc!='minilm-l12' else {}))
   for layer,(p,vectors,done) in targets.items():
    h=out.hidden_states[layer] if layer is not None else out.last_hidden_state
    if enc=='minilm-l12':
     mask=batch['attention_mask'].unsqueeze(-1);pooled=(h.float()*mask).sum(1)/mask.sum(1)
    else:pooled=h[:,-1,:].float()
    a=pooled.cpu().numpy();assert np.isfinite(a).all() and (np.linalg.norm(a,axis=1)>0).all()
    vectors[ix]=a;done[ix]=True
   del out
  start=end;batches+=1
  if batches%100==0 or start==len(order):
   for p,vectors,done in targets.values():vectors.flush();np.save(p/'done.npy',done)
   print(enc,dataset,'complete',start,'of',len(order),'seconds',round(time.time()-t0),flush=True)
 for layer,(p,vectors,done) in targets.items():
  assert done[eligible].all();vectors.flush();np.save(p/'done.npy',done)
  write_json(p/'manifest.json',dict(encoder=enc,layer=layer,dataset=dataset,rows=n,valid_rows=int(eligible.sum()),dimension=dim,pooling='attention_mask_mean' if enc=='minilm-l12' else 'last_nonpadding',max_tokens=limit,truncation=False,chat_template=False,append_eos=enc=='gte-qwen2-7b',dtype_storage='float32',model_dtype=str(next(model.parameters()).dtype),tokens_max=int(lengths[eligible].max()),vector_sha256=sha(p/'vectors.npy'),query_sha256=sha(ROOT/'datasets'/dataset/'prepared/queries.jsonl'),code_sha256=sha(__file__)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);p.add_argument('--dataset',required=True);a=p.parse_args();main(a.encoder,a.dataset)
