"""Outcome-free GraphRouter inputs. Does not modify prior description files."""
import json,sys,shutil
import numpy as np
from common import ROOT,E4,load,sha
TASKS={
 'gpqa':'GPQA: graduate-level multiple-choice science questions.',
 'math':'MATH: solve mathematics problems.',
 'mmlupro':'MMLU-Pro: multiple-choice questions across academic subjects.',
 'musr':'MuSR: reason about narrative scenarios.',
 'openhermes':'OpenHermes: respond to diverse instructions and conversational requests.',
 'ragbench':'RAGBench: answer questions using provided retrieved context.'}
def main():
 rb=E4/'results/routerbench_baselines_exp64/graphrouter_inputs';dest=ROOT/'inputs/graph_descriptions';dest.mkdir(exist_ok=False)
 out=dest/'routerbench-0shot';out.mkdir()
 for name in ['descriptions.json','descriptions.npz']:shutil.copy2(rb/name,out/name)
 d=load('sprout-sixfam');models=d['meta']['models'];families=d['meta']['families'];assert set(families)==set(TASKS)
 texts=['Language model: '+str(m).replace('/',' ').replace('-',' ')+'.' for m in models]+[TASKS[f] for f in families]
 sys.path.insert(0,str(E4/'scripts'));from embed_gpu import ST_MODELS,_register_optional_flash_attention_stub
 _register_optional_flash_attention_stub()
 import torch
 from sentence_transformers import SentenceTransformer
 path,max_seq,prompt,dtype,batch=ST_MODELS['gte-qwen2-7b']
 encoder=SentenceTransformer(path,trust_remote_code=True,device='cuda',model_kwargs={'torch_dtype':getattr(torch,dtype),'attn_implementation':'sdpa'})
 lengths=[len(encoder.tokenizer(t,add_special_tokens=True,truncation=False)['input_ids']) for t in texts]
 assert max(lengths)<max_seq
 encoder.max_seq_length=max_seq
 v=encoder.encode(texts,batch_size=batch,convert_to_numpy=True,normalize_embeddings=False)
 assert np.isfinite(v).all() and (np.linalg.norm(v,axis=1)>0).all()
 out=dest/'sprout-sixfam';out.mkdir();n=len(models)
 np.savez(out/'descriptions.npz',models=np.array(models),tasks=np.array(families),llm_embedding=v[:n].astype(np.float32),task_embedding=v[n:].astype(np.float32))
 (out/'descriptions.json').write_text(json.dumps(dict(models=models,tasks=families,model_texts=texts[:n],task_texts=texts[n:],tokens=lengths,embedder=path,description_source='Fixed model identifiers and outcome-free task definitions; native task description input explicitly disclosed',code_sha256=sha(__file__)),indent=2))
 print('DONE',v.shape,flush=True)
if __name__=='__main__':main()
