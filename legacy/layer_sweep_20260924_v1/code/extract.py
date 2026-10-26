"""Prefill hidden states of a frozen decoder at EVERY layer (hidden_states[0..L]), last-token and mean
pooling, float16 memmaps under features/<panel>/<reader>/L<layer>_<pool>.npy plus meta.json (written last,
so its presence marks a complete extraction). No outcome is read.

Replicates the two extractions the paper already uses, so the cached layers can be checked:
  sprout-sixfam      = ood_cost_geometry_20260921/scripts/extract_hidden_sprout.py: all 44,241 rows of
                       sprout/panel/rows.jsonl in file order, right padding, token-budget batches,
                       truncation at 8,192 (truncated rows are excluded downstream by panel_rows)
  routerbench-0shot  = scripts/extract_hidden_all_layers.py: the 34,661 filtered panel prompts in panel
                       order, left padding and left truncation, length-bucketed batches
Raw prompt text, no chat template, no special tokens, no EOS. Nothing is downloaded (HF_HUB_OFFLINE).
Readers with style 'gte' (GTE-Qwen2-7B-instruct's own layers) follow the RouterBench GTE extraction of
scripts/extract_hidden_all_layers.py on every panel: its remote (bidirectional) code, EOS appended, left padding
and truncation; --layers limits what is saved.
Usage: extract.py --reader qwen2.5-7b-instruct --panel sprout-sixfam
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
from common_ls import E4,ROOT,READERS,model_path

MAXLEN=8192;POOLS=('last','mean')
# already-saved layers of the same extraction, compared at the end
REFERENCE={('sprout-sixfam','qwen2.5-7b-instruct'):E4/'ood_cost_geometry_20260921/features/sprout/hidden/qwen2.5-7b-instruct'}

def texts_for(panel):
 if panel=='sprout-sixfam':
  return [json.loads(l)['prompt'] for l in open(E4/'sprout/panel/rows.jsonl')]
 sys.path.insert(0,str(E4/'src'));from rb.data import load_panel
 return [str(t) for t in load_panel(panel).text('prompt')]

def batches_sprout(lens,order,budget):
 n=len(order);pos=0
 while pos<n:
  bs=max(1,min(64,budget//max(int(lens[order[min(pos+63,n-1)]]),1)));yield order[pos:pos+bs];pos+=bs

def batches_routerbench(lens,order,big):
 n=len(order);s=0
 while s<n:
  L=int(lens[order[min(s+31,n-1)]])
  bs=(32 if L<=256 else 16 if L<=1024 else 4 if L<=4096 else 1) if big else (64 if L<=256 else 32 if L<=1024 else 8 if L<=4096 else 2)
  yield order[s:s+bs];s+=bs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reader',required=True,choices=list(READERS));ap.add_argument('--panel',required=True,choices=['sprout-sixfam','routerbench-0shot'])
 ap.add_argument('--token-budget',type=int,default=24000,help='SPROUT batch token budget (the original run used 24,000)');ap.add_argument('--overwrite',action='store_true')
 ap.add_argument('--layers',help='comma-separated hidden_states indices to save (default: all)');a=ap.parse_args()
 from transformers import AutoModel,AutoTokenizer
 if a.panel in READERS[a.reader]['cached']:sys.exit(f'{a.reader} on {a.panel} is already cached at {READERS[a.reader]["cached"][a.panel]}; nothing to do')
 out=ROOT/'features'/a.panel/a.reader
 if (out/'meta.json').exists() and not a.overwrite:sys.exit(f'{out} is complete; pass --overwrite to redo')
 out.mkdir(parents=True,exist_ok=True);path=model_path(a.reader);t0=time.time()
 texts=texts_for(a.panel);n=len(texts);gte=READERS[a.reader].get('style')=='gte';sprout=a.panel=='sprout-sixfam' and not gte
 if gte:   # GTE's remote module imports flash_attn optionally; the project's stub lets the SDPA path load
  sys.path.insert(0,str(E4/'scripts'));from embed_gpu import _register_optional_flash_attention_stub;_register_optional_flash_attention_stub()
 tok=AutoTokenizer.from_pretrained(path,trust_remote_code=gte);tok.pad_token=tok.pad_token or tok.eos_token
 tok.padding_side='right' if sprout else 'left';tok.truncation_side='right' if sprout else 'left'
 model=AutoModel.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='cuda',trust_remote_code=gte,**({} if gte else {'attn_implementation':'sdpa'})).eval()
 H=model.config.hidden_size;NL=model.config.num_hidden_layers+1;keep=sorted(int(v) for v in a.layers.split(',')) if a.layers else list(range(NL))
 eos=tok.eos_token_id if gte else None
 enc=[tok(t,add_special_tokens=False,truncation=True,max_length=MAXLEN-(1 if gte else 0))['input_ids'] for t in texts]
 if gte:enc=[e+([eos] if not e or e[-1]!=eos else []) for e in enc]
 lens=np.array([len(e) for e in enc]);order=np.argsort(lens)  # same (default) sort as both originals -> same batches
 full=np.array([len(tok(t,add_special_tokens=False)['input_ids']) for t in texts]);truncated=int((full>MAXLEN).sum())
 mm={(L,p):np.lib.format.open_memmap(out/f'L{L}_{p}.npy',mode='w+',dtype=np.float16,shape=(n,H)) for L in keep for p in POOLS}
 gen=batches_sprout(lens,order,a.token_budget) if sprout else batches_routerbench(lens,order,H>=3584);done=0;nb=0
 for idx in gen:
  idx=np.asarray(idx);mx=int(lens[idx].max())
  ids=torch.full((len(idx),mx),tok.pad_token_id,dtype=torch.long);att=torch.zeros((len(idx),mx),dtype=torch.long)
  for j,i in enumerate(idx):
   e=torch.tensor(enc[i])
   if sprout:ids[j,:len(e)]=e;att[j,:len(e)]=1
   else:ids[j,mx-len(e):]=e;att[j,mx-len(e):]=1
  ids,att=ids.cuda(),att.cuda()
  with torch.no_grad():hs=model(input_ids=ids,attention_mask=att,output_hidden_states=True,use_cache=False).hidden_states
  assert len(hs)==NL,(len(hs),NL)
  m=att.unsqueeze(-1).float();last_pos=(att.sum(1)-1) if sprout else torch.full((len(idx),),mx-1,device='cuda')
  for L in keep:
   h=hs[L].float()
   mm[(L,'last')][idx]=h[torch.arange(len(idx),device='cuda'),last_pos].cpu().numpy().astype(np.float16)
   mm[(L,'mean')][idx]=((h*m).sum(1)/m.sum(1)).cpu().numpy().astype(np.float16)
  done+=len(idx);nb+=1
  if nb%50==0 or done==n:print(f'{done}/{n} {time.time()-t0:.0f}s len {mx}',flush=True)
  del hs
 for v in mm.values():v.flush()
 check={}
 ref=REFERENCE.get((a.panel,a.reader))
 if ref:
  for f in sorted(ref.glob('L*_*.npy')):
   L,p=f.stem[1:].split('_');mine=np.asarray(mm[(int(L),p)],np.float32);theirs=np.asarray(np.load(f,mmap_mode='r'),np.float32)
   ok=np.isfinite(mine).all(1)&np.isfinite(theirs).all(1)&(full<=MAXLEN)
   cos=(mine[ok]*theirs[ok]).sum(1)/(np.linalg.norm(mine[ok],axis=1)*np.linalg.norm(theirs[ok],axis=1))
   check[f.stem]=dict(rows=int(ok.sum()),max_abs_diff=float(np.abs(mine[ok]-theirs[ok]).max()),cosine_min=float(cos.min()),cosine_mean=float(cos.mean()))
   print('check vs cached',f.stem,check[f.stem],flush=True)
 json.dump(dict(panel=a.panel,reader=a.reader,path=path,rows=n,layers=NL,hidden_size=H,poolings=list(POOLS),max_len=MAXLEN,
  padding=tok.padding_side,truncation=tok.truncation_side,prefix='',append_eos=gte,layers_saved=keep,chat_template=False,texts_truncated=truncated,
  token_budget=a.token_budget if sprout else None,dtype='float16',check_vs_cached=check,seconds=time.time()-t0),open(out/'meta.json','w'),indent=1)
 print('done',out,round(time.time()-t0),flush=True)

if __name__=='__main__':main()
