"""Build fresh, auditable panels from pinned public releases; no generated labels."""
import ast,collections,json,re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from common import ROOT,sha,write_json

def norm(s):return re.sub(r'\s+',' ',str(s)).strip()
def routerbench():
 path=ROOT/'datasets/routerbench/raw/routerbench_0shot.pkl'
 d=pd.read_pickle(path) # Official, checksum-verified upstream pickle, not user-supplied code.
 models=sorted(c for c in d.columns if c+'|total_cost' in d.columns)
 rows=[];skip=[];rewards=[];costs=[]
 for i,r in d.iterrows():
  name=r['eval_name'];family='mmlu' if name.startswith('mmlu-') else ('mtbench' if name.startswith('mtbench') else name)
  if family not in ['mmlu','mtbench','arc-challenge','grade-school-math','hellaswag','mbpp','winogrande']:
   skip.append(dict(key=str(r['sample_id']),reason='outside_seven_paper_families'));continue
  v=ast.literal_eval(r['prompt']);assert isinstance(v,list) and all(isinstance(t,str) for t in v)
  rows.append(dict(key=str(r['sample_id']),prompt='\n\n'.join(v),family=family,topic=name[5:] if family=='mmlu' else '',subgroup=name if family in ['mmlu','mtbench'] else family,topic_source='released_eval_name' if family=='mmlu' else '',release_split='',format='choice' if family in ['mmlu','arc-challenge','hellaswag','winogrande'] else 'free',human_level='',prompt_parts=len(v)))
  rewards.append([r[m] for m in models]);costs.append([r[m+'|total_cost'] for m in models])
 return rows,np.asarray(rewards,float),np.asarray(costs,float),models,skip,{'release_rows':len(d),'cost':'Released total_cost in USD'}

def sprout():
 base=ROOT/'datasets/sprout'; files=sorted((base/'raw/data').glob('*.parquet'),key=lambda p:({'train':0,'validation':1,'test':2}[p.name.split('-')[0]],p.name))
 mmlu=pq.read_table(next((base/'source_metadata/mmlupro/data').glob('test*.parquet'))).to_pylist()
 math={}
 for p in sorted((base/'source_metadata/math').glob('*/*.parquet')):
  for r in pq.read_table(p).to_pylist():math.setdefault(norm(r['problem']),set()).add(str(r.get('type',p.parent.name)))
 prices={'aws-claude-3-5-sonnet-v1':(3,15),'aws-titan-text-premier-v1':(.5,1.5),'openai-gpt-4o':(2.5,10),'openai-gpt-4o-mini':(.15,.6),'wxai-granite-3-2b-instruct-8k-max-tokens':(.1,.1),'wxai-granite-3-8b-instruct-8k-max-tokens':(.2,.2),'wxai-llama-3-1-70b-instruct':(.9,.9),'wxai-llama-3-1-8b-instruct':(.2,.2),'wxai-llama-3-2-1b-instruct':(.06,.06),'wxai-llama-3-2-3b-instruct':(.06,.06),'wxai-llama-3-3-70b-instruct':(.9,.9),'wxai-llama-3-405b-instruct':(3.5,3.5),'wxai-mixtral-8x7b-instruct-v01':(.6,.6)}
 models=sorted(prices);rows=[];rewards=[];costs=[];joins=collections.Counter()
 for p in files:
  table=pq.read_table(p)
  assert set(models).issubset(table.column_names)
  for r in table.to_pylist():
   ds=r['dataset'];f=('mmlupro' if ds.startswith('TIGER-Lab') else 'math' if ds.startswith('lighteval/MATH') else 'musr' if ds.startswith('TAUR-Lab') else 'ragbench' if ds.startswith('rungalileo') else 'openhermes' if ds.startswith('openhermes') else 'gpqa' if 'gpqa' in ds else None)
   assert f is not None,ds
   text=r['prompt'];t='';src=''
   if f=='mmlupro':
    j=int(r['dataset_idx'])
    if 0<=j<len(mmlu) and norm(mmlu[j]['question']) in norm(text):t=mmlu[j]['category'];src='verified_original_question'
   if f=='math':
    vals=math.get(norm(text),set())
    if len(vals)==1:t=next(iter(vals));src='exact_normalized_original_question'
   subgroup=f+'::'+t if t else ds if f in ['musr','ragbench'] else f
   joins[f+('::matched' if t else '::no_topic')]+=1
   rows.append(dict(key=r['key'],prompt=text,family=f,topic=t,subgroup=subgroup,topic_source=src,release_dataset=ds,dataset_idx=r['dataset_idx'],release_split=p.name.split('-')[0],format='choice' if f in ['mmlupro','gpqa','musr'] else 'free',human_level=r['dataset_level'],prompt_parts=1))
   rewards.append([r[m]['score'] for m in models]);costs.append([(r[m]['num_input_tokens']*prices[m][0]+r[m]['num_output_tokens']*prices[m][1])/1e6 for m in models])
 return rows,np.asarray(rewards,float),np.asarray(costs,float),models,[],dict(release_rows=len(rows),native_subject_joins=dict(joins),prices_usd_per_million=prices,price_source='https://huggingface.co/CARROT-LLM-Routing',cost='(released input tokens * input price + released output tokens * output price) / 1e6')

def main(name):
 out=ROOT/'datasets'/name/'prepared';out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise RuntimeError('Prepared data already frozen; do not overwrite')
 rows,r,c,models,excluded,info=routerbench() if name=='routerbench' else sprout()
 lengths={}
 for enc in ['gte-qwen2-7b','qwen2.5-7b-instruct','minilm-l12']:
  tok=AutoTokenizer.from_pretrained(ROOT/'embedders'/enc,local_files_only=True,trust_remote_code=enc=='gte-qwen2-7b')
  ll=[]
  for start in range(0,len(rows),128):
   ids=tok([x['prompt'] for x in rows[start:start+128]],add_special_tokens=enc=='minilm-l12',truncation=False)['input_ids']
   ll.extend(len(v)+(1 if enc=='gte-qwen2-7b' and (not v or v[-1]!=tok.eos_token_id) else 0) for v in ids)
  lengths[enc]=np.asarray(ll)
 seen=set();keep=[]
 for i,row in enumerate(rows):
  reason=None
  if not row['prompt'].strip():reason='empty_prompt'
  elif not (np.isfinite(r[i]).all() and np.isfinite(c[i]).all() and (c[i]>=0).all() and ((r[i]>=0)&(r[i]<=1)).all()):reason='invalid_outcome'
  elif max(lengths[e][i] for e in ['gte-qwen2-7b','qwen2.5-7b-instruct'])>8192:reason='over_8192_tokens'
  elif row['prompt'] in seen:reason='exact_duplicate_prompt'
  if reason:excluded.append(dict(key=row['key'],reason=reason));continue
  keep.append(i);seen.add(row['prompt'])
 ix=np.asarray(keep);rows=[rows[i] for i in ix]
 assert len({v['key'] for v in rows})==len(rows)
 for row,i in zip(rows,ix):row['tokens']={e:int(a[i]) for e,a in lengths.items()}
 (out/'queries.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 (out/'excluded.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in excluded))
 np.savez_compressed(out/'panel.npz',reward=r[ix],cost=c[ix],family=np.asarray([v['family'] for v in rows]),topic=np.asarray([v['topic'] for v in rows]),subgroup=np.asarray([v['subgroup'] for v in rows]),format=np.asarray([v['format'] for v in rows]),keys=np.asarray([v['key'] for v in rows]),minilm_keep=lengths['minilm-l12'][ix]<=512)
 info.update(rows=len(rows),models=models,family_counts=dict(collections.Counter(v['family'] for v in rows)),exclusions=dict(collections.Counter(v['reason'] for v in excluded)),shared_minilm_rows=int((lengths['minilm-l12'][ix]<=512).sum()),panel_sha256=sha(out/'panel.npz'),queries_sha256=sha(out/'queries.jsonl'),prepare_code_sha256=sha(Path(__file__)),config_sha256=sha(ROOT/'configs/geometry.json'),raw_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'datasets'/name/'raw').rglob('*')) if p.is_file() and '.cache' not in p.parts})
 write_json(out/'manifest.json',info);print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1])
