"""Read-only scientific audit of the frozen controlled-format experiment."""
from pathlib import Path
import json,hashlib,collections,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/diagnostics'))
from format_factorial_analyze import interval
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
out=ROOT/'results/geometry/_diagnostics/format_factorial_v1'
frozen=read(out/'source_hashes.json')
for p,h in frozen.items():assert sha(ROOT/p)==h,p
prep=ROOT/'datasets/format_factorial_v1/prepared'
m=read(prep/'manifest.json')
assert sha(ROOT/'datasets/routerbench/prepared/queries.jsonl')==m['source_sha256']
assert sha(prep/'queries.jsonl')==m['queries_sha256']
assert sha(ROOT/'configs/format_factorial_v1.json')==m['config_sha256']
base=[json.loads(l) for l in (prep/'questions.jsonl').open()]
rows=[json.loads(l) for l in (prep/'queries.jsonl').open()]
source=[json.loads(l) for l in (ROOT/'datasets/routerbench/prepared/queries.jsonl').open()]
groups=collections.defaultdict(list)
for r in source:
 if r['family']=='mmlu':groups[r['topic']].append(r)
expected=[]
for t,rs in sorted(groups.items()):
 expected.extend(sorted(rs,key=lambda r:hashlib.sha256(f"20261002|{r['key']}".encode()).hexdigest())[:50])
assert len(base)==2850 and len(rows)==11400 and len(groups)==57
prefix='Please answer with the letter of the correct answer.\n\n'
suffix='\nPrint only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:'
for i,(b,r) in enumerate(zip(base,expected)):
 assert b['key']==r['key'] and b['topic']==r['topic'] and b['base']==i
 assert r['prompt']==prefix+b['body']+suffix and b['stem']+b['options']==b['body']
 for v in range(4):
  q=rows[4*i+v];t,a=divmod(v,2)
  ins=['Return only the letter of the correct option, without explanation.','Return only the exact text of the correct option, without its letter or explanation.'][a]
  p=b['body']+'\n\n'+ins if t==0 else 'The following is a multiple-choice question.\n\nQuestion:\n'+b['stem']+'\nOptions:\n'+b['options']+'\n\n'+ins+'\nAnswer:'
  assert q['prompt']==p and q['base']==i and q['variant']==v and q['source_key']==b['key'] and q['topic']==b['topic']
topics=np.array([b['topic'] for b in base]);n=len(rows)
comparison=read(out/'comparison.json');saved=[]
audit={'sources_verified':len(frozen),'base_questions':2850,'variants':11400,'encoders':{},'audit_code_sha256':sha(__file__)}
for enc in ['gte-qwen2-7b','qwen2.5-7b-instruct/layer_21']:
 src=ROOT/'artifacts/embeddings'/enc/'format_factorial_v1'
 dest=ROOT/'results/geometry'/enc/'format_factorial_v1'
 em=read(src/'manifest.json');r=read(dest/'report.json');z=np.load(dest/'per_question.npz')
 assert em['query_sha256']==m['queries_sha256']==r['query_sha256']
 assert em['vector_sha256']==sha(src/'vectors.npy')==r['vector_sha256']
 assert em['code_sha256']==frozen['src/encode.py'] and r['code_sha256']==frozen['src/diagnostics/format_factorial_analyze.py']
 x=np.load(src/'vectors.npy');assert x.shape==(n,3584) and np.isfinite(x).all()
 norms=np.linalg.norm(x,axis=1,keepdims=True);assert np.all(norms>0);x=x/norms
 ids=z['neighbour_base'];vv=z['neighbour_variant']
 assert ids.shape==vv.shape==(n,100)
 assert not np.any(ids==np.arange(n)[:,None]//4)
 assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
 assert ids.min()>=0 and ids.max()<2850 and vv.min()>=0 and vv.max()<4
 selected_ties=0
 for lo in range(0,n,128):
  hi=min(lo+128,n);s=(x[lo:hi]@x.T).reshape(hi-lo,2850,4)
  best=s.max(axis=2);best[np.arange(hi-lo),np.arange(lo,hi)//4]=-np.inf
  expected_ids=np.argsort(-best,axis=1,kind='stable')[:,:100]
  expected_v=np.take_along_axis(s.argmax(axis=2),expected_ids,axis=1)
  assert np.array_equal(ids[lo:hi],expected_ids) and np.array_equal(vv[lo:hi],expected_v)
  chosen=np.take_along_axis(s,expected_ids[:,:,None],axis=1)
  selected_ties+=int(((chosen==chosen.max(2)[:,:,None]).sum(2)>1).sum())
 vals={}
 for k in [10,100]:
  perrow={'topic':(topics[ids[:,:k]]==topics[np.arange(n)//4,None]).mean(1),
          'template':(vv[:,:k]//2==(np.arange(n)%4//2)[:,None]).mean(1),
          'answer_format':(vv[:,:k]%2==(np.arange(n)%2)[:,None]).mean(1)}
  for name,v in perrow.items():
   key=f'{name}_{k}';p=v.reshape(2850,4);vals[key]=p.mean(1)
   assert np.allclose(r['per_condition'][key],p.mean(0),atol=1e-12)
   for t in groups:assert abs(r['per_subject'][key][t]-p[topics==t].mean())<1e-12
  for name,pairs in [('change_template',[(0,2),(1,3)]),('change_answer_format',[(0,1),(2,3)])]:
   vals[f'{name}_jaccard_{k}']=np.array([np.mean([len(set(ids[4*b+a,:k])&set(ids[4*b+c,:k]))/len(set(ids[4*b+a,:k])|set(ids[4*b+c,:k])) for a,c in pairs]) for b in range(2850)])
 for name,pairs in [('change_template',[(0,2),(1,3)]),('change_answer_format',[(0,1),(2,3)])]:
  vals[f'{name}_cosine']=np.mean([np.sum(x[a::4]*x[c::4],axis=1) for a,c in pairs],axis=0)
 for key,v in vals.items():
  assert np.allclose(z[key],v,atol=1e-12),key
  check=interval(v)
  assert np.allclose(check['ci95'],r['metrics'][key]['ci95'],atol=1e-12)
  assert abs(check['mean']-r['metrics'][key]['mean'])<1e-12
 saved.append(vals)
 audit['encoders'][enc]={'all_11400_retrieval_rows_reconstructed':True,'finite_vectors':True,'unique_neighbours_and_self_exclusion':True,'metrics_and_bootstrap_reconstructed':True,'selected_variant_exact_ties':selected_ties,'queries_sha256':em['query_sha256'],'vectors_sha256':em['vector_sha256']}
for key in saved[0]:
 check=interval(saved[1][key]-saved[0][key]);old=comparison['L21_minus_GTE'][key]
 assert np.allclose(check['ci95'],old['ci95'],atol=1e-12) and abs(check['mean']-old['mean'])<1e-12
audit['paired_alignment_and_intervals_verified']=True
audit['status']='passed'
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
