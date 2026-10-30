"""Exact cosine retrieval, unique source questions and paired format interventions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CFG=json.loads((ROOT/'configs/format_factorial_v1.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def paths(enc):
 sub=Path(enc)/'layer_21' if enc=='qwen2.5-7b-instruct' else Path(enc)
 return ROOT/'artifacts/embeddings'/sub/'format_factorial_v1',ROOT/'results/geometry'/sub/'format_factorial_v1'
def interval(v):
 v=np.asarray(v);rng=np.random.default_rng(CFG['seed']);means=[]
 for _ in range(CFG['bootstrap_repetitions']):means.append(v[rng.integers(len(v),size=len(v))].mean())
 return dict(mean=float(v.mean()),ci95=np.quantile(means,[.025,.975]).tolist())
def analyze(enc):
 src,out=paths(enc);out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(l) for l in (ROOT/'datasets/format_factorial_v1/prepared/queries.jsonl').open()]
 meta=json.loads((src/'manifest.json').read_text())
 assert meta['query_sha256']==sha(ROOT/'datasets/format_factorial_v1/prepared/queries.jsonl')
 assert meta['vector_sha256']==sha(src/'vectors.npy')
 x=np.load(src/'vectors.npy');assert np.isfinite(x).all()
 x=x/np.linalg.norm(x,axis=1,keepdims=True);n=len(x);nb=n//4
 assert n==11400 and all(r['base']==i//4 and r['variant']==i%4 for i,r in enumerate(rows))
 topics=np.array([r['topic'] for r in rows[::4]])
 ids=np.empty((n,100),np.int32);vv=np.empty((n,100),np.int8)
 exact_ties=0
 for start in range(0,n,128):
  stop=min(start+128,n);sim=(x[start:stop]@x.T).reshape(stop-start,nb,4)
  choice=sim.argmax(2);best=sim.max(2)
  exact_ties+=int(((sim==best[:,:,None]).sum(2)>1).sum())
  best[np.arange(stop-start),np.arange(start,stop)//4]=-np.inf
  order=np.argsort(-best,axis=1,kind='stable')[:,:100]
  ids[start:stop]=order;vv[start:stop]=np.take_along_axis(choice,order,axis=1)
 assert not np.any(ids==np.arange(n)[:,None]//4)
 assert all(len(set(v))==100 for v in ids)
 result=dict(encoder=enc,base_questions=nb,prompts=n,subjects=len(set(topics)),random_topic_share=49/(nb-1),random_template_share=.5,random_answer_format_share=.5,exact_variant_similarity_ties=exact_ties,query_sha256=meta['query_sha256'],vector_sha256=meta['vector_sha256'],code_sha256=sha(__file__),metrics={},per_condition={},per_subject={})
 arrays=dict(neighbour_base=ids,neighbour_variant=vv)
 for k in CFG['ks']:
  metrics=dict(topic=(topics[ids[:,:k]]==topics[np.arange(n)//4,None]).mean(1),
   template=(vv[:,:k]//2==(np.arange(n)%4//2)[:,None]).mean(1),
   answer_format=(vv[:,:k]%2==(np.arange(n)%2)[:,None]).mean(1))
  for name,v in metrics.items():
   perbase=v.reshape(nb,4);arrays[f'{name}_{k}']=perbase.mean(1)
   result['metrics'][f'{name}_{k}']=interval(perbase.mean(1))
   result['per_condition'][f'{name}_{k}']=[float(perbase[:,i].mean()) for i in range(4)]
   result['per_subject'][f'{name}_{k}']={s:float(perbase[topics==s].mean()) for s in sorted(set(topics))}
  for name,pairs in [('change_template',[(0,2),(1,3)]),('change_answer_format',[(0,1),(2,3)])]:
   jacc=[]
   for b in range(nb):
    values=[]
    for a,c in pairs:
     inter=len(set(ids[4*b+a,:k])&set(ids[4*b+c,:k]))
     values.append(inter/(2*k-inter))
    jacc.append(np.mean(values))
   arrays[f'{name}_jaccard_{k}']=np.array(jacc);result['metrics'][f'{name}_jaccard_{k}']=interval(jacc)
 for name,pairs in [('change_template',[(0,2),(1,3)]),('change_answer_format',[(0,1),(2,3)])]:
  vals=np.mean([np.sum(x[a::4]*x[c::4],axis=1) for a,c in pairs],axis=0)
  arrays[f'{name}_cosine']=vals;result['metrics'][f'{name}_cosine']=interval(vals)
 np.savez_compressed(out/'per_question.npz',**arrays)
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print('ANALYZED',enc,result['metrics'],flush=True)
def compare():
 encs=['gte-qwen2-7b','qwen2.5-7b-instruct'];reports=[];values=[]
 for e in encs:
  _,p=paths(e);reports.append(json.loads((p/'report.json').read_text()));values.append(np.load(p/'per_question.npz'))
 assert reports[0]['query_sha256']==reports[1]['query_sha256']
 out=ROOT/'results/geometry/_diagnostics/format_factorial_v1';out.mkdir(parents=True,exist_ok=True)
 d={k:interval(values[1][k]-values[0][k]) for k in values[0].files if not k.startswith('neighbour')}
 (out/'comparison.json').write_text(json.dumps(dict(L21_minus_GTE=d,reports=reports),indent=2)+'\n')
 lines=['# Controlled prompt and answer format experiment','','Same questions and answer options; 57 official MMLU subjects, 50 questions each. Four crossed template/response variants per question. Exact cosine kNN; exclude all versions of the query itself; each reference question counts once, using its closest variant. Intervals resample original questions with all four variants together. Topic random share is49/2849; template and answer-format random shares are1/2.','','This tests requested letter versus answer-text response, not multiple-choice versus free-form solving. It is a within-MMLU geometry diagnostic, with no routing rewards or generated answers. Topic association is measured under balanced formats; topic itself is not experimentally changed.','','| Quantity | GTE | L21 |','|---|---:|---:|']
 for k in reports[0]['metrics']:
  lines.append('| '+k+' | '+' | '.join(f"{r['metrics'][k]['mean']:.6f}" for r in reports)+' |')
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 print('COMPARISON_COMPLETE')
if __name__=='__main__':
 if sys.argv[1]=='compare':compare()
 else:analyze(sys.argv[1])
