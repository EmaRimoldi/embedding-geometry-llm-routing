"""Audit released vs generated SPROUT cells on the frozen common cohort.
Label construction uses metadata/query text only. All candidate results retained.
Historical label files and the other agent's experiment results are not modified.
"""
import json,re
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from common import ROOT,E4,load,score,sha

def read(path):
 rows=[json.loads(l) for l in open(path) if l.strip()];assert len({r['key'] for r in rows})==len(rows);return {r['key']:r for r in rows}
def norm(s):return re.sub(r'\s+',' ',str(s)).strip()

def build(d):
 root=E4/'sprout';ap=root/'topics/authoritative_and_pending.jsonl';rp=root/'panel/rows.jsonl';vp=E4/'ood_cost_geometry_20260921/results/sprout_labels/topics_v2.jsonl'
 raw,auth,v2=read(rp),read(ap),read(vp);assert set(raw)==set(auth)==set(v2)
 srcpath=root/'data/source/mmlu-pro/data/test-00000-of-00001.parquet';mmlu=pq.read_table(srcpath,columns=['question','category']).to_pylist()
 variants={k:[] for k in ['family','released_source','native_subjects','native_metadata_cells','native_plus_qwen_openhermes','native_plus_qwen_missing','existing_v2_mixed']}
 details=[];mismatches=[]
 for key,g in zip(d['query_ids'],d['fam']):
  r,a,v=raw[str(key)],auth[str(key)],v2[str(key)];fam=d['meta']['families'][g]
  assert r['family']==a['family']==v['family']==fam and r['prompt']==a['prompt']
  subject=None;native=None;verified=True
  if fam=='mmlupro':
   s=mmlu[int(r['dataset_idx'])];verified=norm(s['question']) in norm(r['prompt'])
   if verified:
    native='subject:'+s['category'];subject=native if s['category'].lower()!='other' else None
   else:mismatches.append(dict(key=str(key),dataset_idx=r['dataset_idx'],family=fam,reason='Source question not an exact normalized substring; retain family fallback'))
  elif fam=='math':
   subject='subject:'+a['topic'] if a['topic'] else None;native=subject
  elif fam=='musr':native='task:'+r['dataset'].rsplit('/',1)[-1] if r['dataset']!='TAUR-Lab/MuSR' else 'task:murder_mysteries'
  elif fam=='ragbench':native='source:'+r['dataset'].rsplit('/',1)[-1]
  # v2 sub-topics are collapsed only in explicitly broad Qwen variants.
  generated=v['topic_v2'].split('/')[0] if v['source_v2'].startswith('qwen') else None
  prefix=fam+'::';base=prefix+(native or 'family_fallback');sub=prefix+(subject or 'family_fallback')
  vals=dict(family=fam,released_source=r['dataset'],native_subjects=sub,native_metadata_cells=base,
   native_plus_qwen_openhermes=prefix+'qwen:'+generated if fam=='openhermes' and generated else base,
   native_plus_qwen_missing=prefix+'qwen:'+generated if native is None and generated else base,
   existing_v2_mixed=prefix+v['topic_v2'])
  for name,value in vals.items():variants[name].append(value)
  details.append(dict(key=str(key),family=fam,raw_dataset=r['dataset'],native_subject=subject,native_cell=native,
   original_topic=a['topic'],original_verification=a['verification'],source_question_verified=verified,
   generated_broad=generated,v2_source=v['source_v2'],v2_topic=v['topic_v2']))
 variants={k:np.asarray(v) for k,v in variants.items()};summary={}
 for fam in d['meta']['families']:
  rr=[v for v in details if v['family']==fam]
  summary[fam]=dict(rows=len(rr),released_sources=dict(Counter(v['raw_dataset'] for v in rr)),
   original_topics=dict(Counter(v['original_topic'] or 'MISSING' for v in rr)),
   native_subject_rows=sum(v['native_subject'] is not None for v in rr),native_cell_rows=sum(v['native_cell'] is not None for v in rr),
   native_cells=dict(Counter(v['native_cell'] or 'FAMILY_FALLBACK' for v in rr)),
   generated_sources=dict(Counter(v['v2_source'] for v in rr)),generated_broad_topics=dict(Counter(v['generated_broad'] or 'NONE' for v in rr)))
 return variants,details,dict(by_family=summary,unverified_source_joins=mismatches,label_source_hashes={str(p):sha(p) for p in [ap,rp,vp,srcpath]},
   note='Native metadata cells include subject, task-type and source identifiers; they are not uniformly semantic topics. Existing v2 is a diagnostic only, not certified manual ground truth.')

def lookup(cells,fam,r,c,tr,te):
 rt=np.repeat(r[tr].mean(0)[None],len(te),0);ct=np.repeat(c[tr].mean(0)[None],len(te),0);fallback=0
 for g in np.unique(fam[te]):
  src=tr[fam[tr]==g];ix=fam[te]==g
  if len(src):rt[ix]=r[src].mean(0);ct[ix]=c[src].mean(0)
 for cell in np.unique(cells[te]):
  src=tr[cells[tr]==cell];ix=cells[te]==cell
  if len(src):rt[ix]=r[src].mean(0);ct[ix]=c[src].mean(0)
  else:fallback+=int(ix.sum())
 return rt,ct,fallback

def main():
 d=load('sprout-sixfam');variants,details,audit=build(d);out=ROOT/'results/sprout_cell_audit';out.mkdir(exist_ok=False)
 (out/'label_inventory.json').write_text(json.dumps(audit,indent=2));np.savez_compressed(out/'candidate_cells.npz',query_ids=d['query_ids'],**variants)
 (out/'label_provenance.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in details))
 splits=json.loads((ROOT/'inputs/sprout-sixfam/splits.json').read_text());lams=json.loads((ROOT/'config/protocol.json').read_text())['lambdas']['sprout-sixfam'];res=[]
 for split,s in splits.items():
  if not split.startswith('iid'):continue
  tr=np.array(s['train']);te=np.array(s['test']);dest=out/split;dest.mkdir();arrays={'test_rows':te}
  for name,cells in variants.items():
   rr,cc,fb=lookup(cells,d['fam'],d['r'],d['c'],tr,te)
   for lam in lams:
    ch=(rr-lam*cc).argmax(1);rec=score(ch,d['r'][te],d['c'][te],d['fam'][te],lam)
    res.append(dict(split=split,variant=name,lambda_value=lam,n_test=len(te),family_fallback_rows=fb,cell_count=len(set(cells)),**rec));arrays[f'{name}|{lam:g}']=ch.astype(np.int16)
  np.savez_compressed(dest/'predictions.npz',**arrays)
 (out/'results.json').write_text(json.dumps(res,indent=2));print(json.dumps(audit['by_family'],indent=2),flush=True)
 print('DONE',len(res),'lookup scores',flush=True)
if __name__=='__main__':main()
