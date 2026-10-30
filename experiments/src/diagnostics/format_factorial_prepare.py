"""Freeze balanced MMLU questions and their four controlled variants."""
import json,hashlib,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CFG=json.loads((ROOT/'configs/format_factorial_v1.json').read_text())
OUT=ROOT/'datasets/format_factorial_v1/prepared'
PREFIX='Please answer with the letter of the correct answer.\n\n'
SUFFIX='\nPrint only a single choice  from "A" or "B" or "C" or "D" without explanation. Answer:'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (OUT/'manifest.json').exists(),'Frozen input already exists'
 source=ROOT/'datasets/routerbench/prepared/queries.jsonl'
 groups=collections.defaultdict(list)
 for line in source.open():
  row=json.loads(line)
  if row['family']=='mmlu':groups[row['topic']].append(row)
 assert len(groups)==57
 selected=[];variants=[]
 for topic,rows in sorted(groups.items()):
  rows=sorted(rows,key=lambda r:hashlib.sha256(f"{CFG['seed']}|{r['key']}".encode()).hexdigest())[:CFG['questions_per_subject']]
  assert len(rows)==50
  for row in rows:
   p=row['prompt'];assert p.startswith(PREFIX) and p.endswith(SUFFIX),row['key']
   body=p[len(PREFIX):-len(SUFFIX)]
   matches=list(re.finditer(r'(?m)^([A-D])\) ',body))
   assert [m.group(1) for m in matches]==list('ABCD'),row['key']
   stem=body[:matches[0].start()];opts=body[matches[0].start():]
   assert stem+opts==body
   base=len(selected);selected.append(dict(base=base,key=row['key'],topic=topic,body=body,stem=stem,options=opts))
   for t in range(2):
    for a in range(2):
     instruction=['Return only the letter of the correct option, without explanation.',
                  'Return only the exact text of the correct option, without its letter or explanation.'][a]
     text=(body+'\n\n'+instruction) if t==0 else ('The following is a multiple-choice question.\n\nQuestion:\n'+stem+'\nOptions:\n'+opts+'\n\n'+instruction+'\nAnswer:')
     variants.append(dict(key=row['key']+f'::t{t}a{a}',base=base,variant=2*t+a,source_key=row['key'],topic=topic,template=t,answer_format=a,prompt=text))
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 (OUT/'queries.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in variants))
 assert len(selected)==2850 and len(variants)==11400
 (OUT/'manifest.json').write_text(json.dumps(dict(config=CFG,base_questions=len(selected),variants=len(variants),source_sha256=sha(source),queries_sha256=sha(OUT/'queries.jsonl'),config_sha256=sha(ROOT/'configs/format_factorial_v1.json'),code_sha256=sha(__file__),subject_counts={s:50 for s in groups}),indent=2)+'\n')
 print('PREPARED',len(selected),len(variants),flush=True)
if __name__=='__main__':main()
