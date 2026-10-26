"""Audit preprocessing without loading models or changing recorded embeddings."""
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from geometry import E4
from presentation_controls import DEST,VARIANTS

def main():
    rows=[json.loads(s) for s in (DEST/'inputs/controlled.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['kind']=='presentation']
    record={}
    for rep,model in [('gte','gte-Qwen2-7B-instruct'),('reader','Qwen2.5-7B-Instruct__a09a3545')]:
        path=E4.parent/'artifacts/models'/model
        tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=True)
        if rep=='gte':
            config=json.loads((path/'sentence_bert_config.json').read_text())
            assert not config.get('do_lower_case',False),'Include lowercase preprocessing if enabled'
        texts=[r['text'].strip() if rep=='gte' else r['text'] for r in rows]
        encoded=tok(texts,add_special_tokens=(rep=='gte'),truncation=False)['input_ids']
        originals={(r['panel'],r['row']):ids for r,ids in zip(rows,encoded) if r['variant']=='original'}
        rec={}
        for v in VARIANTS:
            pairs=[(r,ids) for r,ids in zip(rows,encoded) if r['variant']==v]
            same=sum(ids==originals[r['panel'],r['row']] for r,ids in pairs)
            rec[v]=dict(rows=len(pairs),identical_to_original_tokens=same,fraction_identical=same/len(pairs),max_tokens=max(len(ids) for r,ids in pairs))
        record[rep]=dict(preprocessing='str.strip()' if rep=='gte' else 'raw text; no chat template',variants=rec)
    record['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (DEST/'token_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
