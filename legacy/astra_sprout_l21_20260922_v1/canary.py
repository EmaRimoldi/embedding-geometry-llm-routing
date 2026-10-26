"""Recompute 12 stratified rows to validate cached L21 row order and pooling."""
import json,os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent

def main():
    meta=json.load(open(ROOT/'inputs/manifest.json'));z=np.load(ROOT/'inputs/panel.npz')
    rows=[json.loads(l) for l in open(ROOT/'inputs/queries.jsonl')]
    modelpath=meta['reader_extraction_meta']['path']
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,device_map='cuda',local_files_only=True).eval()
    checks=[]
    for g in range(6):
        ix=np.flatnonzero(z['fam']==g);order=ix[np.argsort(z['reader_tokens'][ix])]
        for j in (int(order[len(order)//2]),int(order[int(.9*(len(order)-1))])):
            t=tok(rows[j]['prompt'],add_special_tokens=False,truncation=False,return_tensors='pt').to('cuda')
            assert t['input_ids'].shape[1]==z['reader_tokens'][j] and t['input_ids'].shape[1]<=8192
            with torch.inference_mode():h=model(**t,output_hidden_states=True,use_cache=False).hidden_states[21][0].float()
            for rep,v in [('reader',h[-1]),('reader_mean',h.mean(0))]:
                a=v.cpu().numpy();b=np.asarray(np.load(meta['representations'][rep]['path'],mmap_mode='r')[z['panel_rows'][j]],np.float32)
                cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
                checks.append(dict(row=rows[j]['row'],family=rows[j]['family'],representation=rep,cosine=cosine))
    passed=all(a['cosine']>=.995 for a in checks)
    out={'pass':passed,'snapshot_sha256':meta['snapshot_sha256'],'checks':checks}
    (ROOT/'canary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    assert passed,'Cached extraction validation failed; routing must not start'
if __name__=='__main__':main()
