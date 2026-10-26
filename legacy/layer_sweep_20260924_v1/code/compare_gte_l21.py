"""Quick check (PI, 2026-09-24): does kNN route better on GTE's own layer 21 than on GTE's emitted embedding?
IID splits iid_0 and iid_1, RouterBench and SPROUT; Qwen2.5-7B-Instruct L21 shown for reference.
Also checks the SPROUT GTE extraction: its last layer (EOS token) must match the stored GTE embedding.
"""
import json
import numpy as np
from common_ls import E4,PROTOCOL,ROOT,load,unit

REPS=[('gte','emitted',None,'GTE emitted'),('gte-qwen2-7b','last',21,'GTE L21'),('qwen2.5-7b-instruct','last',21,'Qwen L21')]
SPLITS=['iid_0','iid_1']

def main():
 md=['# kNN on GTE layer 21 vs GTE emitted (IID splits iid_0, iid_1)\n']
 f=ROOT/'features/sprout-sixfam/gte-qwen2-7b/L28_last.npy'
 if f.exists():
  d=load('sprout-sixfam');mine=unit(np.asarray(np.load(f,mmap_mode='r')[d['panel_rows']],np.float32))
  ref=unit(np.load(E4/'sprout/features/gte-qwen2-7b.npy',mmap_mode='r')[d['gte_rows']]);cos=(mine*ref).sum(1)
  md.append(f'SPROUT extraction check, last layer vs stored GTE embedding: cosine mean {cos.mean():.6f}, min {cos.min():.6f}\n')
 for panel in ('routerbench-0shot','sprout-sixfam'):
  lams=PROTOCOL['lambdas'][panel];md.append(f'\n## {panel}\n\n| representation | router | '+' | '.join(f'λ={l:g}' for l in lams)+' | mean |\n|'+'---|'*(len(lams)+3))
  for reader,pool,layer,label in REPS:
   tag='emitted' if reader=='gte' else f'L{layer}_{pool}';res=[]
   for sp in SPLITS:
    p=ROOT/'results/knn'/panel/reader/tag/f'{sp}.json'
    if p.exists():res.append(json.loads(p.read_text()))
   if len(res)<len(SPLITS):md.append(f'| {label} | incomplete | '+' | '*len(lams)+' |');continue
   for key,name in (('100','kNN-100'),('validated','validated kNN')):
    u=np.mean([np.array(r['test_sum'][key])/r['n_test'] for r in res],0)
    md.append(f'| {label} | {name} | '+' | '.join(f'{x:.4f}' for x in u)+f' | **{u.mean():.4f}** |')
 out=ROOT/'results/summary/gte_l21_iid.md';out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(md)+'\n');print('\n'.join(md))

if __name__=='__main__':main()
