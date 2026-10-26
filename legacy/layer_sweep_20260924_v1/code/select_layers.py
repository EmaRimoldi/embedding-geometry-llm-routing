"""Aggregate the sweep and select the layer on source families only.

Pooled utility = sum of test utilities over the outer family holdouts / number of test queries (query
weighted, as in the paper). A (reader, pooling, router) curve is reported only if every outer split of every
layer is complete.

Nested selection (the deployable router, 'auto'): in each outer fold, every candidate representation is
scored by its INNER leave-one-source-family-out utility, averaged over the penalty grid; the argmax is then
evaluated on the held-out family. Variants: one layer per fold (headline), one layer per (fold, penalty),
and joint (layer, pooling) selection. The held-out family never enters the choice.

Reproduction: hidden_states[21]/last and emitted GTE are compared with the definitive study's saved scores.
Writes results/summary/{summary.json,summary.md}.
"""
import json
from collections import defaultdict
import numpy as np
from common_ls import DEF,ROOT,PROTOCOL,split_names

VARIANTS={'knn':{'knn100':('100','100'),'knn_validated':('validated','max_k'),'format_restricted_100':('format_restricted_100',None)},
          'prior':{'soft_prior':('soft_prior','soft_prior'),'best_single_train':('best_single_train',None),'realized_oracle':('realized_oracle',None),
                   'format_lookup_pooled':('format_lookup_pooled',None),'format_lookup_balanced':('format_lookup_balanced',None)}}
MIN_LAYER=15   # scope (PI, 2026-09-24): upper half only; candidates are layers 15..28
DEF_NAME={'knn100':'knn100','knn_validated':'knn_validated','soft_prior':'soft_prior','best_single_train':'best_single_train','realized_oracle':'realized_oracle'}

def load_all():
 R=defaultdict(dict)   # (method,panel,reader,pool,layer) -> split -> result
 for f in (ROOT/'results').glob('*/*/*/*/*.json'):
  if f.parts[-5]=='summary':continue
  r=json.loads(f.read_text());R[(r['method'],r['panel'],r['reader'],r['pool'],r['layer'])][r['split']]=r
 return R

def inner_crit(res,src):
 """Per-penalty inner validation utility (mean per validation query)."""
 if src=='max_k':return np.max([np.array(v) for v in res['inner_sum'].values()],0)/res['inner_n']
 return np.array(res['inner_sum'][src])/res['inner_n']

def pooled(per_split,key,names):
 if not all(s in per_split and key in per_split[s]['test_sum'] for s in names):return None
 tot=np.sum([per_split[s]['test_sum'][key] for s in names],0);n=sum(per_split[s]['n_test'] for s in names);return (tot/n).tolist()

def definitive(panel,method,rep,name,names):
 tot=0;n=0
 for s in names:
  r=json.loads((DEF/'results'/method/panel/s/rep/'result.json').read_text());lams=PROTOCOL['lambdas'][panel]
  tot=tot+np.array([r['results'][str(l)][name]['utility'] for l in lams])*r['n_test'];n+=r['n_test']
 return tot/n

def main():
 R=load_all();out={'curves':{},'auto':{},'selected':{},'diagnostics':{},'reproduction':{}};md=['# Layer sweep: held-out-family routing\n']
 for panel in ('routerbench-0shot','sprout-sixfam'):
  names=split_names(panel,'ood');lams=PROTOCOL['lambdas'][panel];md.append(f'\n## {panel}\n\npenalties: {lams}\n')
  for method,variants in VARIANTS.items():
   readers=sorted({(k[2],k[3]) for k in R if k[0]==method and k[1]==panel})
   for reader,pool in readers:
    layers=sorted(k[4] for k in R if k[:4]==(method,panel,reader,pool) and k[4]>=MIN_LAYER) if reader!='gte' else [None]
    for v,(key,src) in variants.items():
     curve={L:pooled(R[(method,panel,reader,pool,L)],key,names) for L in layers}
     if any(c is None for c in curve.values()) or not curve:continue
     out['curves'][f'{panel}|{reader}|{pool}|{v}']={str(L):c for L,c in curve.items()}
  # nested selection over layers (and over layers x poolings)
  for method,variants in VARIANTS.items():
   for v,(key,src) in variants.items():
    if src is None:continue
    for reader in sorted({k[2] for k in R if k[0]==method and k[1]==panel and k[2]!='gte'}):
     for pools in (['last'],['mean'],['last','mean']):
      cands=[(p,L) for p in pools for L in sorted(k[4] for k in R if k[:4]==(method,panel,reader,p) and k[4]>=MIN_LAYER)]
      if not cands or not all(all(s in R[(method,panel,reader,p,L)] for s in names) for p,L in cands):continue
      tot=np.zeros(len(lams));tot_l=np.zeros(len(lams));n=0;chosen={}
      for s in names:
       crit={c:inner_crit(R[(method,panel,reader)+c][s],src) for c in cands}
       best=max(cands,key=lambda c:(crit[c].mean(),-c[1]));chosen[s]=dict(pool=best[0],layer=best[1],test_family=R[(method,panel,reader)+best][s]['test_families'][0])
       res=R[(method,panel,reader)+best][s];tot+=np.array(res['test_sum'][key])
       for il in range(len(lams)):   # per-penalty choice
        b=max(cands,key=lambda c:(crit[c][il],-c[1]));tot_l[il]+=R[(method,panel,reader)+b][s]['test_sum'][key][il]
       n+=res['n_test']
      name=f'{panel}|{reader}|{"+".join(pools)}|{v}'
      out['auto'][name]=dict(one_layer_per_fold=(tot/n).tolist(),one_layer_per_fold_and_penalty=(tot_l/n).tolist());out['selected'][name]=chosen
  # outcome-free diagnostics: same-format neighbour share
  for k,per in R.items():
   if k[0]=='knn' and k[1]==panel and all(s in per for s in names):
    N=sum(per[s]['n_test'] for s in names)
    out['diagnostics'][f'{panel}|{k[2]}|{k[3]}|{k[4]}']={m:sum(per[s][m]*per[s]['n_test'] for s in names)/N for m in ('same_format_share_k20','same_format_share_k100')}|{
      'per_family_k20':{per[s]['test_families'][0]:per[s]['same_format_share_k20'] for s in names}}
  # reproduction against the definitive study
  for reader,pool,layer,rep in (('qwen2.5-7b-instruct','last',21,'reader'),('gte','emitted',None,'gte')):
   for method,v in (('knn','knn100'),('knn','knn_validated'),('prior','soft_prior'),('prior','best_single_train')):
    per=R.get((method,panel,reader,pool,layer),{})
    mine=pooled(per,VARIANTS[method][v][0],names)
    if mine is None:continue
    ref=definitive(panel,method,rep,DEF_NAME[v],names);out['reproduction'][f'{panel}|{reader}|{v}']=float(np.abs(np.array(mine)-ref).max())
  # markdown
  def row(label,c):return f'| {label} | '+' | '.join(f'{x:.4f}' for x in c)+f' | **{np.mean(c):.4f}** |'
  head='| representation | '+' | '.join(f'λ={l:g}' for l in lams)+' | mean |\n|'+'---|'*(len(lams)+2)
  for v in ('knn100','knn_validated','soft_prior'):
   md.append(f'\n### {v}: pooled held-out utility by layer\n\n'+head)
   for key,c in out['curves'].items():
    p,reader,pool,vv=key.split('|')
    if p!=panel or vv!=v:continue
    for L,cc in c.items():md.append(row(f'{reader} {pool} L{L}' if reader!='gte' else 'GTE (emitted)',cc))
   for key,a in out['auto'].items():
    p,reader,pools,vv=key.split('|')
    if p==panel and vv==v:md.append(row(f'**auto** {reader} {pools} (1 layer/fold)',a['one_layer_per_fold']));md.append(row(f'auto {reader} {pools} (layer per fold×λ)',a['one_layer_per_fold_and_penalty']))
  md.append('\n### references (no representation choice)\n\n'+head)
  for key,c in out['curves'].items():
   p,reader,pool,vv=key.split('|')
   if p==panel and reader=='gte' and vv in ('best_single_train','realized_oracle','format_lookup_pooled','format_lookup_balanced','format_restricted_100'):md.append(row(f'{vv} (GTE run)' if vv=='format_restricted_100' else vv,c['None']))
  md.append('\n### selected layers (inner source validation, one layer per fold)\n')
  for key,ch in out['selected'].items():
   if key.startswith(panel):md.append(f'- {key}: '+', '.join(f'{c["test_family"]}→{c["pool"]} L{c["layer"]}' for c in ch.values()))
  md.append('\n### same-format neighbour share (k=20, held-out family removed)\n')
  for key,dg in sorted(out['diagnostics'].items()):
   if key.startswith(panel):md.append(f'- {key}: {dg["same_format_share_k20"]:.3f}  '+str({f:round(x,2) for f,x in dg['per_family_k20'].items()}))
 md.append('\n## Reproduction of the definitive study (max abs pooled-utility difference)\n')
 md+= [f'- {k}: {v:.2e}' for k,v in out['reproduction'].items()]
 dest=ROOT/'results/summary';dest.mkdir(parents=True,exist_ok=True)
 (dest/'summary.json').write_text(json.dumps(out,indent=1));(dest/'summary.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))

if __name__=='__main__':main()
