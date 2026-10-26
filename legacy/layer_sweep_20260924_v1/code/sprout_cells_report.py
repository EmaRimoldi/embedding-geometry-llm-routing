"""Read-only report (PI request, 2026-09-25): what do the 53 approved SPROUT cells actually split?

For every family: the kind of subdivision and its label provenance, every cell with size, outcome profile and
example prompts, and two kinds of pairs under the emitted GTE embedding used in the paper:
  split together-looking  = a query and its nearest same-family neighbour that fall in DIFFERENT cells
  kept apart-looking      = two queries in the SAME cell that are among its least similar pairs
Also reports how often a query's nearest same-family neighbour sits in another cell.
Second part: why the GTE layer re-extraction for the quick GTE-L21 check differs from the stored GTE embedding.
Writes results/summary/sprout_cells_report.md.
"""
import json
from collections import Counter,defaultdict
import numpy as np
from common_ls import DEF,E4,ROOT,load,unit

CELLS=DEF/'inputs/sprout_cells_native_openhermes_v1'
RNG=np.random.default_rng(20260925);SAMPLE=1500;CUT=280

def short(t,n=CUT):
 t=' '.join(str(t).split());return t if len(t)<=n else t[:n]+' …'

def main():
 d=load('sprout-sixfam','gte');x,r,c,fam=d['x'],d['r'],d['c'],d['fam'];fams=d['meta']['families'];models=d['meta']['models']
 rows=[json.loads(l) for l in open(E4/'sprout/panel/rows.jsonl')];prompt=[rows[i]['prompt'] for i in d['panel_rows']]
 z=np.load(CELLS/'cells.npz');cell_of=dict(zip(z['query_ids'],z['cells']));cell=np.array([cell_of[q] for q in d['query_ids']])
 lab={j['key']:j for j in map(json.loads,open(CELLS/'labels.jsonl'))};meta=[lab[q] for q in d['query_ids']]
 md=['# What the 53 approved SPROUT cells split\n',
     f'{len(cell):,} queries, {len(set(cell))} cells. Cell = `family::subdivision`. Similarity = cosine of the emitted GTE embedding (the paper\'s). '
     'Prompts are whitespace-collapsed and cut at 280 characters. Outcome profile: mean reward over the 13 models and the best model at λ=0 (mean reward).\n']
 for g,fname in enumerate(fams):
  idx=np.flatnonzero(fam==g);cs=Counter(cell[idx]);prov=Counter((m['v2_source'],m['quality_status']) for m in (meta[i] for i in idx))
  md.append(f'\n## {fname} ({len(idx):,} queries, {len(cs)} cells)\n')
  md.append('Label provenance: '+'; '.join(f'{k[0]} / {k[1]}: {v:,}' for k,v in prov.most_common()))
  md.append('\n| cell | queries | mean reward | best model (λ=0) |\n|---|---|---|---|')
  for cname,n in cs.most_common():
   ii=idx[cell[idx]==cname];mr=r[ii].mean(0);md.append(f'| `{cname}` | {n:,} | {r[ii].mean():.3f} | {models[int(mr.argmax())]} ({mr.max():.3f}) |')
  if len(cs)==1:
   md.append('\n*Single cell: the whole family is one lookup cell (no subdivision is available), so nothing is split.*')
  md.append('\n### Examples from each cell\n')
  for cname,_ in cs.most_common():
   ii=idx[cell[idx]==cname];pick=RNG.choice(ii,size=min(2,len(ii)),replace=False)
   extra=''
   if fname=='openhermes':extra=' · generated sub-topic: '+', '.join(sorted({str(meta[i].get("v2_topic")) for i in pick}))
   md.append(f'- **`{cname}`**{extra}')
   for i in pick:md.append(f'  - {short(prompt[i])}')
  if len(cs)==1:continue
  # nearest same-family neighbour of a sample, in the full family
  samp=RNG.choice(idx,size=min(SAMPLE,len(idx)),replace=False);nn=np.empty(len(samp),int);ns=np.empty(len(samp))
  xi=x[idx]
  for s in range(0,len(samp),256):
   sim=x[samp[s:s+256]]@xi.T;sim[np.arange(len(sim)),np.searchsorted(idx,samp[s:s+256])]=-np.inf
   j=sim.argmax(1);nn[s:s+256]=idx[j];ns[s:s+256]=sim[np.arange(len(sim)),j]
  cross=cell[samp]!=cell[nn]
  md.append(f'\n### How geometric is the split?\n\nFor {len(samp):,} sampled queries, the nearest same-family neighbour lies in a **different cell {cross.mean():.1%}** of the time '
            f'(if cells were random with these sizes: {1-sum((v/len(idx))**2 for v in cs.values()):.1%}).')
  md.append('\n### Split: near-identical-looking queries that land in different cells\n')
  order=np.argsort(-ns*cross);shown=0
  for k in order:
   if not cross[k] or shown>=4:break
   a,b=samp[k],nn[k];md.append(f'{shown+1}. cosine {ns[k]:.3f}\n   - `{cell[a]}`: {short(prompt[a])}\n   - `{cell[b]}`: {short(prompt[b])}');shown+=1
  md.append('\n### Kept together: dissimilar queries that share a cell\n')
  shown=0
  for cname,n in cs.most_common(4):
   ii=idx[cell[idx]==cname]
   if len(ii)<2:continue
   s=RNG.choice(ii,size=min(400,len(ii)),replace=False);sim=x[s]@x[s].T;np.fill_diagonal(sim,np.inf);a,b=np.unravel_index(sim.argmin(),sim.shape)
   md.append(f'{shown+1}. `{cname}`, cosine {sim[a,b]:.3f} (least similar pair among 400 sampled)\n   - {short(prompt[s[a]])}\n   - {short(prompt[s[b]])}');shown+=1
 # part 2: the GTE layer re-extraction check
 f=ROOT/'features/sprout-sixfam/gte-qwen2-7b/L28_last.npy'
 if f.exists():
  mine=unit(np.asarray(np.load(f,mmap_mode='r')[d['panel_rows']],np.float32));cos=(mine*x).sum(1)
  ws=np.array([p!=p.strip() for p in prompt]);low=cos<.99
  md.append('\n## Appendix: GTE layer re-extraction vs the stored SPROUT GTE embedding\n')
  md.append(f'Cosine between my last-layer EOS state and the stored embedding: mean {cos.mean():.4f}, min {cos.min():.4f}; {low.mean():.1%} of queries below 0.99.')
  md.append(f'Queries with leading/trailing whitespace: {ws.mean():.1%} overall, {ws[low].mean():.1%} among the <0.99 queries, {ws[~low].mean():.1%} among the rest; '
            f'mean cosine {cos[ws].mean():.4f} with whitespace vs {cos[~ws].mean():.4f} without.')
  per=defaultdict(list)
  for g,v in zip(fam,cos):per[fams[g]].append(v)
  md.append('Per family mean cosine: '+', '.join(f'{k} {np.mean(v):.4f}' for k,v in per.items()))
 out=ROOT/'results/summary/sprout_cells_report.md';out.write_text('\n'.join(md)+'\n');print(out)

if __name__=='__main__':main()
