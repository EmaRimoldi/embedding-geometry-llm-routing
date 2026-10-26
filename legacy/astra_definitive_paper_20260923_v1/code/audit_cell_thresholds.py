"""Retain both explicit minimum-cell-size choices, without choosing by outcomes."""
import sys,json
from pathlib import Path
import numpy as np
from common import ROOT,E4,load,score,sha,source
sys.path.insert(0,str(E4/'astra_encoder_geometry_20260922_v1/vendor'))
from rb.data import load_panel
from rb.panelctx import topic_str

def main():
 out=ROOT/'cell_lookup_v2';out.mkdir(exist_ok=True)
 for panel in ['routerbench-0shot','sprout-sixfam']:
  d=load(panel);names=d['meta']['families'];f=d['fam'];r=d['r'];c=d['c'];dest=out/panel;dest.mkdir(exist_ok=False)
  if panel=='routerbench-0shot':
   p=load_panel(panel);frozen=np.load(source(panel)/'panel.npz');rows=frozen['panel_rows'];assert np.array_equal(p.query_ids[rows].astype(str),d['query_ids'])
   topics=np.asarray(topic_str(p),dtype=str)[rows]
   cells=np.array([names[g]+'::'+(str(t) if str(t) not in ['', 'None','nan'] else 'family_fallback') for g,t in zip(f,topics)])
   provenance='Released RouterBench labels from pinned vendor topic_str, including native source subdivisions; exact query-ID join. Cells are not uniformly semantic topics.'
  else:
   p=ROOT/'inputs/sprout_cells_native_openhermes_v1/cells.npz';z=np.load(p);assert np.array_equal(z['query_ids'],d['query_ids']);cells=z['cells'];provenance='User-approved native metadata plus broad OpenHermes generated subjects; explicit family fallback.'
  np.savez_compressed(dest/'cells.npz',query_ids=d['query_ids'],cells=cells)
  splits=json.loads((ROOT/'inputs'/panel/'splits.json').read_text());lams=json.loads((ROOT/'config/protocol.json').read_text())['lambdas'][panel];results=[]
  for split,s in splits.items():
   if not split.startswith('iid'):continue
   tr=np.array(s['train']);te=np.array(s['test']);arrays={'test_rows':te,'query_ids':d['query_ids'][te]}
   for minimum in [1,50]:
    rr=np.repeat(r[tr].mean(0)[None],len(te),0);cc=np.repeat(c[tr].mean(0)[None],len(te),0)
    for g in np.unique(f[te]):
     donor=tr[f[tr]==g];ix=f[te]==g;assert len(donor);rr[ix]=r[donor].mean(0);cc[ix]=c[donor].mean(0)
    fallback=0
    for cell in np.unique(cells[te]):
     donor=tr[cells[tr]==cell];ix=cells[te]==cell
     if len(donor)>=minimum:rr[ix]=r[donor].mean(0);cc[ix]=c[donor].mean(0)
     else:fallback+=int(ix.sum())
    for lam in lams:
     ch=(rr-lam*cc).argmax(1);v=score(ch,r[te],c[te],f[te],lam);arrays[f'{minimum}|{lam:g}']=ch
     results.append(dict(panel=panel,split=split,minimum_cell_rows=minimum,lambda_value=lam,n_test=len(te),fallback_rows=fallback,**v))
   np.savez_compressed(dest/f'{split}_predictions.npz',**arrays)
  (dest/'results.json').write_text(json.dumps(results,indent=2));(dest/'manifest.json').write_text(json.dumps(dict(panel=panel,n_queries=len(cells),n_cells=len(set(cells)),provenance=provenance,code_sha256=sha(__file__),split_sha256=sha(ROOT/'inputs'/panel/'splits.json'),cell_sha256=sha(dest/'cells.npz'),selection='Both fixed thresholds retained. Awaiting user policy decision; no threshold selected on test utility.'),indent=2));print('DONE',panel,len(results),flush=True)
if __name__=='__main__':main()
