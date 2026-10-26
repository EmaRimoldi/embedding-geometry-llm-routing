"""Freeze the user-approved cell policy separately from all historical labels."""
import json
from collections import Counter
import numpy as np
from common import ROOT,load,score,sha
src=ROOT/'results/sprout_cell_audit';out=ROOT/'inputs/sprout_cells_native_openhermes_v1'
out.mkdir(exist_ok=False)
d=load('sprout-sixfam');z=np.load(src/'candidate_cells.npz');policy='native_plus_qwen_openhermes'
assert np.array_equal(z['query_ids'],d['query_ids'])
cells=z[policy];details=[json.loads(v) for v in (src/'label_provenance.jsonl').read_text().splitlines()]
assert [v['key'] for v in details]==list(d['query_ids'])
for v,c in zip(details,cells):
 v['cell']=str(c);v['policy']=policy
 v['quality_status']='generated; not independently human-certified' if '::qwen:' in c else ('unresolved; family fallback' if c.endswith('family_fallback') else 'native metadata')
(out/'labels.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in details))
np.savez_compressed(out/'cells.npz',query_ids=d['query_ids'],cells=cells)
splits=json.loads((ROOT/'inputs/sprout-sixfam/splits.json').read_text());scores=json.loads((src/'results.json').read_text());audited=[]
for v in scores:
 if v['variant']!=policy:continue
 te=np.array(splits[v['split']]['test']);pred=np.load(src/v['split']/'predictions.npz');assert np.array_equal(te,pred['test_rows'])
 ch=pred[f'{policy}|{v["lambda_value"]:g}'];s=score(ch,d['r'][te],d['c'][te],d['fam'][te],v['lambda_value'])
 assert all(abs(s[k]-v[k])<1e-12 for k in ['reward','cost','utility'])
 audited.append(v)
(out/'iid_results.json').write_text(json.dumps(audited,indent=2))
manifest=dict(policy=policy,n_queries=len(cells),n_cells=len(set(cells)),counts=dict(Counter(cells)),
 semantics='MATH/MMLU native subjects, MuSR tasks, RAG sources, broad generated OpenHermes subjects, family fallback otherwise.',
 selection_note='Policy approved after exploratory seven-policy IID comparison. These scores are not independent confirmation of the policy-selection decision.',
 old_labels_modified=False,score_audit_records=len(audited),source_hashes={p.name:sha(p) for p in [src/'candidate_cells.npz',src/'label_provenance.jsonl',src/'results.json',ROOT/'inputs/sprout-sixfam/splits.json']},
 output_hashes={p.name:sha(p) for p in out.iterdir()})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['policy','n_queries','n_cells','score_audit_records']}))
