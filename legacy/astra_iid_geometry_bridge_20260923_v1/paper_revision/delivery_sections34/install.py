from pathlib import Path
import hashlib,json,shutil,os
E4=Path('__E4_ROOT__')
stage=E4/'astra_iid_geometry_bridge_20260923_v1/paper_revision/delivery_sections34'
paper=E4/'astra_definitive_paper_20260923_v1/paper_draft_visual_v1'
m=json.loads((stage/'SECTIONS34_REVISION.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(paper/'main.tex')==m['source_main_sha256'],'Author source changed; abort and rebase.'
for name,digest in m['output_sha256'].items(): assert sha(stage/name)==digest,name
history=paper/'revision_history/before_sections34_20260923'
history.mkdir(parents=True,exist_ok=False)
for name in ['main.tex','main.pdf']:
 shutil.copy2(paper/name,history/name)
assert sha(paper/'main.tex')==m['source_main_sha256'],'Author source changed during backup; abort.'
# Install new assets before atomically replacing the authoritative source/PDF.
for name in ['sec4_iid_frontiers_cells.pdf','SECTIONS34_REVISION.json','SECTIONS34_CHANGES.md','main.tex','main.pdf']:
 temporary=paper/(name+'.sections34-new')
 shutil.copy2(stage/name,temporary)
 if name=='main.tex': assert sha(paper/'main.tex')==m['source_main_sha256']
 os.replace(str(temporary),str(paper/name))
for name,digest in m['output_sha256'].items(): assert sha(paper/name)==digest,name
print('Installed and verified:',paper)
print('Backup:',history)
