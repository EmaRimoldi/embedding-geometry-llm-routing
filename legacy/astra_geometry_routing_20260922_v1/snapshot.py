import shutil,json
from common import *
src=E4/'ood_cost_geometry_20260921';manifest={}
for p in (src/'results/exp27_representation_swap').glob('*/*/result.json'):
    target=ROOT/'snapshots/other_agent_exp27'/p.relative_to(src/'results/exp27_representation_swap')
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);manifest[str(p)]=code_sha(p)
p=src/'scripts/exp27_representation_swap.py';shutil.copy2(p,ROOT/'snapshots/exp27_representation_swap.py');manifest[str(p)]=code_sha(p)
(ROOT/'snapshots/manifest.json').write_text(json.dumps(manifest,indent=2))
print('Read-only snapshots saved',len(manifest))
