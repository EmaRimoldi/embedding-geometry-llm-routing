import urllib.request,json,hashlib,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
base='https://api.github.com/repos/somerstep/CARROT'
def get(url):
 return subprocess.check_output(['curl','--fail','--silent','--show-error','--max-time','60','-L',url])
c=json.loads(get(base+'/commits/main'));rev=c['sha'];tree=json.loads(get(base+'/git/trees/'+rev+'?recursive=1'))
(D/'carrot_tree.json').write_text(json.dumps(tree,indent=2));manifest={'commit':rev,'files':{}}
for a in tree['tree']:
 p=a['path']
 if a['type']=='blob' and (p in ['README.md','carrot/gen_sprout.py','carrot/data_utils.py','demo/router.py'] or (p.startswith('notebooks/') and ('sprout' in p or 'accuracy' in p)) or ('cost' in p and p.endswith('.py'))):
  data=get('https://raw.githubusercontent.com/somerstep/CARROT/'+rev+'/'+p);dst=D/'carrot'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data);manifest['files'][p]={'sha256':hashlib.sha256(data).hexdigest(),'url':'https://github.com/somerstep/CARROT/blob/'+rev+'/'+p}
(D/'carrot_manifest.json').write_text(json.dumps(manifest,indent=2));print(rev,list(manifest['files']))
