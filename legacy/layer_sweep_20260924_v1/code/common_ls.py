"""Shared helpers for the layer sweep.

Retrieval (`nearest`), scoring, frozen panels, splits and the protocol are imported unchanged from
astra_definitive_paper_20260923_v1, so the hidden_states[21]/last cell reproduces the definitive kNN
and soft-prior numbers. Reads everything outside this folder in place; writes only inside it.
"""
from pathlib import Path
import glob,json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E4=Path(os.environ.get('E4_ROOT',ROOT.parent))
DEF=E4/'astra_definitive_paper_20260923_v1'
sys.path.insert(0,str(DEF/'code'))
from common import PANELS,load,nearest,score,sha,unit  # noqa: E402,F401
PROTOCOL=json.loads((DEF/'config/protocol.json').read_text())
FORMATS=json.loads((ROOT/'config/formats.json').read_text())
READERS={k:v for k,v in json.loads((ROOT/'config/readers.json').read_text()).items() if not k.startswith('_')}
NFAM={'routerbench-0shot':7,'sprout-sixfam':6}

def model_path(reader):
 p=READERS[reader]['path'];hits=sorted(glob.glob(p))
 assert hits,f'no weights for {reader}: {p}'
 return hits[-1]

def feature_dir(panel,reader):
 cached=READERS[reader]['cached'].get(panel)
 return E4/cached if cached else ROOT/'features'/panel/reader

def splits(panel):
 return json.loads((DEF/'inputs'/panel/'splits.json').read_text())

def split_names(panel,kind):
 ood=[f'ood_{i}' for i in range(NFAM[panel])];iid=[f'iid_{i}' for i in range(5)]
 named={'ood':ood,'iid':iid,'all':iid+ood}
 return named[kind] if kind in named else kind.split('+')   # explicit list, e.g. 'iid_0+iid_1'

def family_format(panel,meta_families):
 """Integer family id -> 0 (choice) / 1 (open)."""
 m=FORMATS[panel];return np.array([0 if m[f]=='choice' else 1 for f in meta_families])

def load_rep(panel,reader,layer=None,pool=None):
 """Frozen panel plus unit-normalised representation. reader='gte' is the emitted GTE embedding
 used by the definitive study; any other reader reads hidden_states[layer] with the given pooling.
 Feature files are in extraction order; panel_rows maps panel rows onto them (identity on RouterBench)."""
 if reader=='gte':return load(panel,'gte')
 d=load(panel);path=feature_dir(panel,reader)/f'L{layer}_{pool}.npy'
 x=np.load(path,mmap_mode='r');rows=d['panel_rows']
 assert rows.max()<len(x),(path,x.shape)
 d['x']=unit(np.asarray(x[rows],np.float32));d['feature_path']=str(path)
 return d

def n_layers(panel,reader):
 meta=json.loads((feature_dir(panel,reader)/'meta.json').read_text())
 return int(meta['layers'])

def tag(reader,layer,pool):
 return 'emitted' if reader=='gte' else f'L{layer}_{pool}'
