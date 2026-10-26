import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import common,geometry

def test_unicode_jsonl_is_read_without_splitting_prompt(tmp_path,monkeypatch):
 p=tmp_path/'datasets/toy/prepared';p.mkdir(parents=True)
 (p/'queries.jsonl').write_text(json.dumps({'prompt':'alpha\u2028beta'},ensure_ascii=False)+'\n')
 (p/'manifest.json').write_text('{}');np.savez(p/'panel.npz',minilm_keep=np.array([True]))
 monkeypatch.setattr(common,'ROOT',tmp_path)
 d,rows,ix,m=common.load_data('toy');assert rows[0]['prompt']=='alpha\u2028beta'

def test_full_iid_and_ood_smoke(tmp_path,monkeypatch):
 rng=np.random.default_rng(42);n=120;m=3;x=rng.normal(size=(n,8)).astype(np.float32)
 f=np.array(['a']*60+['b']*60);topics=np.array(['t0','t1']*60);r=rng.integers(0,2,(n,m)).astype(float)
 data=dict(reward=r,cost=rng.uniform(0,.01,(n,m)),family=f,topic=topics,subgroup=np.array([a+'::'+b for a,b in zip(f,topics)]),format=np.where(f=='a','choice','free'),keys=np.array([str(i) for i in range(n)]))
 p=tmp_path/'artifacts/embeddings/toy/toy';p.mkdir(parents=True);np.save(p/'vectors.npy',x);(p/'manifest.json').write_text(json.dumps({'vector_sha256':'toy'}))
 q=tmp_path/'configs';q.mkdir();(q/'geometry.json').write_text('{}')
 monkeypatch.setattr(geometry,'ROOT',tmp_path)
 monkeypatch.setattr(geometry,'CONFIG',dict(ks=[5,20],ridge_alphas=[1],bootstrap_repetitions=20,lambdas={'toy':[0,10]}))
 monkeypatch.setattr(geometry,'load_data',lambda *args:(data,[{}]*n,np.arange(n),{'panel_sha256':'toy'}))
 for fold in ['iid_0','ood_0']:
  geometry.main('toy','toy','primary',fold)
  out=tmp_path/'results/geometry/toy/toy/primary'/fold
  a=json.loads((out/'metrics.json').read_text());assert a['n_test']>0
  split=np.load(out/'split.npz');assert not np.intersect1d(split['train_rows'],split['test_rows']).size
  vals=np.load(out/'per_query.npz');nn=vals['unrestricted/neighbors'];assert np.isin(nn,split['train_rows']).all()
  if fold.startswith('ood'):assert (f[nn]!='a').all()

def test_absent_family_fold_is_not_an_experiment_failure(tmp_path,monkeypatch):
 import json
 x=np.ones((4,2),np.float32);d={'family':np.array(['a','a','b','b'])};p=tmp_path/'artifacts/embeddings/toy/toy';p.mkdir(parents=True);np.save(p/'vectors.npy',x);(p/'manifest.json').write_text('{}')
 monkeypatch.setattr(geometry,'ROOT',tmp_path)
 monkeypatch.setattr(geometry,'load_data',lambda *args:(d,[{}]*4,np.arange(4),{'panel_sha256':'toy','family_counts':{'a':2,'b':2,'c':2}}))
 geometry.main('toy','toy','shared_minilm','ood_2')
 out=tmp_path/'results/geometry/toy/toy/shared_minilm/ood_2'
 a=json.loads((out/'not_applicable.json').read_text());assert a['absent_families']==['c'];assert not (out/'complete.json').exists()
