"""Freeze the exact source folds and test NVIDIA's utility adaptation before extraction."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from common_nv import *
def prepare():
 original=json.loads((ROOT/'configs/baselines.json').read_text());red=json.loads((ROOT/'configs/mlp_reduced.json').read_text())
 tasks=[];sets={}
 for p,n in [('routerbench-0shot',7),('sprout-sixfam',6)]:
  sets[p]={};d=data(p)
  for sp in [f'iid_{i}' for i in range(5)]+[f'ood_{i}' for i in range(n)]:
   tasks.append([p,sp]);s=splits(p)[sp];tr=set(s['train']);te=set(s['test']);assert not tr&te
   for rows in [s['train']]+[s['inner'][i]['fit'] for i in red['folds'][p][sp]['inner_indices']]:sets[p][sig(rows)]=rows
   for i in red['folds'][p][sp]['inner_indices']:
    f=s['inner'][i];fi=set(f['fit']);va=set(f['val']);assert fi|va==tr and not fi&va and not te&(fi|va)
    if sp.startswith('ood'):
     assert not set(d['fam'][f['fit']])&set(d['fam'][f['val']])
     assert not set(d['fam'][s['train']])&set(d['fam'][s['test']])
 record=dict(version='nvidia_utility_v1',tasks=tasks,layers=list(range(15,29)),lambdas=original['lambdas'],validation_folds=red['folds'],reduced_mlp_config_sha256=sha(ROOT/'configs/mlp_reduced.json'),screen_seed=42,ensemble_seeds=list(range(42,52)),keep_seeds=5,pca_components=100,pca_random_state=42,pca_whiten=False,prompt_format='raw',pooling='last_input_token',layer_rule='One common layer per outer split: maximize mean across penalties of query-weighted source-validation routing utility; lower layer breaks ties.',controls=['selected_layer','fixed_l21'],source_sets=sets,pca_tasks=[[p,l] for p in sets for l in range(15,29)])
 if CONFIG.exists():assert cfg()==record,'Frozen config changed'
 else:atomic_json(CONFIG,record)
 paths=['configs/nvidia.json','src/nvidia/common_nv.py','src/nvidia/timing.py','src/nvidia/compute_report.py','src/nvidia/prepare.py','src/nvidia/extract.py','src/nvidia/pca.py','src/nvidia/run.py','src/nvidia/report.py','src/nvidia/vendor/trunk.py','src/encode.py','src/common.py','jobs/nvidia_extract.sbatch','jobs/nvidia_pca.sbatch','jobs/nvidia_train.sbatch','jobs/nvidia_report.sbatch']
 hashes={p:sha(ROOT/p) for p in paths};fp=OUT/'provenance/source_hashes.json'
 if fp.exists():assert json.loads(fp.read_text())==hashes
 else:atomic_json(fp,hashes)
 verify();torch.set_num_threads(2);mod=trunk();rng=np.random.RandomState(11)
 x=rng.normal(size=(80,12)).astype(np.float32);y=np.full((80,2),-1.5,np.float32)
 net=mod.train_mlp(mod.SharedTrunkNet(12,2),x,y,seed=42,epochs=10,batch_size=32,device='cpu')
 with torch.no_grad():pred=net(torch.as_tensor(x)).numpy()
 assert np.isfinite(pred).all() and pred.mean()<0
 order=torch.randperm(len(x),generator=torch.Generator().manual_seed(42)).numpy();val=order[int(.85*len(x)):]
 mse=float(np.mean((pred[val]-y[val])**2));np.testing.assert_allclose(mse,net._best_val_loss,rtol=2e-5,atol=1e-7)
 inner_fits=sum(len(red['folds'][p][sp]['inner_indices'])*14*13 for p,sp in tasks)
 atomic_json(OUT/'provenance/preflight.json',dict(passed=True,outer_evaluations=len(tasks),screening_network_fits=inner_fits,max_final_network_fits=len(tasks)*2*13*10,validation_mse_reconstructed=mse,negative_outputs=True,loss_only_adaptation=True,source_sets={p:len(v) for p,v in sets.items()}))
 print('PREFLIGHT_PASS',len(tasks),'outer evaluations;',inner_fits,'screening fits;', {p:len(v) for p,v in sets.items()},flush=True)
if __name__=='__main__':prepare()
