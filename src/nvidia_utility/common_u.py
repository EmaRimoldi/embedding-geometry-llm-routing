"""Frozen inputs for the approved per-target NVIDIA utility adaptation."""
import functools, hashlib, json, os, sys, types
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/baselines"))
sys.path.insert(0, str(ROOT / "src/nvidia"))
import common as base
import common_nv as previous
from run_mlp import atomic_json, atomic_npz as _atomic_npz
from timing import hardware
OUT = ROOT / "results/nvidia_layer_validation/utility_v1"
CONFIG = ROOT / "configs/nvidia_utility_v1.json"
REFERENCE = ROOT / "src/nvidia_utility/vendor"
def atomic_npz(p, **arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    _atomic_npz(p, **arrays)
@functools.lru_cache(None)
def cfg(): return json.loads(CONFIG.read_text())
@functools.lru_cache(None)
def sha(path): return base.sha(path)
@functools.lru_cache(None)
def data(panel): return base.load(panel)
@functools.lru_cache(None)
def splits(panel): return json.loads(base.split_path(panel).read_text())
def task(i): return cfg()["tasks"][i]
def rows(panel, split):
    s = splits(panel)[split]
    return np.asarray(s["train"]), np.asarray(s["test"])
def dest(panel, split): return OUT / base.DATASETS[panel] / split
def pca_dir(panel, split, layer):
    tr, _ = rows(panel, split)
    return previous.transform_dir(panel, tr, layer)
@functools.lru_cache(maxsize=32)
def features(panel, split, layer):
    p = pca_dir(panel, split, layer)
    rec = json.loads((p / "ready.json").read_text())
    tr, _ = rows(panel, split)
    assert rec["identity"]["train_row_hash"] == previous.sig(tr)
    assert rec["identity"]["config_sha256"] == sha(previous.CONFIG)
    assert rec["identity"]["layer"] == layer and rec["identity"]["panel"] == panel
    assert sha(p / "features.npy") == rec["feature_output_sha256"]
    with np.load(p / "transform.npz") as z: np.testing.assert_array_equal(z["train_rows"], tr)
    x = np.load(p / "features.npy", mmap_mode="r")
    assert x.shape == (len(data(panel)["r"]), 100) and np.isfinite(x).all()
    return x
def verify():
    for p, h in json.loads((OUT / "provenance/source_hashes.json").read_text()).items():
        assert base.sha(ROOT / p) == h, ("source changed", p)
    for p, h in cfg()["input_hashes"].items():
        assert sha(ROOT / p) == h, ("input changed", p)
    sources = json.loads((REFERENCE / "sources.json").read_text())
    for name, rec in sources["files"].items():
        assert sha(REFERENCE / name) == rec["sha256"]
def trunk():
    text = (REFERENCE / "trunk.py").read_text()
    original = "loss_fn = nn.BCEWithLogitsLoss()"
    assert text.count(original) == 1
    adapted = text.replace(original, "loss_fn = nn.MSELoss()")
    mod = types.ModuleType("nvidia_utility_v1_trunk")
    exec(compile(adapted, str(REFERENCE / "trunk.py"), "exec"), mod.__dict__)
    return mod
def identity(panel, split):
    return dict(panel=panel, split=split, config_sha256=sha(CONFIG),
                split_sha256=sha(base.split_path(panel)))
def selection(panel, split):
    p = dest(panel, split) / "selection.json"
    rec = json.loads(p.read_text())
    assert rec["identity"] == identity(panel, split)
    return rec
def assemble_features(panel, split, layers):
    tr, te = rows(panel, split)
    assert len(layers) == data(panel)["r"].shape[1]
    # Preserve one block per target, including independently selected duplicates.
    x = np.hstack([features(panel, split, int(l))[tr] for l in layers])
    xe = np.hstack([features(panel, split, int(l))[te] for l in layers])
    assert x.shape[1] == len(layers) * 100
    return np.asarray(x, np.float32), np.asarray(xe, np.float32)
