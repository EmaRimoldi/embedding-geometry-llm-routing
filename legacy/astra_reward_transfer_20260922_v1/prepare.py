"""Freeze a matched no-truncation dataset, independently of mutable agent code."""
from pathlib import Path
import hashlib, json, os, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
E4 = Path(os.environ['E4_ROOT'])
sys.path.insert(0, str(ROOT / 'vendor'))
from rb.data import load_panel
from rb.reps import find_rep, load_rep
from transformers import AutoTokenizer

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def lengths(tok, texts):
    out = []
    for s in range(0, len(texts), 256):
        ids = tok(list(texts[s:s+256]), add_special_tokens=False, truncation=False)['input_ids']
        out.extend(map(len, ids))
    return np.asarray(out, np.int64)

def main():
    panel_name = sys.argv[1]
    dest = ROOT / 'inputs' / panel_name
    dest.mkdir(parents=True, exist_ok=False)
    p = load_panel(panel_name)
    spec = find_rep(panel_name, 'computed/gte-qwen2-7b@prompt')
    hp = E4/'features'/panel_name/'prompt/hidden/qwen2.5-7b-instruct/L21_last.npy'
    hm = json.loads((hp.parent/'meta.json').read_text())
    gm = json.loads(spec.path.with_suffix('.npy.json').read_text())
    toks = [AutoTokenizer.from_pretrained(v, local_files_only=True, trust_remote_code=True)
            for v in (gm['model'], hm['path'])]
    gl, hl = [lengths(t, p.prompts) for t in toks]
    keep = (gl + 1 <= 8192) & (hl <= 8192)
    # Outcomes are only frozen here; no statistical selection uses them.
    assert p.available.all(), 'Need an explicit missing-availability policy'
    assert np.isfinite(p.rewards).all() and np.isfinite(p.costs).all()
    assert np.all(p.costs >= 0)
    ids = np.flatnonzero(keep)
    xg = load_rep(spec, p.n, p.keep)
    xh = np.load(hp, mmap_mode='r')
    if p.keep is not None and len(xh) == len(p.keep): xh = xh[p.keep]
    assert len(xh) == len(xg) == p.n
    source_paths = [spec.path, hp]
    source_stats = {str(q): [q.stat().st_size, q.stat().st_mtime_ns] for q in source_paths}
    for name, x in [('gte', xg), ('reader', xh)]:
        z = np.asarray(x[ids], np.float32)
        assert np.isfinite(z).all() and np.all(np.linalg.norm(z, axis=1) > 0)
        np.save(dest/f'{name}.npy', z)
    missing = np.zeros_like(p.rewards, bool) if p.release_missing is None else p.release_missing
    np.savez_compressed(dest/'panel.npz', r=p.rewards[ids], c=p.costs[ids],
        fam=p.dataset_index()[ids], panel_rows=ids, query_ids=p.query_ids[ids].astype(str),
        gte_tokens=gl[ids], reader_tokens=hl[ids], missing=missing[ids],
        original_gte_tokens=gl, original_reader_tokens=hl)
    assert source_stats == {str(q): [q.stat().st_size, q.stat().st_mtime_ns] for q in source_paths}
    meta = dict(panel=panel_name, rows=len(ids), original_rows=p.n,
        excluded=int((~keep).sum()), families=list(map(str,p.datasets)), models=list(map(str,p.models)),
        original_panel=p.provenance(), gte_meta=gm, reader_meta=hm,
        excluded_by_family={str(f):int(((p.dataset_ids==f)&~keep).sum()) for f in p.datasets},
        source_sha256={str(q):sha(q) for q in source_paths},
        snapshot_sha256={q.name:sha(q) for q in dest.iterdir() if q.is_file()},
        missing_cells=int(missing[ids].sum()))
    (dest/'manifest.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k:meta[k] for k in ('panel','rows','excluded','excluded_by_family','missing_cells')}, indent=2), flush=True)

if __name__ == '__main__': main()
