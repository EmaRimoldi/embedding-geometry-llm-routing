"""Model-free control representations: hashed bag-of-n-grams (CPU only).

Writes features/<panel>/<field>/hash-bow.npy — a 4,096-d L2-normalised TF vector
of word uni/bigrams (sklearn HashingVectorizer, no fitted vocabulary, so no
information from any other row).  If a contrastive encoder does no better than
this within a family, its semantic content is not what routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rb.data import ANALYSIS_ROOT, load_panel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--text-field", default="origin_query")
    args = p.parse_args()
    panel = load_panel(args.panel)
    texts = [str(t) for t in panel.text(args.text_field)]
    vec = HashingVectorizer(n_features=4096, alternate_sign=True, norm="l2", ngram_range=(1, 2), lowercase=True)
    x = vec.transform(texts).astype(np.float32).toarray()
    out = ANALYSIS_ROOT / "features" / args.panel / args.text_field / "hash-bow.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x, allow_pickle=False)
    out.with_suffix(".npy.json").write_text(json.dumps({
        "rep": "hash-bow", "model": "sklearn HashingVectorizer(4096, ngram 1-2, l2)", "normalized": True,
        "shape": list(x.shape), "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "note": f"{args.panel}/{args.text_field}; model-free lexical control"}, indent=2))
    print("wrote", out, x.shape)


if __name__ == "__main__":
    main()
