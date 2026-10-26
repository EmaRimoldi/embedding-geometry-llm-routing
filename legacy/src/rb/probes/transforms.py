"""Input transforms, fit on the training fold only (standardise, unit, whiten, residualise, concatenate)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Transform:
    def __init__(self, kind: str, groups: np.ndarray, n_groups: int, controls: np.ndarray | None = None,
                 log_norm: np.ndarray | None = None):
        self.kind = kind
        self.groups = groups
        self.n_groups = n_groups
        self.controls = controls
        self.log_norm = log_norm
        self.metric = "cosine"

    def _onehot(self, idx: np.ndarray) -> np.ndarray:
        out = np.zeros((len(idx), self.n_groups), dtype=np.float32)
        out[np.arange(len(idx)), self.groups[idx]] = 1.0
        return out

    def fit_transform(self, x: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self.kind
        if k == "onehot":
            self.metric = "euclidean"
            return self._onehot(tr), self._onehot(te)
        if k == "controls":
            self.metric = "euclidean"
            sc = StandardScaler().fit(self.controls[tr])
            return sc.transform(self.controls[tr]), sc.transform(self.controls[te])
        if k == "norm":
            self.metric = "euclidean"
            feats = np.column_stack([self.log_norm, self.controls])
            sc = StandardScaler().fit(feats[tr])
            return sc.transform(feats[tr]), sc.transform(feats[te])
        if k == "raw":
            sc = StandardScaler().fit(x[tr])
            return sc.transform(x[tr]), sc.transform(x[te])
        if k == "unit":
            u = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            return u[tr], u[te]
        if k == "whiten":
            self.metric = "euclidean"
            n_comp = int(min(x.shape[1], 512, len(tr) - 1))
            pca = PCA(n_components=n_comp, whiten=True, random_state=0).fit(x[tr])
            return pca.transform(x[tr]), pca.transform(x[te])
        if k == "resid":
            # Remove the dataset-mean subspace estimated on the train fold. Under
            # LODO the held-out dataset has no train mean: its rows are centred
            # with the grand mean, which is the honest deployment condition.
            xc = x.astype(np.float32)
            grand = xc[tr].mean(0)
            means = {g: xc[tr][self.groups[tr] == g].mean(0) for g in np.unique(self.groups[tr])}
            def centre(idx):
                out = np.empty((len(idx), x.shape[1]), dtype=np.float32)
                for i, row in enumerate(idx):
                    out[i] = xc[row] - means.get(self.groups[row], grand)
                return out
            a, b = centre(tr), centre(te)
            sc = StandardScaler().fit(a)
            self.metric = "cosine"
            return sc.transform(a), sc.transform(b)
        if k == "surf+raw":
            # Embedding concatenated with the surface features (scaled so the
            # 10 surface columns are not shrunk away relative to the k embedding columns).
            self.metric = "euclidean"
            sc = StandardScaler().fit(x[tr])
            ss = StandardScaler().fit(self.controls[tr])
            w = np.sqrt(x.shape[1] / self.controls.shape[1])
            return (np.hstack([ss.transform(self.controls[tr]) * w, sc.transform(x[tr])]),
                    np.hstack([ss.transform(self.controls[te]) * w, sc.transform(x[te])]))
        if k == "desurf":
            # Residualise every embedding dimension on the surface features
            # (least squares fit on the train fold), leaving what the embedding
            # knows beyond length/format.
            self.metric = "cosine"
            ss = StandardScaler().fit(self.controls[tr])
            s_tr = np.hstack([ss.transform(self.controls[tr]), np.ones((len(tr), 1))])
            s_te = np.hstack([ss.transform(self.controls[te]), np.ones((len(te), 1))])
            beta, *_ = np.linalg.lstsq(s_tr, x[tr].astype(np.float64), rcond=None)
            a = x[tr] - s_tr @ beta
            b = x[te] - s_te @ beta
            sc = StandardScaler().fit(a)
            return sc.transform(a), sc.transform(b)
        if k == "onehot+raw":
            sc = StandardScaler().fit(x[tr])
            self.metric = "euclidean"
            return (np.hstack([self._onehot(tr) * np.sqrt(x.shape[1]), sc.transform(x[tr])]),
                    np.hstack([self._onehot(te) * np.sqrt(x.shape[1]), sc.transform(x[te])]))
        raise ValueError(k)
