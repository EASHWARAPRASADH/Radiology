"""Out-of-distribution gate (stage 4b): Mahalanobis distance on encoder embeddings.

Lee et al., NeurIPS 2018 (arXiv:1807.03888); Fort et al., NeurIPS 2021 (arXiv:2106.03004) show this
works well on pretrained transformer features. We fit a single Gaussian with a Ledoit-Wolf shrunk
covariance on in-distribution training embeddings and set the threshold at a high quantile of
held-out in-distribution scores (default 99.5th percentile -> ~0.5% false-abstention rate by design).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class MahalanobisOOD:
    mean: np.ndarray
    precision: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, train_feats: np.ndarray, val_feats: np.ndarray, quantile: float = 0.995) -> MahalanobisOOD:
        lw = LedoitWolf().fit(train_feats.astype(np.float64))
        obj = cls(mean=lw.location_, precision=lw.precision_, threshold=float("inf"))
        obj.threshold = float(np.quantile(obj.score(val_feats), quantile))
        return obj

    def score(self, feats: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(feats).astype(np.float64) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))

    def is_ood(self, feats: np.ndarray) -> np.ndarray:
        return self.score(feats) > self.threshold

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, precision=self.precision, threshold=np.array(self.threshold))

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisOOD:
        z = np.load(path)
        return cls(mean=z["mean"], precision=z["precision"], threshold=float(z["threshold"]))
