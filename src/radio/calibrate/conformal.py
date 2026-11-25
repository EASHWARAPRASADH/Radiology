"""Per-label decision thresholds with finite-sample error control -> present / uncertain / absent.

For each label on a held-out calibration set (calibrated probabilities ``p``):

* ``t_absent``: largest threshold such that the fraction of *positives* with p < t_absent is at most
  ``alpha_fn`` (with the (n+1) conformal correction). Calling "absent" below it controls the
  false-negative rate among positives (split-conformal / conformal risk control style; Angelopoulos &
  Bates arXiv:2107.07511, Angelopoulos et al. arXiv:2208.02814).
* ``t_present``: smallest threshold such that the fraction of *negatives* with p >= t_present is at most
  ``alpha_fp``. Calling "present" above it controls the false-positive rate among negatives.
* Between the two -> ``uncertain`` (deferred to the radiologist). If the bands cross (well-separated
  label), both collapse to their midpoint and there is no uncertain band.
* Labels with fewer than ``min_positives`` positives (or negatives) in calibration are marked
  ``validated=False`` and are emitted as ``not_assessed`` - we do not output findings we cannot
  support with validation data.

Guarantees are marginal and only hold under exchangeability with the deployment population; they must
be re-established per site (see 08-evaluation-protocol.md).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from radio.schema.findings import FindingStatus


@dataclass
class LabelThreshold:
    label: str
    temperature: float
    t_absent: float
    t_present: float
    n_pos: int
    n_neg: int
    validated: bool
    alpha_fn: float
    alpha_fp: float


def _lower_quantile(scores: np.ndarray, alpha: float) -> float:
    """Threshold t with (#scores < t) <= floor(alpha*(n+1)) - 1, i.e. conformal lower quantile."""
    s = np.sort(scores)
    n = len(s)
    k = math.floor(alpha * (n + 1))  # number of positives we may lose
    if k < 1:
        return 0.0
    return float(s[min(k, n) - 1]) + 1e-12 if k <= n else 1.0


def _upper_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(scores)[::-1]
    n = len(s)
    k = math.floor(alpha * (n + 1))
    if k < 1:
        return 1.0 + 1e-9  # cannot call present with this little data at this alpha
    return float(s[min(k, n) - 1]) if k <= n else 0.0


def fit_thresholds(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    labels: list[str],
    temps: np.ndarray,
    alpha_fn: float = 0.05,
    alpha_fp: float = 0.05,
    min_positives: int = 30,
    min_negatives: int = 100,
) -> list[LabelThreshold]:
    out = []
    for j, lab in enumerate(labels):
        m = mask[:, j] > 0
        pj, yj = probs[m, j], y[m, j]
        pos, neg = pj[yj > 0.5], pj[yj <= 0.5]
        validated = len(pos) >= min_positives and len(neg) >= min_negatives
        if validated:
            t_abs = _lower_quantile(pos, alpha_fn)
            t_pre = _upper_quantile(neg, alpha_fp)
            if t_abs > t_pre:
                mid = (t_abs + t_pre) / 2
                t_abs = t_pre = mid
            t_pre = min(t_pre, 1.0 + 1e-9)
        else:
            t_abs, t_pre = 0.0, 1.0 + 1e-9
        out.append(LabelThreshold(lab, float(temps[j]), float(t_abs), float(t_pre), int(len(pos)), int(len(neg)),
                                  bool(validated), alpha_fn, alpha_fp))
    return out


def decide(p: float, th: LabelThreshold) -> FindingStatus:
    if not th.validated:
        return FindingStatus.not_assessed
    if p >= th.t_present:
        return FindingStatus.present
    if p < th.t_absent:
        return FindingStatus.absent
    return FindingStatus.uncertain


def thresholds_to_json(ths: list[LabelThreshold]) -> list[dict]:
    return [asdict(t) for t in ths]


def thresholds_from_json(d: list[dict]) -> list[LabelThreshold]:
    return [LabelThreshold(**x) for x in d]
