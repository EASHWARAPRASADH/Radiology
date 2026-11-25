"""Per-label temperature scaling (Guo et al., ICML 2017; arXiv:1706.04599), masked."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperatures(logits: np.ndarray, y: np.ndarray, mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    lo = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    mt = torch.tensor(mask, dtype=torch.float64)
    log_t = torch.zeros(lo.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lo / log_t.exp(), yt, reduction="none")
        loss = (loss * mt).sum() / mt.sum().clamp_min(1)
        loss.backward()
        return loss

    opt.step(closure)
    return np.clip(log_t.detach().exp().numpy(), 0.05, 20.0)


def apply_temperature(logits: np.ndarray, temps: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(logits / temps)))
