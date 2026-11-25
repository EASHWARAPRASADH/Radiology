"""Masked multi-label losses: labels that are not assessed (mask=0) contribute nothing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
               pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_asl(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, gamma_neg: float = 4.0,
               gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """Asymmetric loss (Ridnik et al., ICCV 2021) with masking."""
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    lp = y * torch.log(p.clamp(min=eps)) * (1 - p) ** gamma_pos
    ln = (1 - y) * torch.log((1 - p_neg).clamp(min=eps)) * p_neg ** gamma_neg
    loss = -(lp + ln)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
