"""Multi-label CXR classifier on a timm encoder (default: DINOv2 ViT-S/14, Apache-2.0).

The head is linear on mean-pooled patch tokens, so per-class activation maps are exact decompositions
of the logit: logit_c = mean_i(w_c . t_i) + b_c. Maps are returned as *coarse, unvalidated*
localization (saliency maps are known to under-perform expert localization on CXR; see literature
review) - they are never labelled ``image_observed`` without a validated detector.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from radio.models.encoder import create_encoder


@dataclass
class ClassifierConfig:
    encoder: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    img_size: int = 448
    labels: tuple[str, ...] = ()
    dropout: float = 0.1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["labels"] = list(self.labels)
        return d


class CXRClassifier(nn.Module):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = create_encoder(cfg.encoder, pretrained=cfg.pretrained, img_size=cfg.img_size)
        self.num_features = self.encoder.num_features
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(self.num_features, len(cfg.labels))

    def tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        f = self.encoder.forward_features(x)
        if f.ndim == 4:  # CNN: B,C,H,W
            b, c, h, w = f.shape
            return f.flatten(2).transpose(1, 2), (h, w)
        t = f[:, getattr(self.encoder, "num_prefix_tokens", 1):]
        g = math.isqrt(t.shape[1])
        return t, (g, t.shape[1] // g)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t, _ = self.tokens(x)
        pooled = t.mean(1)
        return self.head(self.drop(pooled)), pooled

    @torch.no_grad()
    def forward_with_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns logits [B,L], pooled features [B,D], activation maps [B,L,h,w]."""
        t, (h, w) = self.tokens(x)
        pooled = t.mean(1)
        logits = self.head(pooled)
        maps = (t @ self.head.weight.T).transpose(1, 2).reshape(t.shape[0], -1, h, w)
        return logits, pooled, maps


def build_classifier(cfg: ClassifierConfig) -> CXRClassifier:
    if not cfg.labels:
        raise ValueError("ClassifierConfig.labels must be non-empty")
    return CXRClassifier(cfg)
