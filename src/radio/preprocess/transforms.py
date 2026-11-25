"""Deterministic preprocessing (letterbox resize) with an invertible geometric record.

No horizontal flips anywhere in the pipeline: laterality is clinically meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# DINOv2 / ImageNet normalisation (grayscale replicated to 3 channels).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    size: int
    src_h: int
    src_w: int

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map a coordinate in the model input frame back to source-image pixels."""
        sx = (x - self.pad_x) / self.scale
        sy = (y - self.pad_y) / self.scale
        return float(np.clip(sx, 0, self.src_w)), float(np.clip(sy, 0, self.src_h))


def letterbox(img: np.ndarray, size: int) -> tuple[torch.Tensor, Letterbox]:
    """Resize the long side to ``size`` keeping aspect ratio, pad to a square. Returns 1xHxW tensor."""
    h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None]
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)[0]
    pad_y, pad_x = (size - nh) // 2, (size - nw) // 2
    out = torch.zeros((1, size, size), dtype=torch.float32)
    out[:, pad_y : pad_y + nh, pad_x : pad_x + nw] = t
    return out, Letterbox(scale, pad_x, pad_y, size, h, w)


def normalize(t: torch.Tensor) -> torch.Tensor:
    """1xHxW in [0,1] -> 3xHxW normalised."""
    t3 = t.expand(3, -1, -1) if t.shape[0] == 1 else t
    mean = torch.tensor(MEAN, dtype=t3.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=t3.dtype).view(3, 1, 1)
    return (t3 - mean) / std


def prepare(img: np.ndarray, size: int) -> tuple[torch.Tensor, Letterbox]:
    t, lb = letterbox(img, size)
    return normalize(t), lb
