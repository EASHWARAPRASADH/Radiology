"""Image encoder factory (stage 2 of the training strategy).

Phase 1: timm DINOv2 ViT-S/14 (Apache-2.0 code and weights). Continued CXR self-supervised pretraining
on commercially licensed images (Phase 2+, cloud) will produce checkpoints loaded via ``checkpoint``.
Weight provenance is checked against ``radio.data.registry.WEIGHTS`` at training time.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn


def create_encoder(name: str, *, pretrained: bool, img_size: int, checkpoint: str | Path | None = None) -> nn.Module:
    kwargs: dict = {"num_classes": 0, "global_pool": ""}
    if "vit" in name or "dinov2" in name:
        kwargs["img_size"] = img_size
    enc = timm.create_model(name, pretrained=pretrained and checkpoint is None, **kwargs)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = enc.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys in encoder checkpoint: {unexpected[:5]}")
    return enc
