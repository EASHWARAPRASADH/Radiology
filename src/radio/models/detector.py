"""Finding detector interface (Phase 2: localisation).

Phase 1 ships only the interface and a torchvision Faster R-CNN baseline builder (BSD-3). Training on
ChestX-Det boxes (Apache-2.0 annotations, NIH images) is Phase 2. Only once a detector is validated
against expert boxes (FROC / mAP, see 08-evaluation-protocol.md) may its outputs set
``Localization.validated=True`` and ``Provenance.image_observed``.

Deliberately not used: Ultralytics YOLO (AGPL-3.0, which Ultralytics applies to trained models too).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass
class Detection:
    concept: str  # internal ontology key
    score: float  # raw detector score (calibrate before use)
    box_xyxy: tuple[float, float, float, float]  # source-image pixels


class FindingDetector(Protocol):
    concepts: tuple[str, ...]
    version: str
    validated: bool

    def detect(self, image: np.ndarray) -> list[Detection]: ...


def build_faster_rcnn(concepts: tuple[str, ...], *, pretrained_backbone: bool = False) -> torch.nn.Module:
    """torchvision Faster R-CNN (ResNet-50 FPN v2). ``num_classes`` includes background.

    ``pretrained_backbone`` pulls ImageNet weights (licence verdict CONDITIONAL in the registry);
    the default trains from scratch or from our own encoder checkpoint.
    """
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    return fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
        num_classes=len(concepts) + 1,
    )
