"""Finding detector (Phase 2: localisation): torchvision Faster R-CNN (BSD-3 code).

A detector *bundle* stores weights, config, per-class operating points and per-class validation status.
Only classes that pass the Phase-2 gate on held-out data (08-evaluation-protocol.md section 8: FROC
sensitivity >= 0.80 at 1 FP/image with enough reference lesions) are ``validated``; only those may set
``Localization.validated=True`` / ``Provenance.image_observed`` downstream.

Deliberately not used: Ultralytics YOLO (AGPL-3.0, which Ultralytics applies to trained models too).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


@dataclass
class Detection:
    concept: str  # internal ontology key
    score: float  # raw detector score
    box_xyxy: tuple[float, float, float, float]  # source-image pixels


class FindingDetector(Protocol):
    concepts: tuple[str, ...]
    version: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...


@dataclass
class DetectorConfig:
    concepts: tuple[str, ...]
    arch: str = "fasterrcnn_resnet50_fpn_v2"
    pretrained_backbone: bool = True
    min_size: int = 512
    max_size: int = 512
    trainable_backbone_layers: int = 3
    score_thresh: float = 0.01  # keep low-score boxes for FROC; operating points applied later
    detections_per_img: int = 50

    def to_dict(self) -> dict:
        d = asdict(self)
        d["concepts"] = list(self.concepts)
        return d


def build_detector(cfg: DetectorConfig) -> torch.nn.Module:
    """``pretrained_backbone`` pulls torchvision ImageNet ResNet-50 weights (registry verdict CONDITIONAL)."""
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    if cfg.arch != "fasterrcnn_resnet50_fpn_v2":
        raise ValueError(f"unsupported detector arch {cfg.arch}")
    return fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained_backbone else None,
        num_classes=len(cfg.concepts) + 1,
        trainable_backbone_layers=cfg.trainable_backbone_layers,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        box_score_thresh=cfg.score_thresh,
        box_detections_per_img=cfg.detections_per_img,
    )


# Backwards-compatible helper used in Phase 1 docs/tests.
def build_faster_rcnn(concepts: tuple[str, ...], *, pretrained_backbone: bool = False) -> torch.nn.Module:
    return build_detector(DetectorConfig(concepts=tuple(concepts), pretrained_backbone=pretrained_backbone))


@dataclass
class ClassOperatingPoint:
    concept: str
    box_score_threshold: float  # box-level threshold chosen at target FP/image on validation
    target_fppi: float
    validated: bool  # passed the held-out gate
    gate: dict = field(default_factory=dict)  # metrics used for the decision


@torch.no_grad()
def run_detector(model: torch.nn.Module, images: list[np.ndarray], concepts: list[str],
                 device: torch.device) -> list[list[Detection]]:
    model.eval()
    batch = [torch.from_numpy(np.ascontiguousarray(im, dtype=np.float32))[None].expand(3, -1, -1).to(device)
             for im in images]
    outs = model(batch)
    res = []
    for o in outs:
        b, s, lab = o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()
        res.append([Detection(concepts[int(lab[i]) - 1], float(s[i]), tuple(float(v) for v in b[i]))
                    for i in range(len(s))])
    return res


class DetectorBundle:
    """Loaded detector + operating points; implements FindingDetector."""

    def __init__(self, path: str | Path, device: torch.device | str = "cpu"):
        p = Path(path)
        self.path = p
        self.device = torch.device(device)
        self.config = DetectorConfig(**{**json.loads((p / "det_config.json").read_text()),
                                        "pretrained_backbone": False})
        self.config.concepts = tuple(self.config.concepts)
        self.model = build_detector(self.config)
        self.model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()
        ops = json.loads((p / "operating_points.json").read_text())
        self.operating_points = {o["concept"]: ClassOperatingPoint(**o) for o in ops}
        meta = json.loads((p / "bundle.json").read_text())
        self.model_id = meta["model_id"]
        self.version = meta["model_version"]
        self.research_only = bool(meta.get("research_only", True))
        self.concepts = self.config.concepts

    def detect(self, image: np.ndarray) -> list[Detection]:
        return run_detector(self.model, [image], list(self.concepts), self.device)[0]

    def validated_concepts(self) -> set[str]:
        return {c for c, o in self.operating_points.items() if o.validated}


def write_detector_bundle(out: Path, model_state: dict, cfg: DetectorConfig, ops: list[ClassOperatingPoint],
                          lineage: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, out / "model.pt")
    (out / "det_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out / "operating_points.json").write_text(json.dumps([asdict(o) for o in ops], indent=2))
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
    sha = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    (out / "bundle.json").write_text(json.dumps({
        "model_id": "radio-cxr-detector",
        "model_version": f"{cfg.arch}@{sha[:12]}",
        "weights_sha256": sha,
        "research_only": lineage.get("research_only", True),
        "intended_use": "Research prototype. Not a medical device. Not for clinical use.",
    }, indent=2))
    return out
