"""Model bundle = weights + config + calibration + OOD statistics + lineage, versioned by content hash.

Built from a training run directory by ``build_bundle`` (stage 7 calibration happens here, on the
validation split, which is disjoint by patient from both training and test).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from radio.calibrate.conformal import LabelThreshold, fit_thresholds, thresholds_from_json, thresholds_to_json
from radio.calibrate.temperature import apply_temperature, fit_temperatures
from radio.models.classifier import ClassifierConfig, CXRClassifier, build_classifier
from radio.qc.ood import MahalanobisOOD


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_bundle(run_dir: str | Path, *, alpha_fn: float = 0.05, alpha_fp: float = 0.05, min_positives: int = 30,
                 min_negatives: int = 100, ood_quantile: float = 0.995) -> Path:
    run = Path(run_dir)
    val = np.load(run / "preds_val.npz", allow_pickle=True)
    labels = [str(x) for x in val["labels"]]
    temps = fit_temperatures(val["logits"], val["y"], val["mask"])
    probs = apply_temperature(val["logits"], temps)
    ths = fit_thresholds(probs, val["y"], val["mask"], labels, temps, alpha_fn=alpha_fn, alpha_fp=alpha_fp,
                         min_positives=min_positives, min_negatives=min_negatives)
    tr = np.load(run / "feats_train.npz")
    ood = MahalanobisOOD.fit(tr["feats"], val["feats"], quantile=ood_quantile)

    out = run / "bundle"
    out.mkdir(exist_ok=True)
    shutil.copy2(run / "model.pt", out / "model.pt")
    shutil.copy2(run / "model_config.json", out / "model_config.json")
    if (run / "lineage.json").exists():
        shutil.copy2(run / "lineage.json", out / "lineage.json")
    (out / "calibration.json").write_text(json.dumps(
        {"method": "per-label temperature + conformal FN/FP thresholds", "calibration_split": "val",
         "n_calibration_images": int(len(val["y"])), "thresholds": thresholds_to_json(ths)}, indent=2))
    ood.save(out / "ood.npz")
    sha = _sha256(out / "model.pt")
    cfg = json.loads((out / "model_config.json").read_text())
    lineage = json.loads((out / "lineage.json").read_text()) if (out / "lineage.json").exists() else {}
    (out / "bundle.json").write_text(json.dumps({
        "model_id": "radio-cxr-classifier",
        "model_version": f"{cfg['encoder']}@{sha[:12]}",
        "weights_sha256": sha,
        "research_only": lineage.get("research_only", True),
        "intended_use": "Research prototype. Not a medical device. Not for clinical use.",
    }, indent=2))
    return out


@dataclass
class ModelBundle:
    model: CXRClassifier
    config: ClassifierConfig
    thresholds: list[LabelThreshold]
    ood: MahalanobisOOD
    model_id: str
    model_version: str
    research_only: bool
    path: Path

    @property
    def labels(self) -> list[str]:
        return list(self.config.labels)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> ModelBundle:
        p = Path(path)
        cfgd = json.loads((p / "model_config.json").read_text())
        cfg = ClassifierConfig(encoder=cfgd["encoder"], pretrained=False, img_size=cfgd["img_size"],
                               labels=tuple(cfgd["labels"]), dropout=cfgd.get("dropout", 0.1))
        model = build_classifier(cfg)
        model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
        model.to(device).eval()
        cal = json.loads((p / "calibration.json").read_text())
        meta = json.loads((p / "bundle.json").read_text())
        return cls(model=model, config=cfg, thresholds=thresholds_from_json(cal["thresholds"]),
                   ood=MahalanobisOOD.load(p / "ood.npz"), model_id=meta["model_id"],
                   model_version=meta["model_version"], research_only=bool(meta.get("research_only", True)), path=p)
