"""ChestX-Det loader (Deepwise; 3,578 NIH ChestX-ray14 images; 13 classes; radiologist boxes + polygons).

Annotation format (per image): {"file_name", "syms": [class...], "boxes": [[x1,y1,x2,y2]...], "polygons": ...}.
Images are renamed (e.g. "36199.png"), so NIH patient IDs are not available. Consequences:
* the official train/test split is used as-is (we cannot re-check patient disjointness inside ChestX-Det);
* overlap with our NIH CXR14 classifier data is detected by image fingerprinting (``fingerprint_matches``)
  so leaked images can be excluded from classifier-side evaluations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from radio.ontology.cxr import CHESTX_DET_LABEL_MAP, label_space_for

DET_CONCEPTS: list[str] = label_space_for(CHESTX_DET_LABEL_MAP)  # 12 internal concepts (Nodule+Diffuse Nodule)


@dataclass
class DetRecord:
    file_name: str
    path: Path
    boxes: np.ndarray  # [N,4] x1,y1,x2,y2 source pixels
    labels: list[str]  # internal concept keys, len N
    image_labels: set[str]  # concepts present in the image (from syms)


def load_split(root: str | Path, split: str) -> list[DetRecord]:
    root = Path(root)
    ann = json.loads((root / f"ChestX_Det_{split}.json").read_text())
    out: list[DetRecord] = []
    for a in ann:
        p = root / split / a["file_name"]
        if not p.exists():
            continue
        labs = [CHESTX_DET_LABEL_MAP[s] for s in a["syms"]]
        boxes = np.asarray(a["boxes"], dtype=np.float32).reshape(-1, 4)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        out.append(DetRecord(a["file_name"], p, boxes[keep], [lab for lab, k in zip(labs, keep) if k], set(labs)))
    return out


def split_train_val(records: list[DetRecord], val_frac: float = 0.15, salt: str = "cxdet") -> tuple[list, list]:
    def bucket(name: str) -> float:
        return int(hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    val = [r for r in records if bucket(r.file_name) < val_frac]
    train = [r for r in records if bucket(r.file_name) >= val_frac]
    return train, val


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


class DetectionDataset(Dataset):
    """Yields (image [3,H,W] in [0,1], target dict) for torchvision detection models.

    Resizing is left to the model's GeneralizedRCNNTransform so predictions come back in source pixels.
    Augmentation: intensity jitter only (no flips: laterality; no geometric warps to keep boxes exact).
    """

    def __init__(self, records: list[DetRecord], concepts: list[str], augment: bool = False):
        self.records = records
        self.cls = {c: i + 1 for i, c in enumerate(concepts)}  # 0 = background
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = read_gray(r.path)
        if self.augment:
            rng = np.random.default_rng()
            img = np.clip((img - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
        t = torch.from_numpy(img)[None].expand(3, -1, -1).contiguous()
        target = {
            "boxes": torch.from_numpy(r.boxes).float().reshape(-1, 4),
            "labels": torch.tensor([self.cls[lab] for lab in r.labels], dtype=torch.int64),
            "image_id": torch.tensor(i),
        }
        return t, target


def collate(batch):
    return tuple(zip(*batch))


def _thumb(path: Path, size: int = 32) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def fingerprint_matches(det_paths: list[Path], nih_paths: list[Path], threshold: float = 0.985,
                        size: int = 32) -> dict[str, str]:
    """Map ChestX-Det file name -> NIH image name for near-identical images (normalised 32x32 correlation)."""
    if not det_paths or not nih_paths:
        return {}
    D = np.stack([_thumb(p, size) for p in det_paths])
    N = np.stack([_thumb(p, size) for p in nih_paths])
    out: dict[str, str] = {}
    for start in range(0, len(D), 512):
        sims = D[start:start + 512] @ N.T
        j = sims.argmax(1)
        best = sims[np.arange(len(j)), j]
        for k in np.flatnonzero(best >= threshold):
            out[det_paths[start + k].name] = nih_paths[j[k]].name
    return out


def patient_leak_report(root: str | Path, matches: dict[str, str], nih_csv: str | Path) -> dict:
    """Use NIH fingerprint matches to find ChestX-Det test images whose NIH patient also appears in ChestX-Det
    train (the official split is not patient-disjoint). Only matched images can be checked."""
    import pandas as pd

    pid = pd.read_csv(nih_csv).set_index("Image Index")["Patient ID"].astype(str)
    tr = [r.file_name for r in load_split(root, "train")]
    te = [r.file_name for r in load_split(root, "test")]
    p_tr = {pid[matches[f]] for f in tr if f in matches and matches[f] in pid.index}
    leaky = sorted(f for f in te if f in matches and matches[f] in pid.index and pid[matches[f]] in p_tr)
    return {
        "test_images": len(te),
        "test_images_mapped_to_nih": sum(f in matches for f in te),
        "test_images_patient_in_train": len(leaky),
        "leaky_test_images": leaky,
        "note": "Unmapped test images could not be checked; true leakage is likely higher.",
    }
