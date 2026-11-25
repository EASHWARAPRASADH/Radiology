"""NIH ChestX-ray14 loader (Wang et al., CVPR 2017). Commercially usable with attribution per the NIH
README/FAQ (see docs/blueprint/04-dataset-licensing-matrix.md).

* Labels are NLP-mined from reports (noisy); "No Finding" != normal.
* All 14 labels are treated as assessed for every image (mask=1). Concepts outside the NIH label space
  are not part of this model's output and will be reported as not_assessed downstream.
* Splits are patient-disjoint. If the official ``train_val_list.txt`` / ``test_list.txt`` are present
  they define the test set; validation is carved from train_val by patient.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from radio.ontology.cxr import NIH_CXR14_LABEL_MAP, label_space_for
from radio.preprocess.transforms import letterbox, normalize

NIH_LABELS = label_space_for(NIH_CXR14_LABEL_MAP)


def _patient_bucket(pid: str, salt: str = "radio") -> float:
    h = hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def index_images(images_root: Path) -> dict[str, Path]:
    return {p.name: p for p in images_root.rglob("*.png")}


@dataclass
class NIHSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]


def load_nih(
    csv_path: str | Path,
    images_root: str | Path,
    *,
    split_dir: str | Path | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    max_images: int | None = None,
    frontal_only: bool = True,
    seed: int = 0,
) -> NIHSplits:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Image Index": "image", "Finding Labels": "finding_labels",
                            "Patient ID": "patient_id", "View Position": "view"})
    files = index_images(Path(images_root))
    df = df[df["image"].isin(files)].copy()
    if df.empty:
        raise FileNotFoundError(f"No CSV-listed images found under {images_root}")
    df["path"] = df["image"].map(lambda n: str(files[n]))
    df["patient_id"] = df["patient_id"].astype(str)
    if frontal_only:
        df = df[df["view"].isin(["PA", "AP"])]

    raw = df["finding_labels"].str.split("|")
    for src, key in NIH_CXR14_LABEL_MAP.items():
        df[key] = raw.map(lambda labs, s=src: float(s in labs))

    if max_images is not None and len(df) > max_images:
        rng = np.random.default_rng(seed)
        pats = df["patient_id"].unique()
        rng.shuffle(pats)
        counts = df.groupby("patient_id").size()
        keep, total = [], 0
        for p in pats:
            if total >= max_images:
                break
            keep.append(p)
            total += int(counts[p])
        df = df[df["patient_id"].isin(keep)]

    sd = Path(split_dir) if split_dir else None
    if sd and (sd / "test_list.txt").exists():
        test_names = set((sd / "test_list.txt").read_text().split())
        is_test = df["image"].isin(test_names)
        test_pats = set(df.loc[is_test, "patient_id"])
        # Guard against leakage if a patient straddles lists.
        is_test = df["patient_id"].isin(test_pats)
    else:
        is_test = df["patient_id"].map(lambda p: _patient_bucket(p, "test") < test_frac)
    rest = df[~is_test]
    is_val = rest["patient_id"].map(lambda p: _patient_bucket(p, "val") < val_frac)
    return NIHSplits(train=rest[~is_val].reset_index(drop=True), val=rest[is_val].reset_index(drop=True),
                     test=df[is_test].reset_index(drop=True), labels=list(NIH_LABELS))


def train_augment() -> v2.Compose:
    # Geometric jitter + intensity jitter. Deliberately no horizontal flip (laterality).
    return v2.Compose([
        v2.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.92, 1.08)),
        v2.ColorJitter(brightness=0.15, contrast=0.15),
    ])


class NIHDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: list[str], img_size: int, augment: bool = False):
        self.df = frame.reset_index(drop=True)
        self.labels = labels
        self.size = img_size
        self.aug = train_augment() if augment else None
        self.y = torch.tensor(self.df[labels].to_numpy(dtype=np.float32))
        mcols = [f"m_{lab}" for lab in labels]
        if all(c in self.df.columns for c in mcols):  # mixed sources: per-row label masks (data/mixed.py)
            self.mask = torch.tensor(self.df[mcols].to_numpy(dtype=np.float32))
        else:
            self.mask = torch.ones_like(self.y)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        with Image.open(self.df.at[i, "path"]) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        t, _ = letterbox(arr, self.size)
        if self.aug is not None:
            t = self.aug(t).clamp(0, 1)
        return normalize(t), self.y[i], self.mask[i], i
