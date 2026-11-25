"""NIH ChestX-ray14 (NLP-mined labels) + ChestX-Det train (radiologist labels) for the classifier.

Both sets are NIH images (ChestX-Det re-annotated a subset), so the merge must avoid duplicates and leaks:

* A ChestX-Det image that is a pixel copy of a local NIH image (``nih_matches.json``) inherits that NIH
  patient. It goes to the same split as the patient (train/val), or is dropped if the patient is in the NIH
  test split. The NIH copy is removed so the image is not seen twice; the ChestX-Det row keeps the radiologist
  labels and takes the NIH labels for concepts ChestX-Det does not annotate.
* Unmapped ChestX-Det images (their NIH patient is unknown locally) are split by file name hash. They can
  still share a patient with other images; that residual leak cannot be checked without full NIH metadata.
* Label masks: ChestX-Det annotates 10 of the 14 NIH concepts. The other four (lung opacity, pneumonia, edema,
  hernia) are masked out on ChestX-Det rows unless the NIH copy supplies them.
* Calibration policy ``radiologist_where_available``: on the validation split, concepts that ChestX-Det
  annotates are calibrated on radiologist labels only (NIH rows masked for those concepts), because deployment
  error rates are judged against radiologists. NIH-only concepts keep NIH labels.

ChestX-Det contains no normal images, so its negatives for a concept are images with other abnormalities.
Conformal thresholds calibrated on them are therefore set against hard negatives.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from radio.data.chestx_det import load_split
from radio.data.nih import NIHSplits

RADIOLOGIST_CONCEPTS = ("atelectasis", "cardiomegaly", "pleural_effusion", "mass", "nodule", "pneumothorax",
                        "consolidation", "emphysema", "fibrosis", "pleural_thickening")


def _bucket(name: str, salt: str) -> float:
    return int(hashlib.sha256(f"{salt}:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def mask_cols(labels: list[str]) -> list[str]:
    return [f"m_{lab}" for lab in labels]


def load_mixed(nih: NIHSplits, chestx_root: str | Path, matches_path: str | Path | None, *,
               val_frac: float = 0.15, calibration_policy: str = "radiologist_where_available") -> NIHSplits:
    labels = list(nih.labels)
    shared = [c for c in labels if c in RADIOLOGIST_CONCEPTS]
    matches = json.loads(Path(matches_path).read_text()) if matches_path and Path(matches_path).exists() else {}
    frames = {"train": nih.train.copy(), "val": nih.val.copy(), "test": nih.test.copy()}
    for f in frames.values():
        f["source"] = "nih"
        for c in labels:
            f[f"m_{c}"] = 1.0
    where = {}  # nih image -> (split, row)
    for split, f in frames.items():
        for _, row in f.iterrows():
            where[row["image"]] = (split, row)

    rows: dict[str, list[dict]] = {"train": [], "val": []}
    dropped_test_patient = 0
    used_nih: set[str] = set()
    for rec in load_split(chestx_root, "train"):
        nih_img = matches.get(rec.file_name)
        row = {"image": f"chestx_det/{rec.file_name}", "path": str(rec.path), "source": "chestx_det",
               "view": "PA"}
        nih_row = None
        if nih_img and nih_img in where:
            split, nih_row = where[nih_img]
            if split == "test":
                dropped_test_patient += 1
                continue
            row["patient_id"] = str(nih_row["patient_id"])
            used_nih.add(nih_img)
        else:
            split = "val" if _bucket(rec.file_name, "cxd-val") < val_frac else "train"
            row["patient_id"] = f"cxd:{rec.file_name}"
        for c in labels:
            if c in RADIOLOGIST_CONCEPTS:
                row[c], row[f"m_{c}"] = float(c in rec.image_labels), 1.0
            elif nih_row is not None:
                row[c], row[f"m_{c}"] = float(nih_row[c]), 1.0
            else:
                row[c], row[f"m_{c}"] = 0.0, 0.0
        rows[split].append(row)

    out = {}
    for split in ("train", "val"):
        base = frames[split][~frames[split]["image"].isin(used_nih)]
        out[split] = pd.concat([base, pd.DataFrame(rows[split])], ignore_index=True)
    if calibration_policy == "radiologist_where_available":
        nih_val = out["val"]["source"] == "nih"
        for c in shared:
            out["val"].loc[nih_val, f"m_{c}"] = 0.0
    elif calibration_policy != "all":
        raise ValueError(f"unknown calibration_policy {calibration_policy!r}")
    test = frames["test"]
    assert not (set(out["train"].patient_id) & set(test.patient_id)), "patient leakage train/test"
    assert not (set(out["train"].patient_id) & set(out["val"].patient_id)), "patient leakage train/val"
    info = {"chestx_det_train_rows": len(rows["train"]), "chestx_det_val_rows": len(rows["val"]),
            "nih_duplicates_replaced": len(used_nih), "dropped_nih_test_patient": dropped_test_patient}
    splits = NIHSplits(train=out["train"], val=out["val"], test=test, labels=labels)
    splits.info = info  # type: ignore[attr-defined]
    return splits


def positives_per_concept(frame: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    return {c: int(np.sum(frame[c].to_numpy() * frame[f"m_{c}"].to_numpy())) for c in labels}
