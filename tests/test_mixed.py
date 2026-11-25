"""NIH + ChestX-Det merge for the classifier: dedupe, patient-safe splits, label masks, calibration policy."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from radio.data.mixed import load_mixed
from radio.data.nih import NIHDataset, load_nih


def _chestx(root, n=10):
    (root / "train").mkdir(parents=True)
    anns = []
    for i in range(n):
        name = f"train{i}.png"
        Image.fromarray(np.full((64, 64), 40 + i, np.uint8)).save(root / "train" / name)
        anns.append({"file_name": name, "syms": ["Effusion"] if i % 2 == 0 else ["Nodule"],
                     "boxes": [[5, 5, 30, 30]], "polygons": []})
    (root / "ChestX_Det_train.json").write_text(json.dumps(anns))
    return root


def test_merge_is_patient_safe_and_masks_labels(fake_nih, tmp_path):
    nih = load_nih(fake_nih / "Data_Entry_2017.csv", fake_nih / "images", split_dir=None, val_frac=0.3,
                   test_frac=0.2)
    tr_img, va_img, te_img = nih.train.image.iloc[0], nih.val.image.iloc[0], nih.test.image.iloc[0]
    matches = {"train0.png": tr_img, "train1.png": te_img, "train2.png": va_img}
    (tmp_path / "m.json").write_text(json.dumps(matches))
    root = _chestx(tmp_path / "cxd")
    m = load_mixed(nih, root, tmp_path / "m.json", val_frac=0.3)

    assert m.info["dropped_nih_test_patient"] == 1 and m.info["nih_duplicates_replaced"] == 2
    assert "chestx_det/train1.png" not in set(m.train.image) | set(m.val.image)  # NIH test patient: dropped
    assert tr_img not in set(m.train.image) and va_img not in set(m.val.image)  # duplicates replaced
    t0 = m.train[m.train.image == "chestx_det/train0.png"].iloc[0]
    assert t0.patient_id == nih.train.patient_id.iloc[0]  # inherits the NIH patient and split
    assert "chestx_det/train2.png" in set(m.val.image)
    assert t0.pleural_effusion == 1.0 and t0.m_pleural_effusion == 1.0  # radiologist label
    assert t0.m_lung_opacity == 1.0  # NIH copy supplies the concept ChestX-Det does not annotate
    unmapped = m.train[m.train.image == "chestx_det/train4.png"]
    unmapped = unmapped if len(unmapped) else m.val[m.val.image == "chestx_det/train4.png"]
    assert unmapped.iloc[0].m_lung_opacity == 0.0 and unmapped.iloc[0].m_nodule == 1.0

    nih_val = m.val[m.val.source == "nih"]
    assert (nih_val.m_pleural_effusion == 0).all() and (nih_val.m_lung_opacity == 1).all()  # calibration policy
    assert not set(m.train.patient_id) & set(m.val.patient_id)
    assert not set(m.train.patient_id) & set(m.test.patient_id)

    ds = NIHDataset(m.val, m.labels, 32)
    assert ds.mask.shape == ds.y.shape and float(ds.mask.sum()) < ds.mask.numel()

    every = load_mixed(nih, root, tmp_path / "m.json", val_frac=0.3, calibration_policy="all")
    assert (every.val[every.val.source == "nih"].m_pleural_effusion == 1).all()
