"""Radiologist-labelled evaluation sets: which ChestX-Det images count as unseen by the classifier."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from radio.data.chestx_det import eval_records


def _split(root, split, names):
    (root / split).mkdir(parents=True)
    anns = []
    for n in names:
        Image.fromarray(np.zeros((32, 32), np.uint8)).save(root / split / n)
        anns.append({"file_name": n, "syms": ["Nodule"], "boxes": [[1, 1, 9, 9]], "polygons": []})
    (root / f"ChestX_Det_{split}.json").write_text(json.dumps(anns))


@pytest.fixture
def cxd(tmp_path):
    _split(tmp_path, "train", ["tr_nihtest.png", "tr_nihtrain.png", "tr_unmapped.png"])
    _split(tmp_path, "test", ["te_nihtest.png", "te_nihtrain.png", "te_unmapped.png", "te_leaky.png"])
    return tmp_path


MATCHES = {"tr_nihtest.png": "a_000.png", "tr_nihtrain.png": "b_000.png", "te_nihtest.png": "c_000.png",
           "te_nihtrain.png": "d_000.png", "te_leaky.png": "e_000.png"}
NIH_TEST = {"a_000.png", "c_000.png", "e_000.png"}


def names(recs):
    return {r.file_name for r in recs}


def test_chestx_test_drops_seen_copies_and_leaky_images(cxd):
    recs, n_ex = eval_records(cxd, "chestx_test", MATCHES, NIH_TEST, leaky={"te_leaky.png"})
    assert names(recs) == {"te_nihtest.png", "te_unmapped.png"} and n_ex == 2


def test_nih_test_patients_uses_both_splits_and_only_mapped_test_patients(cxd):
    recs, n_ex = eval_records(cxd, "nih_test_patients", MATCHES, NIH_TEST, leaky={"te_leaky.png"})
    # leaky-for-the-detector does not matter here: the classifier never sees NIH test patients
    assert names(recs) == {"tr_nihtest.png", "te_nihtest.png", "te_leaky.png"} and n_ex == 4
    sub, _ = eval_records(cxd, "nih_test_patients", MATCHES, NIH_TEST, include={"te_nihtest.png"})
    assert names(sub) == {"te_nihtest.png"}
    with pytest.raises(ValueError):
        eval_records(cxd, "everything", MATCHES, NIH_TEST)
