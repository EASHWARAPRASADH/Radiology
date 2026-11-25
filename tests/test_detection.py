"""Phase-2 tests: detection metrics, ChestX-Det loader, detector training plumbing, pipeline integration."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from radio.eval.detection import ImageGT, ImagePreds, average_precision, evaluate_detections, iou_matrix


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 25 / 175, 0.0], rtol=1e-6)


def test_perfect_and_empty_detections():
    gts = [ImageGT(np.array([[10, 10, 50, 50]], float), ["pneumothorax"]), ImageGT(np.zeros((0, 4)), [])]
    perfect = [ImagePreds(np.array([[10, 10, 50, 50]], float), np.array([0.9]), ["pneumothorax"]),
               ImagePreds(np.zeros((0, 4)), np.zeros(0), [])]
    r = evaluate_detections(perfect, gts, ["pneumothorax"])
    assert r["pneumothorax"]["ap50"] == pytest.approx(1.0)
    assert r["pneumothorax"]["sens@0.25fp"] == pytest.approx(1.0)
    assert r["pneumothorax"]["pointing_accuracy_tp"] == pytest.approx(1.0)
    none = [ImagePreds(np.zeros((0, 4)), np.zeros(0), []) for _ in gts]
    assert evaluate_detections(none, gts, ["pneumothorax"])["pneumothorax"]["ap50"] == 0.0


def test_froc_counts_false_positives_per_image():
    # 4 images, 1 lesion; one high-scoring FP on each of 2 images ranks above the TP.
    gts = [ImageGT(np.array([[0, 0, 10, 10]], float), ["mass"])] + [ImageGT(np.zeros((0, 4)), [])] * 3
    preds = [ImagePreds(np.array([[0, 0, 10, 10]], float), np.array([0.5]), ["mass"]),
             ImagePreds(np.array([[0, 0, 5, 5]], float), np.array([0.9]), ["mass"]),
             ImagePreds(np.array([[0, 0, 5, 5]], float), np.array([0.8]), ["mass"]),
             ImagePreds(np.zeros((0, 4)), np.zeros(0), [])]
    r = evaluate_detections(preds, gts, ["mass"])["mass"]
    assert r["sens@0.25fp"] == 0.0  # reaching the TP costs 2 FP / 4 images = 0.5 FP/image
    assert r["sens@0.5fp"] == 1.0


def _clustered_case():
    """20 images x 5 lesions. Images 0-9: every lesion found; 10-19: none found. No false positives."""
    box = np.array([[0, 0, 10, 10]], float)
    gts = [ImageGT(np.repeat(box, 5, axis=0) + np.arange(5)[:, None] * 20, ["mass"] * 5) for _ in range(20)]
    preds = [ImagePreds(g.boxes.copy(), np.full(5, 0.9), ["mass"] * 5) if i < 10
             else ImagePreds(np.zeros((0, 4)), np.zeros(0), []) for i, g in enumerate(gts)]
    return preds, gts


def test_froc_ci_point_matches_froc_and_perfect_is_exact():
    from radio.eval.detection import froc_sensitivity_ci

    preds, gts = _clustered_case()
    ci = froc_sensitivity_ci(preds, gts, "mass", fppi=1.0, n_boot=500)
    assert ci["point"] == evaluate_detections(preds, gts, ["mass"])["mass"]["sens@1.0fp"] == 0.5
    assert ci["n_gt"] == 100 and ci["n_boot"] == 500
    perfect = froc_sensitivity_ci(preds[:10], gts[:10], "mass", n_boot=200)
    assert perfect["point"] == perfect["lower"] == perfect["upper"] == 1.0
    empty = froc_sensitivity_ci(preds, gts, "nodule", n_boot=50)
    assert math.isnan(empty["lower"]) and empty["n_gt"] == 0


def test_froc_ci_resamples_images_not_boxes():
    from radio.eval.detection import froc_sensitivity_ci

    preds, gts = _clustered_case()
    ci = froc_sensitivity_ci(preds, gts, "mass", fppi=1.0, n_boot=2000, seed=1)
    # A box-level binomial CI for 50/100 is about 0.40-0.60; lesions cluster in images, so the image
    # bootstrap is close to a binomial over 10/20 images (about 0.28-0.72).
    assert ci["lower"] < 0.35 < 0.65 < ci["upper"]
    assert ci == froc_sensitivity_ci(preds, gts, "mass", fppi=1.0, n_boot=2000, seed=1)  # seeded


def test_gate_lower_bound_is_stricter_than_point():
    from radio.train.train_detector import DetTrainConfig, gate_class

    preds, gts = _clustered_case()
    te = evaluate_detections(preds, gts, ["mass"])["mass"]
    base = dict(gate_min_sensitivity=0.45, gate_min_lesions=30, gate_bootstrap=500)
    ok_point, g_point = gate_class(DetTrainConfig(gate_statistic="point", **base), preds, gts, "mass", te)
    ok_lb, g_lb = gate_class(DetTrainConfig(**base), preds, gts, "mass", te)
    assert ok_point and not ok_lb
    assert g_lb["gate_statistic"] == "lower_bound"
    assert g_lb["test_sens_ci_lower"] < 0.45 <= g_lb["test_sens_at_gate_fppi"]
    few, _ = gate_class(DetTrainConfig(gate_statistic="point", gate_min_sensitivity=0.1, gate_min_lesions=101),
                        preds, gts, "mass", te)
    assert not few  # 100 boxes < 101 required
    with pytest.raises(ValueError):
        gate_class(DetTrainConfig(gate_statistic="median"), preds, gts, "mass", te)


def test_ap_duplicate_detection_is_fp():
    tp = np.array([True, False])
    assert average_precision(tp, 1) == pytest.approx(1.0)
    assert average_precision(np.array([False, True]), 1) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def fake_chestx_det(tmp_path_factory):
    root = tmp_path_factory.mktemp("cxdet")
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("test", 8)):
        (root / split).mkdir()
        anns = []
        for i in range(n):
            img = (rng.random((128, 128)) * 60).astype(np.uint8)
            x, y = int(rng.integers(10, 70)), int(rng.integers(10, 70))
            img[y:y + 40, x:x + 40] = 230
            name = f"{split}{i}.png"
            Image.fromarray(img).save(root / split / name)
            anns.append({"file_name": name, "syms": ["Effusion"], "boxes": [[x, y, x + 40, y + 40]],
                         "polygons": []})
        anns.append({"file_name": "missing.png", "syms": [], "boxes": [], "polygons": []})
        (root / f"ChestX_Det_{split}.json").write_text(json.dumps(anns))
    return root


def test_loader_maps_classes_and_skips_missing(fake_chestx_det):
    from radio.data.chestx_det import DET_CONCEPTS, load_split, split_train_val

    recs = load_split(fake_chestx_det, "train")
    assert len(recs) == 24 and recs[0].labels == ["pleural_effusion"]
    assert "nodule" in DET_CONCEPTS and "Diffuse Nodule" not in DET_CONCEPTS and len(DET_CONCEPTS) == 12
    tr, va = split_train_val(recs, 0.25)
    assert not {r.file_name for r in tr} & {r.file_name for r in va} and len(tr) + len(va) == 24


def test_fingerprint_matches_finds_duplicates(fake_chestx_det, tmp_path):
    from radio.data.chestx_det import fingerprint_matches

    src = fake_chestx_det / "train" / "train0.png"
    dup = tmp_path / "00000001_000.png"
    Image.open(src).resize((256, 256)).save(dup)
    other = tmp_path / "00000002_000.png"
    Image.fromarray(np.full((64, 64), 7, np.uint8) + np.eye(64, dtype=np.uint8) * 200).save(other)
    m = fingerprint_matches([src, fake_chestx_det / "train" / "train1.png"], [dup, other])
    assert m == {"train0.png": "00000001_000.png"}


@pytest.fixture(scope="module")
def det_run(fake_chestx_det, tmp_path_factory):
    from radio.train.train_detector import DetTrainConfig, train_detector

    cfg = DetTrainConfig(run_name="d", out_dir=str(tmp_path_factory.mktemp("druns")), root=str(fake_chestx_det),
                         val_frac=0.25, min_size=128, pretrained_backbone=False, epochs=1, batch_size=4,
                         num_workers=0, device="cpu", warmup_iters=5, gate_min_lesions=1, gate_min_sensitivity=0.0)
    return train_detector(cfg)


def test_detector_training_writes_bundle(det_run):
    from radio.models.detector import DetectorBundle

    b = DetectorBundle(det_run / "det_bundle")
    assert b.research_only is False  # random init + OK dataset
    ev = json.loads((det_run / "eval_test.json").read_text())
    assert "map25" in ev["_summary"] and "sens@1.0fp" in ev["pleural_effusion"]
    assert set(b.operating_points) == set(b.concepts)
    img = np.zeros((128, 128), np.float32)
    assert isinstance(b.detect(img), list)


def test_imagenet_backbone_is_research_only():
    from radio.data.registry import LicenceGateError, lineage

    assert lineage(["chestx_det"], "torchvision_resnet50_imagenet", "research")["research_only"] is True
    with pytest.raises(LicenceGateError):
        lineage(["chestx_det"], "torchvision_resnet50_imagenet", "product")


def test_pipeline_uses_only_validated_detector_classes(det_run, tmp_path):
    """Validated classes -> image_observed + validated localisation; unvalidated -> boxes never validated."""
    from radio.models.detector import Detection, DetectorBundle
    from radio.pipeline.run import merge_detections
    from tests.test_pipeline import _pipeline  # noqa: F401  (reuse fixtures indirectly)

    b = DetectorBundle(det_run / "det_bundle")
    for op in b.operating_points.values():
        op.validated = op.concept == "pleural_effusion"
        op.box_score_threshold = 0.5
    dets = [Detection("pleural_effusion", 0.9, (10, 10, 60, 60)), Detection("pleural_effusion", 0.2, (0, 0, 5, 5)),
            Detection("mass", 0.95, (70, 70, 90, 90))]
    locs = merge_detections(dets, b)
    assert [d.score for d in locs["pleural_effusion"]] == [0.9]
    assert all(loc.validated for loc in locs["pleural_effusion"])
    assert "mass" not in locs or not any(loc.validated for loc in locs["mass"])
    assert math.isclose(locs["pleural_effusion"][0].bbox.x_max, 60)


def test_end_to_end_with_detector(bundle_dir, det_run, tmp_path):
    """Classifier + detector through the full pipeline; report must validate with image_observed findings."""
    from radio.ingest.synthetic import phantom, write_dx_dicom
    from radio.models.detector import Detection, DetectorBundle
    from radio.pipeline.bundle import ModelBundle
    from radio.pipeline.run import AnalysisPipeline
    from radio.schema.findings import FindingStatus, Provenance

    det = DetectorBundle(det_run / "det_bundle")
    for op in det.operating_points.values():
        op.validated = op.concept in ("pleural_effusion", "fracture")
        op.box_score_threshold = 0.5
    det.detect = lambda img: [Detection("pleural_effusion", 0.9, (100, 300, 200, 450)),
                              Detection("fracture", 0.8, (300, 50, 340, 90))]
    clf = ModelBundle.load(bundle_dir)
    clf.ood.threshold = math.inf
    r = AnalysisPipeline(clf, detector=det).analyze(write_dx_dicom(phantom(size=512), tmp_path / "x.dcm")).result
    by = {f.code: f for f in r.findings}
    assert r.report is not None and r.report.validation_passed, r.report.validation_issues
    frac = by["fracture"]
    assert frac.status == FindingStatus.present and frac.provenance == Provenance.image_observed
    assert frac.localizations[0].validated and frac.localizations[0].method.value == "detector"
    eff = by["pleural_effusion"]
    assert eff.status in (FindingStatus.present, FindingStatus.uncertain)  # validated box can't be called absent
    assert eff.provenance == Provenance.image_observed
    assert by["calcification"].status == FindingStatus.not_assessed
    assert by["calcification"].not_assessed_reason == "detector_not_validated"
    assert "radio-cxr-detector" in r.model_versions


def test_patient_leak_report_and_regate(fake_chestx_det, det_run, tmp_path):
    import pandas as pd

    from radio.data.chestx_det import patient_leak_report
    from radio.train.train_detector import DetTrainConfig, regate_detector

    # train0 and test0 come from the same NIH patient "7"; test1 from a train-unseen patient.
    matches = {"train0.png": "00000007_000.png", "test0.png": "00000007_001.png", "test1.png": "00000009_000.png"}
    csv = tmp_path / "nih.csv"
    pd.DataFrame({"Image Index": list(matches.values()), "Patient ID": [7, 7, 9]}).to_csv(csv, index=False)
    rep = patient_leak_report(fake_chestx_det, matches, csv)
    assert rep["leaky_test_images"] == ["test0.png"] and rep["test_images_mapped_to_nih"] == 2

    cfg = DetTrainConfig.from_yaml(det_run / "config.yaml", {"gate_min_lesions": 10_000})
    res = regate_detector(det_run, set(rep["leaky_test_images"]), cfg)
    assert res["validated"] == []  # impossible lesion minimum -> nothing validated
    assert res["leak_reduced"]["_summary"]["n_images"] == 7 and res["leaky_only"]["_summary"]["n_images"] == 1
    ops = json.loads((det_run / "det_bundle" / "operating_points.json").read_text())
    assert all(not o["validated"] and o["gate"]["test_split"] == "official_minus_known_patient_leak" for o in ops)
    assert all(o["gate"]["gate_statistic"] == "lower_bound" and "test_sens_ci_lower" in o["gate"] for o in ops)
