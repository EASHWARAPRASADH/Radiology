"""End-to-end tests on a tiny synthetic dataset: train -> calibrate -> pipeline -> API -> exports.

Uses a randomly initialised resnet18 at 64 px on CPU, so no downloads are needed. Performance is
meaningless here; these tests check plumbing and safety invariants.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pydicom
import pytest
import torch
from fastapi.testclient import TestClient

from radio.audit.log import AuditLog
from radio.data.nih import load_nih
from radio.data.registry import LicenceGateError, lineage
from radio.ingest.synthetic import phantom, write_dx_dicom
from radio.models.classifier import ClassifierConfig, build_classifier
from radio.pipeline.bundle import ModelBundle
from radio.pipeline.run import AnalysisPipeline
from radio.schema.findings import FindingStatus, ReviewStatus


def test_patient_disjoint_splits(fake_nih):
    s = load_nih(fake_nih / "Data_Entry_2017.csv", fake_nih / "images", val_frac=0.3, test_frac=0.2)
    assert not set(s.train.patient_id) & set(s.test.patient_id)
    assert not set(s.train.patient_id) & set(s.val.patient_id)
    assert len(s.labels) == 14


def test_licence_gate():
    assert lineage(["nih_cxr14"], "vit_small_patch14_dinov2.lvd142m", "product")["research_only"] is False
    with pytest.raises(LicenceGateError):
        lineage(["nih_cxr14", "mimic_cxr"], "random", "product")
    with pytest.raises(LicenceGateError):
        lineage(["nih_cxr14"], "rad_dino", "product")
    assert lineage(["mimic_cxr"], "random", "research")["research_only"] is True


@pytest.mark.parametrize("encoder,size", [("resnet18", 64), ("vit_small_patch14_dinov2.lvd142m", 56)])
def test_activation_maps_decompose_logits(encoder, size):
    torch.manual_seed(0)
    m = build_classifier(ClassifierConfig(encoder=encoder, pretrained=False, img_size=size, labels=("a", "b"))).eval()
    x = torch.randn(2, 3, size, size)
    logits, feats, maps = m.forward_with_maps(x)
    recon = maps.flatten(2).mean(-1) + m.head.bias
    assert torch.allclose(recon, logits, atol=1e-4)
    assert maps.shape[:2] == (2, 2)


def _pipeline(bundle_dir, tmp_path, threshold=math.inf):
    b = ModelBundle.load(bundle_dir)
    b.ood.threshold = threshold
    return AnalysisPipeline(b, audit=AuditLog(tmp_path / "audit.jsonl"))


def test_pipeline_end_to_end(bundle_dir, tmp_path):
    pipe = _pipeline(bundle_dir, tmp_path)
    dcm = write_dx_dicom(phantom(size=512), tmp_path / "x.dcm")
    out = pipe.analyze(dcm)
    r = out.result
    assert r.abstention is None and r.report is not None
    assert r.report.validation_passed, r.report.validation_issues
    assert r.review_status == ReviewStatus.draft
    codes = {f.code: f for f in r.findings}
    assert codes["fracture"].status == FindingStatus.not_assessed  # outside NIH label space
    assert codes["fracture"].not_assessed_reason == "not_in_model_label_space"
    assert "Not assessed by the AI system" in r.report.to_text()
    assert pipe.audit.verify()


def test_pipeline_abstains_on_ood_and_bad_inputs(bundle_dir, tmp_path):
    pipe = _pipeline(bundle_dir, tmp_path, threshold=-1.0)
    r = pipe.analyze(write_dx_dicom(phantom(size=512), tmp_path / "x.dcm")).result
    assert r.abstention and r.abstention.reason == "out_of_distribution" and r.report is None
    assert all(f.status == FindingStatus.not_assessed for f in r.findings)

    pipe = _pipeline(bundle_dir, tmp_path)
    ct = write_dx_dicom(phantom(size=512), tmp_path / "ct.dcm", modality="CT")
    assert pipe.analyze(ct).result.abstention.reason == "unsupported_modality"
    lat = write_dx_dicom(phantom(size=512), tmp_path / "lat.dcm", view_position="LL")
    assert pipe.analyze(lat).result.abstention.reason == "unsupported_view"
    blank = write_dx_dicom(np.zeros((512, 512), np.uint16), tmp_path / "blank.dcm")
    assert pipe.analyze(blank).result.abstention.reason == "non_diagnostic_quality"


def test_api_review_flow_and_exports(bundle_dir, tmp_path):
    from radio.api.app import create_app

    app = create_app(bundle_dir, storage_dir=tmp_path / "store")
    app.state.pipeline.bundle.ood.threshold = math.inf
    c = TestClient(app)
    rad = {"X-User-Id": "rad-1", "X-User-Role": "radiologist"}
    tech = {"X-User-Id": "tech-1", "X-User-Role": "technologist"}
    dcm = write_dx_dicom(phantom(size=512), tmp_path / "x.dcm")

    assert c.post("/v1/studies", files={"file": ("x.dcm", dcm.read_bytes())}).status_code == 401
    resp = c.post("/v1/studies", files={"file": ("x.dcm", dcm.read_bytes())}, data={"indication": "cough"},
                  headers=tech)
    assert resp.status_code == 201, resp.text
    rid = resp.json()["result_id"]
    assert resp.json()["review_status"] == "draft"
    # Upload/review page and the decoded image used for box overlays
    page = c.get("/")
    assert page.status_code == 200 and "Radio" in page.text and "/v1/studies" in page.text
    assert c.get(f"/v1/studies/{rid}/image.png").status_code == 401
    png = c.get(f"/v1/studies/{rid}/image.png", headers=tech)
    assert png.status_code == 200 and png.headers["content-type"] == "image/png" and png.content[:4] == b"\x89PNG"
    assert c.get("/v1/health").json()["detector"] is False
    assert c.get(f"/v1/studies/{rid}/report", headers=tech).text.startswith("[DRAFT]")

    # FHIR: preliminary until approved
    assert c.get(f"/v1/studies/{rid}/fhir", headers=tech).json()["status"] == "preliminary"
    # SR: incomplete/unverified draft, readable by pydicom
    sr = pydicom.dcmread(io.BytesIO(c.get(f"/v1/studies/{rid}/sr", headers=tech).content))
    assert sr.Modality == "SR" and sr.CompletionFlag == "PARTIAL" and sr.VerificationFlag == "UNVERIFIED"

    assert c.post(f"/v1/reports/{rid}/review", json={"action": "approve"}, headers=tech).status_code == 403
    amended = c.post(f"/v1/reports/{rid}/review", headers=rad,
                     json={"action": "amend", "final_report_text": "Radiologist final text."})
    assert amended.json()["review_status"] == "amended"
    ok = c.post(f"/v1/reports/{rid}/review", json={"action": "approve"}, headers=rad)
    assert ok.status_code == 200 and ok.json()["review_status"] == "approved"
    assert c.post(f"/v1/reports/{rid}/review", json={"action": "approve"}, headers=rad).status_code == 409
    # Approval after an amendment must surface the radiologist's text, not the AI draft.
    final_report = c.get(f"/v1/studies/{rid}/report", headers=rad).text
    assert "Radiologist final text." in final_report and "AI-GENERATED DRAFT" not in final_report
    fhir = c.get(f"/v1/studies/{rid}/fhir", headers=rad).json()
    assert fhir["status"] == "final"
    import base64

    assert base64.b64decode(fhir["presentedForm"][0]["data"]).decode() == "Radiologist final text."
    sr_final = pydicom.dcmread(io.BytesIO(c.get(f"/v1/studies/{rid}/sr", headers=rad).content))
    assert sr_final.CompletionFlag == "COMPLETE" and sr_final.VerificationFlag == "VERIFIED"

    # Comparison priors: a study finalised with radiologist free text cannot supply verified findings.
    upload = {"file": ("x.dcm", dcm.read_bytes())}
    assert c.post("/v1/studies", files=upload, data={"prior_study_id": rid}, headers=tech).status_code == 409
    second = c.post("/v1/studies", files=upload, headers=tech).json()["result_id"]
    assert c.post("/v1/studies", files=upload, data={"prior_study_id": second}, headers=tech).status_code == 409
    assert c.post(f"/v1/reports/{second}/review", json={"action": "approve"}, headers=rad).status_code == 200
    third = c.post("/v1/studies", files=upload, headers=tech,
                   data={"prior_study_id": second, "prior_study_date": "2026-01-02"})
    assert third.status_code == 201, third.text
    prior = third.json()["clinical_context"]["priors"][0]
    assert prior["findings_verified_by"] == "rad-1" and prior["study_date"] == "2026-01-02"
    text = c.get(f"/v1/studies/{third.json()['result_id']}/report", headers=tech).text
    # With definite (present/absent) verified findings the draft compares; with none, it says it did not.
    expected = "radiologist-verified findings" if prior["findings"] else "comparison was not performed"
    assert expected in text
    assert c.get("/v1/health").json()["audit_chain_valid"] is True


def test_audit_tamper_detected(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append("e1", actor="a", subject="s")
    log.append("e2", actor="a", subject="s")
    assert log.verify()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    (tmp_path / "a.jsonl").write_text("\n".join([lines[0].replace('"e1"', '"eX"'), lines[1]]) + "\n")
    assert not log.verify()


def test_conformal_thresholds_control_fnr():
    from radio.calibrate.conformal import fit_thresholds

    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.2).astype(float)
    p = np.clip(np.where(y > 0, rng.beta(5, 2, n), rng.beta(2, 5, n)), 0, 1)
    half = n // 2
    th = fit_thresholds(p[:half, None], y[:half, None], np.ones((half, 1)), ["x"], np.ones(1),
                        alpha_fn=0.05, alpha_fp=0.05)[0]
    test_p, test_y = p[half:], y[half:]
    fnr = (test_p[test_y > 0] < th.t_absent).mean()
    fpr = (test_p[test_y == 0] >= th.t_present).mean()
    assert fnr <= 0.08 and fpr <= 0.08  # ~alpha up to sampling noise
    assert th.t_absent <= th.t_present
