from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from radio.ingest.synthetic import phantom
from radio.ontology.cxr import CONCEPTS
from radio.pipeline.bundle import build_bundle
from radio.schema.findings import (
    AnalysisResult,
    ClinicalContext,
    CriticalStatus,
    Finding,
    FindingStatus,
    ImageQuality,
    ModelEvidence,
    Provenance,
    QualityStatus,
    StudyInfo,
    UncertaintyStatus,
)
from radio.train.train_classifier import DataCfg, ModelCfg, OptimCfg, TrainConfig, train


def make_finding(code: str, status: FindingStatus, **kw) -> Finding:
    evidence = None
    if status != FindingStatus.not_assessed:
        p = {"present": 0.9, "uncertain": 0.5, "absent": 0.02}[status.value]
        evidence = ModelEvidence(model_id="test", model_version="0", raw_probability=p,
                                 calibrated_probability=p, threshold_absent=0.1, threshold_present=0.7)
    defaults = dict(
        code=code,
        display=CONCEPTS[code].display,
        status=status,
        provenance=Provenance.model_inferred,
        evidence=evidence,
        uncertainty_reasons=["between_thresholds"] if status == FindingStatus.uncertain else [],
        not_assessed_reason="not_in_label_space" if status == FindingStatus.not_assessed else None,
    )
    defaults.update(kw)
    return Finding(**defaults)


def make_result(statuses: dict[str, FindingStatus], **kw) -> AnalysisResult:
    findings = [make_finding(c, s, critical=kw.pop(f"critical_{c}", False)) for c, s in statuses.items()]
    crit = CriticalStatus.suspected if any(f.critical for f in findings) else CriticalStatus.none_detected
    return AnalysisResult(
        study=StudyInfo(modality="DX", anatomical_region="chest", view_position="PA", source_format="dicom"),
        clinical_context=kw.pop("clinical_context", ClinicalContext()),
        image_quality=ImageQuality(status=QualityStatus.adequate),
        findings=findings,
        critical_status=crit,
        uncertainty_status=UncertaintyStatus.low,
        **kw,
    )


@pytest.fixture
def normal_result() -> AnalysisResult:
    s = {c: FindingStatus.absent for c in
         ["pneumothorax", "pleural_effusion", "consolidation", "edema", "lung_opacity", "cardiomegaly"]}
    s["fracture"] = FindingStatus.not_assessed
    return make_result(s)


@pytest.fixture
def abnormal_result() -> AnalysisResult:
    return make_result(
        {
            "pneumothorax": FindingStatus.present,
            "pleural_effusion": FindingStatus.uncertain,
            "consolidation": FindingStatus.absent,
            "edema": FindingStatus.absent,
            "lung_opacity": FindingStatus.absent,
            "cardiomegaly": FindingStatus.present,
            "fracture": FindingStatus.not_assessed,
        },
        critical_pneumothorax=True,
        clinical_context=ClinicalContext(indication="Shortness of breath, query pneumothorax"),
    )


NIH_COLS = ["Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Gender",
            "View Position"]


@pytest.fixture(scope="session")
def fake_nih(tmp_path_factory):
    root = tmp_path_factory.mktemp("nih")
    img_dir = root / "images" / "images_001" / "images"
    img_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(160):
        arr = phantom(size=256, seed=i).astype(np.float32) / 4095
        labs = []
        if rng.random() < 0.4:
            arr[180:, :] = np.clip(arr[180:, :] + 0.4, 0, 1)
            labs.append("Effusion")
        if rng.random() < 0.3:
            labs.append("Infiltration")
        name = f"{i:08d}_000.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(img_dir / name)
        rows.append([name, "|".join(labs) or "No Finding", 0, str(i // 2), 50, "F", "PA"])
    pd.DataFrame(rows, columns=NIH_COLS).to_csv(root / "Data_Entry_2017.csv", index=False)
    return root


@pytest.fixture(scope="session")
def bundle_dir(fake_nih, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    cfg = TrainConfig(
        run_name="t", out_dir=str(out),
        data=DataCfg(dataset_ids=["synthetic"], csv=str(fake_nih / "Data_Entry_2017.csv"),
                     images=str(fake_nih / "images"), split_dir=None, val_frac=0.3, test_frac=0.2),
        model=ModelCfg(encoder="resnet18", pretrained=False, img_size=64),
        optim=OptimCfg(epochs=1, batch_size=16, num_workers=0, device="cpu", amp=False, max_ood_train_feats=64),
    )
    run = train(cfg)
    return build_bundle(run, min_positives=3, min_negatives=3)
