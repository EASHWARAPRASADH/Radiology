from __future__ import annotations

import pytest

from radio.ontology.cxr import CONCEPTS
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
