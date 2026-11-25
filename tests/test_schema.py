import pytest
from pydantic import ValidationError

from radio.schema.findings import (
    Abstention,
    AnalysisResult,
    BoundingBox,
    FindingStatus,
    Localization,
    LocalizationMethod,
    Provenance,
    Review,
    ReviewAction,
    ReviewStatus,
    json_schema,
)

from .conftest import make_finding, make_result


def test_status_is_required():
    from radio.schema.findings import Finding

    with pytest.raises(ValidationError):
        Finding(code="pneumothorax", display="pneumothorax", provenance=Provenance.model_inferred)  # no status


def test_not_assessed_requires_reason_and_no_evidence_payload():
    with pytest.raises(ValidationError):
        make_finding("fracture", FindingStatus.not_assessed, not_assessed_reason=None)
    loc = Localization(method=LocalizationMethod.activation_map, bbox=BoundingBox(x_min=1, y_min=1, x_max=5, y_max=5))
    with pytest.raises(ValidationError):
        make_finding("fracture", FindingStatus.not_assessed, localizations=[loc])


def test_absent_cannot_be_critical_or_localized():
    with pytest.raises(ValidationError):
        make_finding("pneumothorax", FindingStatus.absent, critical=True)


def test_uncertain_needs_reason():
    with pytest.raises(ValidationError):
        make_finding("pneumothorax", FindingStatus.uncertain, uncertainty_reasons=[])


def test_image_observed_requires_validated_localization():
    unvalidated = Localization(method=LocalizationMethod.activation_map,
                               bbox=BoundingBox(x_min=1, y_min=1, x_max=5, y_max=5), validated=False)
    with pytest.raises(ValidationError):
        make_finding("pneumothorax", FindingStatus.present, provenance=Provenance.image_observed,
                     localizations=[unvalidated])
    ok = unvalidated.model_copy(update={"method": LocalizationMethod.detector, "validated": True})
    make_finding("pneumothorax", FindingStatus.present, provenance=Provenance.image_observed, localizations=[ok])


def test_bbox_must_be_positive():
    with pytest.raises(ValidationError):
        BoundingBox(x_min=5, y_min=5, x_max=5, y_max=9)


def test_abstained_result_cannot_have_report_or_assessed_findings():
    with pytest.raises(ValidationError):
        make_result({"pneumothorax": FindingStatus.absent}, abstention=Abstention(reason="ood"))
    r = make_result({"pneumothorax": FindingStatus.not_assessed}, abstention=Abstention(reason="ood"))
    assert r.report is None


def test_critical_requires_suspected_status(abnormal_result):
    data = abnormal_result.model_dump()
    data["critical_status"] = "none_detected"
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)


def test_cannot_claim_approved_without_review(normal_result):
    data = normal_result.model_dump()
    data["review_status"] = "approved"
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)


def test_review_flow(normal_result):
    approved = normal_result.apply_review(Review(reviewer_id="rad-1", action=ReviewAction.approve))
    assert approved.review_status == ReviewStatus.approved
    assert normal_result.review_status == ReviewStatus.draft  # original untouched
    with pytest.raises(ValueError):
        approved.apply_review(Review(reviewer_id="rad-2", action=ReviewAction.approve))


def test_amend_requires_content():
    with pytest.raises(ValidationError):
        Review(reviewer_id="rad-1", action=ReviewAction.amend)


def test_json_schema_has_tristate():
    s = json_schema()
    assert set(s["$defs"]["FindingStatus"]["enum"]) == {"present", "absent", "uncertain", "not_assessed"}
    assert s["title"] == "Radio AnalysisResult"
