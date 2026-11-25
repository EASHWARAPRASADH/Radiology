"""Structured intermediate representation (IR) for AI analysis of an imaging study.

Design rules enforced here (see docs/blueprint/06-recommended-architecture.md):

* Every finding has an explicit ``status``: present | absent | uncertain | not_assessed.
  Missing information is never a negative finding - unassessed concepts are ``not_assessed``.
* Every finding and every report sentence carries ``provenance`` so directly observed image findings,
  model inferences, clinical history and prior-report content are never conflated.
* AI output is always a draft. ``review_status`` can only become approved/amended/rejected through a
  recorded radiologist review.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.1.0"

AI_DISCLAIMER = (
    "AI-GENERATED DRAFT - not reviewed. For radiologist review only; not for clinical decision-making "
    "until approved by a qualified radiologist."
)


def _now() -> datetime:
    return datetime.now(UTC)


def _id() -> str:
    return uuid.uuid4().hex


class FindingStatus(StrEnum):
    present = "present"
    absent = "absent"
    uncertain = "uncertain"
    not_assessed = "not_assessed"


class Provenance(StrEnum):
    image_observed = "image_observed"  # AI detection with spatial evidence in this image
    model_inferred = "model_inferred"  # AI classification/inference without validated localization
    clinical_history = "clinical_history"  # supplied by the requester (indication, history)
    prior_report = "prior_report"  # taken from a prior report
    radiologist_entered = "radiologist_entered"  # added or edited by a radiologist
    study_metadata = "study_metadata"  # from DICOM header / order (modality, view), not an image finding


class Laterality(StrEnum):
    left = "left"
    right = "right"
    bilateral = "bilateral"
    not_applicable = "not_applicable"
    unknown = "unknown"


class Severity(StrEnum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


class ReviewStatus(StrEnum):
    draft = "draft"
    in_review = "in_review"
    approved = "approved"
    amended = "amended"
    rejected = "rejected"


class QualityStatus(StrEnum):
    adequate = "adequate"
    limited = "limited"
    non_diagnostic = "non_diagnostic"
    not_assessed = "not_assessed"


class CriticalStatus(StrEnum):
    none_detected = "none_detected"
    suspected = "suspected"  # AI suspects a critical finding: escalate for radiologist confirmation
    not_assessed = "not_assessed"


class UncertaintyStatus(StrEnum):
    low = "low"
    elevated = "elevated"  # at least one finding is uncertain or quality is limited
    high = "high"  # OOD / abstained


class LocalizationMethod(StrEnum):
    detector = "detector"
    segmentation = "segmentation"
    activation_map = "activation_map"  # coarse; not validated for lesion localization
    radiologist = "radiologist"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class BoundingBox(_Model):
    """Axis-aligned box in source-image pixel coordinates (column=x, row=y, origin top-left)."""

    x_min: float = Field(ge=0)
    y_min: float = Field(ge=0)
    x_max: float = Field(ge=0)
    y_max: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")
        return self


class Localization(_Model):
    method: LocalizationMethod
    bbox: BoundingBox | None = None
    heatmap_uri: str | None = None
    region: str | None = Field(None, description="Coarse anatomical region, e.g. 'left lower zone'.")
    score: float | None = Field(None, ge=0, le=1)
    validated: bool = Field(
        False, description="True only if this localization method was validated against expert boxes."
    )


class Measurement(_Model):
    name: str
    value: float
    unit: str
    method: str
    provenance: Provenance


class ModelEvidence(_Model):
    model_id: str
    model_version: str
    raw_probability: float = Field(ge=0, le=1)
    calibrated_probability: float = Field(ge=0, le=1)
    threshold_absent: float = Field(ge=0, le=1)
    threshold_present: float = Field(ge=0, le=1)


class Finding(_Model):
    id: str = Field(default_factory=_id)
    code: str = Field(description="Internal concept key (radio.ontology.cxr.CONCEPTS).")
    display: str
    status: FindingStatus
    provenance: Provenance
    laterality: Laterality = Laterality.unknown
    location: str | None = None
    localizations: list[Localization] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    severity: Severity | None = Field(None, description="Only populated by a validated severity method.")
    evidence: ModelEvidence | None = None
    uncertainty_reasons: list[str] = Field(default_factory=list)
    not_assessed_reason: str | None = None
    critical: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Finding:
        if self.status == FindingStatus.not_assessed:
            if not self.not_assessed_reason:
                raise ValueError(f"{self.code}: not_assessed findings must state not_assessed_reason")
            if self.localizations or self.measurements:
                raise ValueError(f"{self.code}: not_assessed findings cannot carry localization/measurements")
        if self.status == FindingStatus.absent and (self.localizations or self.measurements or self.critical):
            raise ValueError(f"{self.code}: absent findings cannot carry localization, measurements or critical")
        if self.status == FindingStatus.uncertain and not self.uncertainty_reasons:
            raise ValueError(f"{self.code}: uncertain findings must list uncertainty_reasons")
        if self.provenance in (Provenance.image_observed, Provenance.model_inferred) and self.status in (
            FindingStatus.present, FindingStatus.uncertain, FindingStatus.absent
        ) and self.evidence is None:
            raise ValueError(f"{self.code}: AI-asserted findings must include model evidence")
        if self.provenance == Provenance.image_observed and self.status in (
            FindingStatus.present, FindingStatus.uncertain
        ) and not any(loc.validated for loc in self.localizations):
            raise ValueError(f"{self.code}: image_observed requires a validated localization; use model_inferred")
        if self.critical and self.status not in (FindingStatus.present, FindingStatus.uncertain):
            raise ValueError(f"{self.code}: only present/uncertain findings can be flagged critical")
        return self


class StudyInfo(_Model):
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    modality: str
    anatomical_region: str
    view_position: str | None = None
    source_format: Literal["dicom", "png", "jpeg"]
    image_rows: int | None = None
    image_columns: int | None = None


class PriorFinding(_Model):
    """Final, radiologist-verified status of one concept on a prior study."""

    code: str
    status: FindingStatus
    laterality: Laterality = Laterality.unknown


class PriorStudy(_Model):
    id: str = Field(default_factory=_id)
    study_date: str | None = None
    report_text: str | None = None
    provenance: Literal[Provenance.prior_report] = Provenance.prior_report
    findings: list[PriorFinding] = Field(
        default_factory=list,
        description="Structured findings of the prior, only if a radiologist verified them (see findings_verified_by).",
    )
    findings_verified_by: str | None = None

    @model_validator(mode="after")
    def _verified(self) -> PriorStudy:
        if self.findings and not self.findings_verified_by:
            raise ValueError("prior findings must name the radiologist who verified them")
        if any(f.status not in (FindingStatus.present, FindingStatus.absent) for f in self.findings):
            raise ValueError("prior findings may only be present or absent")
        if len({f.code for f in self.findings}) != len(self.findings):
            raise ValueError("each concept may appear at most once per prior")
        return self

    @property
    def comparable(self) -> bool:
        return bool(self.findings and self.findings_verified_by)


class ClinicalContext(_Model):
    indication: str | None = None
    history: str | None = None
    priors: list[PriorStudy] = Field(default_factory=list)


class ImageQuality(_Model):
    status: QualityStatus
    issues: list[str] = Field(default_factory=list)
    ood_score: float | None = None
    ood_threshold: float | None = None
    out_of_distribution: bool | None = None


class SentenceKind(StrEnum):
    positive = "positive"
    negative = "negative"
    uncertain = "uncertain"
    not_assessed = "not_assessed"
    normal_summary = "normal_summary"
    context = "context"  # clinical/study context, not an image finding
    quality = "quality"
    comparison = "comparison"  # change of one finding against a radiologist-verified prior


class ComparisonChange(StrEnum):
    new = "new"  # prior absent -> current present
    persistent = "persistent"  # prior present -> current present
    resolved = "resolved"  # prior present -> current absent


class ReportSentence(_Model):
    text: str
    kind: SentenceKind
    finding_ids: list[str] = Field(default_factory=list)
    provenance: Provenance
    prior_id: str | None = None  # comparison sentences: the PriorStudy compared against
    change: ComparisonChange | None = None  # comparison sentences: the claimed change


class Recommendation(_Model):
    text: str
    guideline_id: str = Field(description="Identifier of the approved guideline in the clinical knowledge base.")
    finding_ids: list[str] = Field(min_length=1)


class ValidationIssue(_Model):
    code: str
    severity: Literal["error", "warning"]
    message: str
    section: str | None = None
    sentence_index: int | None = None


class DraftReport(_Model):
    ai_generated: Literal[True] = True
    disclaimer: str = AI_DISCLAIMER
    study_section: list[ReportSentence] = Field(default_factory=list)
    clinical_section: list[ReportSentence] = Field(default_factory=list)
    technique_quality: list[ReportSentence] = Field(default_factory=list)
    findings: list[ReportSentence] = Field(default_factory=list)
    comparison: list[ReportSentence] = Field(default_factory=list)
    impression: list[ReportSentence] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    generator: str
    validation_passed: bool = False
    validation_issues: list[ValidationIssue] = Field(default_factory=list)

    def sections(self) -> dict[str, list[ReportSentence]]:
        return {
            "study": self.study_section,
            "clinical": self.clinical_section,
            "technique_quality": self.technique_quality,
            "findings": self.findings,
            "comparison": self.comparison,
            "impression": self.impression,
        }

    def to_text(self) -> str:
        def block(title: str, sents: list[ReportSentence]) -> str:
            body = " ".join(s.text for s in sents) if sents else "None."
            return f"{title}:\n{body}"

        parts = [
            self.disclaimer,
            block("STUDY", self.study_section),
            block("CLINICAL INFORMATION", self.clinical_section),
            block("TECHNIQUE / QUALITY", self.technique_quality),
            block("FINDINGS", self.findings),
            *([block("COMPARISON", self.comparison)] if self.comparison else []),
            block("IMPRESSION", self.impression),
            "RECOMMENDATIONS:\n" + (" ".join(r.text for r in self.recommendations) or "None."),
        ]
        return "\n\n".join(parts)


class ReviewAction(StrEnum):
    approve = "approve"
    amend = "amend"
    reject = "reject"


class FindingOverride(_Model):
    finding_id: str
    status: FindingStatus
    note: str | None = None


class Review(_Model):
    id: str = Field(default_factory=_id)
    reviewer_id: str = Field(min_length=1)
    action: ReviewAction
    timestamp: datetime = Field(default_factory=_now)
    final_report_text: str | None = None
    finding_overrides: list[FindingOverride] = Field(default_factory=list)
    comment: str | None = None

    @model_validator(mode="after")
    def _amend_has_text(self) -> Review:
        if self.action == ReviewAction.amend and not (self.final_report_text or self.finding_overrides):
            raise ValueError("amend requires edited report text or finding overrides")
        return self


class Abstention(_Model):
    reason: str
    detail: str | None = None


_REVIEW_TO_STATUS = {
    ReviewAction.approve: ReviewStatus.approved,
    ReviewAction.amend: ReviewStatus.amended,
    ReviewAction.reject: ReviewStatus.rejected,
}


class AnalysisResult(_Model):
    """Top-level machine-readable output for one analysed image/study."""

    schema_version: str = SCHEMA_VERSION
    result_id: str = Field(default_factory=_id)
    created_at: datetime = Field(default_factory=_now)
    study: StudyInfo
    clinical_context: ClinicalContext = Field(default_factory=ClinicalContext)
    image_quality: ImageQuality
    findings: list[Finding] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    critical_status: CriticalStatus
    uncertainty_status: UncertaintyStatus
    abstention: Abstention | None = None
    report: DraftReport | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.draft
    reviews: list[Review] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> AnalysisResult:
        if self.abstention is not None:
            if self.report is not None:
                raise ValueError("abstained results must not carry a report")
            if any(f.status != FindingStatus.not_assessed for f in self.findings):
                raise ValueError("abstained results may only contain not_assessed findings")
        codes = [f.code for f in self.findings]
        if len(codes) != len(set(codes)):
            raise ValueError("each concept may appear at most once per result")
        if self.review_status in (ReviewStatus.approved, ReviewStatus.amended, ReviewStatus.rejected):
            if not self.reviews:
                raise ValueError(f"review_status={self.review_status} requires a recorded radiologist review")
            expected = _REVIEW_TO_STATUS[self.reviews[-1].action]
            if expected != self.review_status:
                raise ValueError("review_status does not match the latest review action")
        has_critical = any(f.critical for f in self.findings)
        if has_critical and self.critical_status != CriticalStatus.suspected:
            raise ValueError("critical findings present but critical_status is not 'suspected'")
        return self

    def radiologist_text(self) -> tuple[str, str] | None:
        """(reviewer_id, text) of the most recent review that supplied report text, if any."""
        for rv in reversed(self.reviews):
            if rv.final_report_text:
                return rv.reviewer_id, rv.final_report_text
        return None

    def final_reviewer(self) -> str | None:
        return self.reviews[-1].reviewer_id if self.reviews else None

    def finding(self, finding_id: str) -> Finding | None:
        return next((f for f in self.findings if f.id == finding_id), None)

    def apply_review(self, review: Review) -> AnalysisResult:
        """Return a new result with the review appended. The AI draft itself is never mutated."""
        if self.review_status in (ReviewStatus.approved, ReviewStatus.rejected):
            raise ValueError(f"result already finalised ({self.review_status})")
        if review.action != ReviewAction.reject and self.abstention is not None and not review.final_report_text:
            raise ValueError("abstained study has no AI draft; approval requires radiologist-authored text")
        if (review.action == ReviewAction.approve and self.report is not None
                and not self.report.validation_passed and not review.final_report_text):
            raise ValueError("draft failed consistency validation; amend with radiologist text or reject")
        known = {f.id for f in self.findings}
        unknown = [o.finding_id for o in review.finding_overrides if o.finding_id not in known]
        if unknown:
            raise ValueError(f"overrides reference unknown findings: {unknown}")
        data = self.model_dump()
        data["reviews"] = [*data["reviews"], review.model_dump()]
        data["review_status"] = _REVIEW_TO_STATUS[review.action]
        return AnalysisResult.model_validate(data)


def json_schema() -> dict:
    schema = AnalysisResult.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://radio.local/schemas/analysis-result/{SCHEMA_VERSION}"
    schema["title"] = "Radio AnalysisResult"
    return schema
