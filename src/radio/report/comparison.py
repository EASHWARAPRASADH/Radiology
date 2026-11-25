"""Comparison with a prior study (Phase 3, WP 3.3).

The AI compares the current IR only against a prior's *radiologist-verified* structured findings, never
against an unreviewed AI draft or free-text prior report. Only definite statuses on both sides are compared:

    prior absent  -> current present : new
    prior present -> current present : persistent ("previously reported"; no claim about size or severity)
    prior present -> current absent  : resolved ("not identified on this study")

Uncertain or not-assessed current findings, absent -> absent pairs, and pairs whose known sides disagree are
not compared. Matching the prior to the same patient is the caller's responsibility; it is recorded in audit.
"""

from __future__ import annotations

from radio.schema.findings import (
    AnalysisResult,
    ComparisonChange,
    Finding,
    FindingStatus,
    Laterality,
    PriorFinding,
    PriorStudy,
    ReviewAction,
    ReviewStatus,
)

_SIDED = (Laterality.left, Laterality.right, Laterality.bilateral)


class PriorNotVerifiable(ValueError):
    pass


def prior_from_result(result: AnalysisResult, study_date: str | None = None) -> PriorStudy:
    """Turn an approved earlier analysis into a comparable prior.

    Allowed only when a radiologist approved the AI draft, or amended it through structured finding overrides
    alone. If the final report is radiologist free text, the structured findings may not match what the
    radiologist wrote, so the prior is refused (it can still be attached as report text without findings)."""
    if result.review_status not in (ReviewStatus.approved, ReviewStatus.amended) or not result.reviews:
        raise PriorNotVerifiable(f"prior is {result.review_status.value}, not radiologist-approved")
    last = result.reviews[-1]
    if last.final_report_text or any(rv.final_report_text for rv in result.reviews):
        raise PriorNotVerifiable("prior was finalised with radiologist free text; structured findings unverified")
    if last.action not in (ReviewAction.approve, ReviewAction.amend):
        raise PriorNotVerifiable(f"last review action is {last.action.value}")
    status = {f.id: f.status for f in result.findings}
    for rv in result.reviews:
        for o in rv.finding_overrides:
            status[o.finding_id] = o.status
    findings = [PriorFinding(code=f.code, status=status[f.id], laterality=f.laterality)
                for f in result.findings if status[f.id] in (FindingStatus.present, FindingStatus.absent)]
    return PriorStudy(study_date=study_date, findings=findings, findings_verified_by=last.reviewer_id)


def expected_change(current: Finding, prior: PriorFinding) -> ComparisonChange | None:
    if current.laterality in _SIDED and prior.laterality in _SIDED and current.laterality != prior.laterality:
        return None
    if current.status == FindingStatus.present:
        return ComparisonChange.new if prior.status == FindingStatus.absent else ComparisonChange.persistent
    if current.status == FindingStatus.absent and prior.status == FindingStatus.present:
        return ComparisonChange.resolved
    return None


def comparable_prior(result: AnalysisResult) -> PriorStudy | None:
    """The prior the generator compares against: the first one with radiologist-verified findings."""
    return next((p for p in result.clinical_context.priors if p.comparable), None)


def compare(result: AnalysisResult, prior: PriorStudy) -> list[tuple[Finding, ComparisonChange]]:
    by_code = {p.code: p for p in prior.findings}
    out = []
    for f in result.findings:
        p = by_code.get(f.code)
        if p is None:
            continue
        ch = expected_change(f, p)
        if ch is not None:
            out.append((f, ch))
    return out
