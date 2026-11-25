"""Clinical consistency checker (stage 11): rule-based, deterministic, fail-closed.

Checks every report sentence against the findings IR:

* references resolve to IR findings and sentence kind matches finding status
  (positive->present, uncertain->uncertain, negative->absent, not_assessed->not_assessed)
* no unsupported mentions: a finding term in a sentence must belong to a finding the sentence cites
* uncertain findings keep hedging language; no comparison language unless a prior study is linked
* negation polarity (positive statements contain no negation; negative statements do)
* laterality: no side words unless the cited finding has that laterality
* numbers in finding sentences must match measurements of cited findings
* "normal" summary only when every core acute concept was assessed and is absent
* every present/uncertain finding is reported; critical findings appear in the impression
* not-assessed findings are disclosed; recommendations cite an approved guideline; disclaimer present
"""

from __future__ import annotations

import re

from radio.ontology.cxr import CONCEPTS, CORE_ACUTE_CONCEPTS
from radio.report.guidelines import APPROVED_GUIDELINES
from radio.schema.findings import (
    AI_DISCLAIMER,
    AnalysisResult,
    DraftReport,
    FindingStatus,
    Laterality,
    Provenance,
    ReportSentence,
    SentenceKind,
    ValidationIssue,
)

_KIND_TO_STATUS = {
    SentenceKind.positive: {FindingStatus.present},
    SentenceKind.uncertain: {FindingStatus.uncertain},
    SentenceKind.negative: {FindingStatus.absent},
    SentenceKind.not_assessed: {FindingStatus.not_assessed},
    SentenceKind.normal_summary: {FindingStatus.absent},
}
_NEGATION = re.compile(r"\b(no|not|without|negative for|absence of|absent|free of|resolved)\b", re.I)
_SIDE = re.compile(r"\b(left|right|bilateral|bilaterally)\b", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_HEDGE = re.compile(r"\b(possible|possibly|may|might|cannot be excluded|cannot exclude|indeterminate|suspected|"
                    r"questionable|equivocal|probable)\b", re.I)
_COMPARISON = re.compile(r"\b(compared|comparison with|unchanged|interval|stable|improved|worsened|"
                         r"previously|new since|since the prior|resolved)\b", re.I)
_CONTEXT_PROVENANCE = {Provenance.clinical_history, Provenance.prior_report, Provenance.study_metadata}

_TERM_PATTERNS: dict[str, re.Pattern[str]] = {
    key: re.compile(r"\b(" + "|".join(re.escape(t) for t in c.terms) + r")\b", re.I)
    for key, c in CONCEPTS.items()
    if c.terms
}


def _mentioned_concepts(text: str) -> set[str]:
    return {k for k, p in _TERM_PATTERNS.items() if p.search(text)}


def validate_sentence(
    s: ReportSentence, result: AnalysisResult, section: str | None = None, index: int | None = None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def err(code: str, msg: str, severity: str = "error") -> None:
        issues.append(ValidationIssue(code=code, severity=severity, message=msg, section=section,
                                      sentence_index=index))

    by_id = {f.id: f for f in result.findings}
    linked = []
    for fid in s.finding_ids:
        f = by_id.get(fid)
        if f is None:
            err("unknown_finding_ref", f"sentence cites unknown finding id {fid}")
        else:
            linked.append(f)

    if s.kind in (SentenceKind.context, SentenceKind.quality):
        if s.kind == SentenceKind.context and s.provenance not in _CONTEXT_PROVENANCE:
            err("context_provenance", "context sentences must come from clinical history, priors or metadata")
        if s.kind == SentenceKind.quality:
            mentioned = _mentioned_concepts(s.text)
            if mentioned:
                err("unsupported_mention", f"quality sentence mentions findings {sorted(mentioned)}")
        return issues

    if not s.finding_ids:
        err("ungrounded_sentence", "finding/impression sentence cites no findings")
        return issues

    allowed = _KIND_TO_STATUS[s.kind]
    for f in linked:
        if f.status not in allowed:
            err("kind_status_mismatch", f"{s.kind.value} sentence cites {f.code} with status {f.status.value}")

    if s.kind == SentenceKind.normal_summary:
        statuses = {f.code: f.status for f in result.findings}
        if any(st in (FindingStatus.present, FindingStatus.uncertain) for st in statuses.values()):
            err("normal_with_positive", "normal summary while present/uncertain findings exist")
        missing = [c for c in CORE_ACUTE_CONCEPTS if statuses.get(c) != FindingStatus.absent]
        if missing:
            err("normal_without_assessment", f"normal summary but core concepts not assessed-absent: {missing}")

    cited_codes = {f.code for f in linked}
    extra = _mentioned_concepts(s.text) - cited_codes
    if extra and s.kind != SentenceKind.normal_summary:
        err("unsupported_mention", f"sentence mentions {sorted(extra)} not among cited findings")

    if s.kind == SentenceKind.uncertain and not _HEDGE.search(s.text):
        err("uncertainty_not_preserved", "uncertain finding stated without hedging language")
    if _COMPARISON.search(s.text) and not result.clinical_context.priors:
        err("unsupported_comparison", "comparison language without a linked prior study")

    has_neg = bool(_NEGATION.search(s.text))
    if s.kind in (SentenceKind.positive, SentenceKind.uncertain) and has_neg:
        err("negation_in_positive", "positive/uncertain sentence contains negation language")
    if s.kind in (SentenceKind.negative, SentenceKind.normal_summary) and not has_neg:
        err("missing_negation", "negative sentence lacks negation language")

    sides = {m.lower().replace("bilaterally", "bilateral") for m in _SIDE.findall(s.text)}
    if sides:
        allowed_sides = {f.laterality.value for f in linked if f.laterality in
                         (Laterality.left, Laterality.right, Laterality.bilateral)}
        bad = sides - allowed_sides
        if bad:
            err("laterality_unsupported", f"sentence states side {sorted(bad)} not supported by cited findings")

    nums = {float(n) for n in _NUMBER.findall(s.text)}
    if nums:
        known = {float(m.value) for f in linked for m in f.measurements}
        bad_nums = {n for n in nums if not any(abs(n - k) < 1e-6 for k in known)}
        if bad_nums:
            err("unsupported_number", f"numbers {sorted(bad_nums)} not present in cited measurements")
    return issues


def validate_report(report: DraftReport, result: AnalysisResult) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for name, sents in report.sections().items():
        for i, s in enumerate(sents):
            issues.extend(validate_sentence(s, result, name, i))

    if report.disclaimer != AI_DISCLAIMER or not report.ai_generated:
        issues.append(ValidationIssue(code="missing_disclaimer", severity="error",
                                      message="AI-generated disclaimer missing or altered"))

    reported = {fid for s in report.findings for fid in s.finding_ids if s.kind in
                (SentenceKind.positive, SentenceKind.uncertain)}
    for f in result.findings:
        if f.status in (FindingStatus.present, FindingStatus.uncertain) and f.id not in reported:
            issues.append(ValidationIssue(code="omitted_finding", severity="error",
                                          message=f"{f.code} ({f.status.value}) not reported in findings"))

    in_impression = {fid for s in report.impression for fid in s.finding_ids}
    for f in result.findings:
        if f.critical and f.id not in in_impression:
            issues.append(ValidationIssue(code="critical_not_in_impression", severity="error",
                                          message=f"critical finding {f.code} missing from impression"))

    disclosed = {fid for s in report.findings for fid in s.finding_ids if s.kind == SentenceKind.not_assessed}
    for f in result.findings:
        if f.status == FindingStatus.not_assessed and f.id not in disclosed:
            issues.append(ValidationIssue(code="not_assessed_undisclosed", severity="error",
                                          message=f"{f.code} not assessed but not disclosed"))

    for r in report.recommendations:
        if r.guideline_id not in APPROVED_GUIDELINES:
            issues.append(ValidationIssue(code="unsupported_recommendation", severity="error",
                                          message=f"recommendation cites unapproved guideline '{r.guideline_id}'"))
        if any(fid not in {f.id for f in result.findings} for fid in r.finding_ids):
            issues.append(ValidationIssue(code="unknown_finding_ref", severity="error",
                                          message="recommendation cites unknown finding"))
    return issues
