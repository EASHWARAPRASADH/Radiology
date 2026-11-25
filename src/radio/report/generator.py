"""Deterministic, evidence-grounded draft report generation from the validated findings IR.

The generator only verbalises what is in the IR. Every sentence carries the IDs of the findings it is
derived from, so the validator (and the reviewing radiologist) can trace each statement to evidence.

An optional ``Rephraser`` (e.g. a small constrained LLM) may restyle individual sentences. Each
rephrased sentence is re-validated in isolation and the template sentence is kept if it fails - the
language model can change wording, never content.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from radio.ontology.cxr import CONCEPTS, CORE_ACUTE_CONCEPTS
from radio.schema.findings import (
    AI_DISCLAIMER,
    AnalysisResult,
    DraftReport,
    Finding,
    FindingStatus,
    Laterality,
    Provenance,
    QualityStatus,
    ReportSentence,
    SentenceKind,
)

GENERATOR_ID = "radio.template-v0.1"


class Rephraser(Protocol):
    name: str

    def rephrase(self, sentence: ReportSentence, findings: list[Finding]) -> str: ...


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " or " + items[-1]


def _side(f: Finding) -> str:
    c = CONCEPTS[f.code]
    if not c.laterality_applicable:
        return ""
    return {Laterality.left: "left ", Laterality.right: "right ", Laterality.bilateral: "bilateral "}.get(
        f.laterality, ""
    )


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _positive_text(f: Finding) -> str:
    c = CONCEPTS[f.code]
    loc = f" in the {f.location}" if f.location else ""
    meas = "".join(f", {m.name} {m.value:g} {m.unit}" for m in f.measurements)
    if c.category == "diagnosis":
        return f"Appearances may represent {_side(f)}{c.display}{loc}{meas}."
    return _cap(f"{_side(f)}{c.display}{loc}{meas}.")


def _uncertain_text(f: Finding) -> str:
    c = CONCEPTS[f.code]
    return f"Possible {_side(f)}{c.display}; AI confidence is indeterminate and this requires radiologist assessment."


def _uncertain_group(fs: list[Finding]) -> ReportSentence:
    """One sentence for several indeterminate findings (keeps every finding id for traceability)."""
    if len(fs) == 1:
        return ReportSentence(text=_uncertain_text(fs[0]), kind=SentenceKind.uncertain, finding_ids=[fs[0].id],
                              provenance=fs[0].provenance)
    names = [f"{_side(f)}{CONCEPTS[f.code].display}" for f in fs]
    text = (f"Possible {', '.join(names[:-1])} and {names[-1]}; AI confidence is indeterminate for each and "
            "these require radiologist assessment.")
    prov = fs[0].provenance if len({f.provenance for f in fs}) == 1 else Provenance.model_inferred
    return ReportSentence(text=text, kind=SentenceKind.uncertain, finding_ids=[f.id for f in fs], provenance=prov)


def _by_priority(fs: Iterable[Finding]) -> list[Finding]:
    return sorted(fs, key=lambda f: CONCEPTS[f.code].priority)


def generate_report(result: AnalysisResult, rephraser: Rephraser | None = None) -> DraftReport:
    """Build a DraftReport from ``result`` (which must not be an abstention)."""
    if result.abstention is not None:
        raise ValueError("cannot generate a report for an abstained analysis")

    fs = result.findings
    present = _by_priority(f for f in fs if f.status == FindingStatus.present)
    uncertain = _by_priority(f for f in fs if f.status == FindingStatus.uncertain)
    absent = _by_priority(f for f in fs if f.status == FindingStatus.absent)
    not_assessed = _by_priority(f for f in fs if f.status == FindingStatus.not_assessed)

    st = result.study
    view = f", {st.view_position} view" if st.view_position else ", view not recorded"
    study_section = [
        ReportSentence(text=f"Chest radiograph ({st.modality}){view}.", kind=SentenceKind.context,
                       provenance=Provenance.study_metadata)
    ]

    ctx = result.clinical_context
    clinical: list[ReportSentence] = []
    if ctx.indication:
        clinical.append(ReportSentence(text=f"Indication (as provided): {ctx.indication.strip().rstrip('.')}.",
                                       kind=SentenceKind.context, provenance=Provenance.clinical_history))
    if ctx.history:
        clinical.append(ReportSentence(text=f"History (as provided): {ctx.history.strip().rstrip('.')}.",
                                       kind=SentenceKind.context, provenance=Provenance.clinical_history))
    clinical.append(ReportSentence(
        text=("Prior imaging is available; comparison was not performed by the AI system."
              if ctx.priors else "No prior imaging provided; no comparison performed."),
        kind=SentenceKind.context, provenance=Provenance.prior_report if ctx.priors else Provenance.study_metadata,
    ))

    q = result.image_quality
    technique: list[ReportSentence] = []
    if q.status in (QualityStatus.limited, QualityStatus.non_diagnostic):
        readable = ", ".join(i.split(":")[0].replace("_", " ") for i in q.issues) or "unspecified"
        technique.append(ReportSentence(text=f"Image quality is {q.status.value} ({readable}).",
                                        kind=SentenceKind.quality, provenance=Provenance.model_inferred))

    findings: list[ReportSentence] = []
    for f in present:
        findings.append(ReportSentence(text=_positive_text(f), kind=SentenceKind.positive, finding_ids=[f.id],
                                       provenance=f.provenance))
    if uncertain:
        findings.append(_uncertain_group(uncertain))
    if absent:
        findings.append(ReportSentence(
            text=f"No {_join([CONCEPTS[f.code].display for f in absent])} identified by AI analysis.",
            kind=SentenceKind.negative, finding_ids=[f.id for f in absent], provenance=Provenance.model_inferred,
        ))
    if not_assessed:
        findings.append(ReportSentence(
            text="Not assessed by the AI system: " + ", ".join(CONCEPTS[f.code].display for f in not_assessed) + ".",
            kind=SentenceKind.not_assessed, finding_ids=[f.id for f in not_assessed],
            provenance=Provenance.model_inferred,
        ))

    impression: list[ReportSentence] = []
    critical = [f for f in present + uncertain if f.critical]
    for f in critical:
        impression.append(ReportSentence(
            text=f"Suspected {_side(f)}{CONCEPTS[f.code].display}: possible critical finding flagged by AI; "
                 "requires immediate radiologist confirmation.",
            kind=SentenceKind.positive if f.status == FindingStatus.present else SentenceKind.uncertain,
            finding_ids=[f.id], provenance=f.provenance,
        ))
    for f in present:
        if not f.critical:
            impression.append(ReportSentence(text=_positive_text(f), kind=SentenceKind.positive,
                                             finding_ids=[f.id], provenance=f.provenance))
    non_critical_uncertain = [f for f in uncertain if not f.critical]
    if non_critical_uncertain:
        impression.append(_uncertain_group(non_critical_uncertain))
    if not present and not uncertain:
        by_code = {f.code: f for f in fs}
        core_absent = all(c in by_code and by_code[c].status == FindingStatus.absent for c in CORE_ACUTE_CONCEPTS)
        if core_absent:
            impression.append(ReportSentence(
                text="No acute cardiopulmonary abnormality identified by AI analysis among the assessed findings.",
                kind=SentenceKind.normal_summary, finding_ids=[by_code[c].id for c in CORE_ACUTE_CONCEPTS],
                provenance=Provenance.model_inferred,
            ))
        elif absent:
            impression.append(ReportSentence(
                text="No abnormality identified by AI analysis among the assessed findings; "
                     "some findings were not assessed.",
                kind=SentenceKind.negative, finding_ids=[f.id for f in absent], provenance=Provenance.model_inferred,
            ))

    report = DraftReport(
        disclaimer=AI_DISCLAIMER,
        study_section=study_section,
        clinical_section=clinical,
        technique_quality=technique,
        findings=findings,
        impression=impression,
        recommendations=[],  # only from APPROVED_GUIDELINES; none approved in the prototype
        generator=GENERATOR_ID if rephraser is None else f"{GENERATOR_ID}+{rephraser.name}",
    )

    if rephraser is not None:
        from radio.report.validator import validate_sentence

        for section in (report.findings, report.impression):
            for i, s in enumerate(section):
                linked = [f for f in fs if f.id in s.finding_ids]
                try:
                    new_text = rephraser.rephrase(s, linked)
                except Exception:  # noqa: BLE001 - any LLM failure falls back to the template
                    continue
                candidate = s.model_copy(update={"text": new_text})
                if not validate_sentence(candidate, result):
                    section[i] = candidate

    from radio.report.validator import validate_report

    issues = validate_report(report, result)
    report.validation_issues = issues
    report.validation_passed = not any(i.severity == "error" for i in issues)
    return report
