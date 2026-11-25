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
from radio.report.comparison import comparable_prior, compare
from radio.report.templates import TemplateLibrary, load_library
from radio.schema.findings import (
    AI_DISCLAIMER,
    AnalysisResult,
    DraftReport,
    Finding,
    FindingStatus,
    Provenance,
    QualityStatus,
    ReportSentence,
    SentenceKind,
)

GENERATOR_ID = load_library().tag


class Rephraser(Protocol):
    name: str

    def rephrase(self, sentence: ReportSentence, findings: list[Finding]) -> str: ...


def _join(items: list[str], word: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def _uncertain_group(fs: list[Finding], lib: TemplateLibrary) -> ReportSentence:
    """One sentence for several indeterminate findings (keeps every finding id for traceability)."""
    if len(fs) == 1:
        f = fs[0]
        text = lib.render("uncertain_one", side=lib.side(f), display=CONCEPTS[f.code].display)
        return ReportSentence(text=text, kind=SentenceKind.uncertain, finding_ids=[f.id], provenance=f.provenance)
    text = lib.render("uncertain_many", list_and=_join([lib.named(f) for f in fs], "and"))
    prov = fs[0].provenance if len({f.provenance for f in fs}) == 1 else Provenance.model_inferred
    return ReportSentence(text=text, kind=SentenceKind.uncertain, finding_ids=[f.id for f in fs], provenance=prov)


def _by_priority(fs: Iterable[Finding]) -> list[Finding]:
    return sorted(fs, key=lambda f: CONCEPTS[f.code].priority)


def generate_report(result: AnalysisResult, rephraser: Rephraser | None = None,
                    library: TemplateLibrary | None = None) -> DraftReport:
    """Build a DraftReport from ``result`` (which must not be an abstention) using ``library`` templates."""
    if result.abstention is not None:
        raise ValueError("cannot generate a report for an abstained analysis")
    lib = library or load_library()

    fs = result.findings
    present = _by_priority(f for f in fs if f.status == FindingStatus.present)
    uncertain = _by_priority(f for f in fs if f.status == FindingStatus.uncertain)
    absent = _by_priority(f for f in fs if f.status == FindingStatus.absent)
    not_assessed = _by_priority(f for f in fs if f.status == FindingStatus.not_assessed)

    st = result.study
    view = (lib.render("view_known", view=st.view_position) if st.view_position
            else lib.sentences["view_unknown"])
    study_section = [ReportSentence(text=lib.render("study", modality=st.modality, view=view),
                                    kind=SentenceKind.context, provenance=Provenance.study_metadata)]

    ctx = result.clinical_context
    clinical: list[ReportSentence] = []
    if ctx.indication:
        clinical.append(ReportSentence(text=lib.render("indication", text=ctx.indication.strip().rstrip(".")),
                                       kind=SentenceKind.context, provenance=Provenance.clinical_history))
    if ctx.history:
        clinical.append(ReportSentence(text=lib.render("history", text=ctx.history.strip().rstrip(".")),
                                       kind=SentenceKind.context, provenance=Provenance.clinical_history))
    prior = comparable_prior(result)
    if prior is not None:
        date = f" of {prior.study_date}" if prior.study_date else ""
        clinical.append(ReportSentence(text=lib.render("priors_compared", date=date), kind=SentenceKind.context,
                                       provenance=Provenance.prior_report))
    else:
        clinical.append(ReportSentence(
            text=lib.render("priors_not_compared" if ctx.priors else "no_priors"), kind=SentenceKind.context,
            provenance=Provenance.prior_report if ctx.priors else Provenance.study_metadata,
        ))
    comparison: list[ReportSentence] = []
    if prior is not None:
        for f, change in sorted(compare(result, prior), key=lambda fc: CONCEPTS[fc[0].code].priority):
            comparison.append(ReportSentence(
                text=lib.render(f"comparison_{change.value}", side=lib.side(f), display=CONCEPTS[f.code].display,
                                date=date),
                kind=SentenceKind.comparison, finding_ids=[f.id], provenance=Provenance.prior_report,
                prior_id=prior.id, change=change))

    q = result.image_quality
    technique: list[ReportSentence] = []
    if q.status in (QualityStatus.limited, QualityStatus.non_diagnostic):
        readable = ", ".join(i.split(":")[0].replace("_", " ") for i in q.issues) or "unspecified"
        technique.append(ReportSentence(text=lib.render("quality_limited", status=q.status.value, issues=readable),
                                        kind=SentenceKind.quality, provenance=Provenance.model_inferred))

    findings: list[ReportSentence] = []
    for f in present:
        findings.append(ReportSentence(text=lib.present(f), kind=SentenceKind.positive, finding_ids=[f.id],
                                       provenance=f.provenance))
    if uncertain:
        findings.append(_uncertain_group(uncertain, lib))
    if absent:
        findings.append(ReportSentence(
            text=lib.render("absent_group", list_or=_join([CONCEPTS[f.code].display for f in absent], "or")),
            kind=SentenceKind.negative, finding_ids=[f.id for f in absent], provenance=Provenance.model_inferred,
        ))
    if not_assessed:
        findings.append(ReportSentence(
            text=lib.render("not_assessed_group", list_comma=", ".join(CONCEPTS[f.code].display for f in not_assessed)),
            kind=SentenceKind.not_assessed, finding_ids=[f.id for f in not_assessed],
            provenance=Provenance.model_inferred,
        ))

    impression: list[ReportSentence] = []
    critical = [f for f in present + uncertain if f.critical]
    for f in critical:
        impression.append(ReportSentence(
            text=lib.render("critical", side=lib.side(f), display=CONCEPTS[f.code].display),
            kind=SentenceKind.positive if f.status == FindingStatus.present else SentenceKind.uncertain,
            finding_ids=[f.id], provenance=f.provenance,
        ))
    for f in present:
        if not f.critical:
            impression.append(ReportSentence(text=lib.present(f), kind=SentenceKind.positive,
                                             finding_ids=[f.id], provenance=f.provenance))
    non_critical_uncertain = [f for f in uncertain if not f.critical]
    if non_critical_uncertain:
        impression.append(_uncertain_group(non_critical_uncertain, lib))
    if not present and not uncertain:
        by_code = {f.code: f for f in fs}
        core_absent = all(c in by_code and by_code[c].status == FindingStatus.absent for c in CORE_ACUTE_CONCEPTS)
        if core_absent:
            impression.append(ReportSentence(
                text=lib.render("normal_summary"),
                kind=SentenceKind.normal_summary, finding_ids=[by_code[c].id for c in CORE_ACUTE_CONCEPTS],
                provenance=Provenance.model_inferred,
            ))
        elif absent:
            impression.append(ReportSentence(
                text=lib.render("partial_negative_summary"),
                kind=SentenceKind.negative, finding_ids=[f.id for f in absent], provenance=Provenance.model_inferred,
            ))

    report = DraftReport(
        disclaimer=AI_DISCLAIMER,
        study_section=study_section,
        clinical_section=clinical,
        technique_quality=technique,
        findings=findings,
        comparison=comparison,
        impression=impression,
        recommendations=[],  # only from APPROVED_GUIDELINES; none approved in the prototype
        generator=lib.tag if rephraser is None else f"{lib.tag}+{rephraser.name}",
    )

    if rephraser is not None:
        from radio.report.validator import novel_words, validate_sentence

        for section in (report.findings, report.impression):
            for i, s in enumerate(section):
                linked = [f for f in fs if f.id in s.finding_ids]
                try:
                    new_text = rephraser.rephrase(s, linked)
                except Exception:  # noqa: BLE001 - any LLM failure falls back to the template
                    continue
                candidate = s.model_copy(update={"text": new_text})
                # Closed vocabulary: a rephrase may reorder template words and add neutral style words only.
                # The term-matching validator cannot catch every synonym ("pleural fluid", "volume loss").
                if not novel_words(s.text, new_text) and not validate_sentence(candidate, result):
                    section[i] = candidate

    from radio.report.validator import validate_report

    issues = validate_report(report, result)
    report.validation_issues = issues
    report.validation_passed = not any(i.severity == "error" for i in issues)
    return report
