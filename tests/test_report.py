import random

from radio.ontology.cxr import CONCEPTS
from radio.report.generator import generate_report
from radio.report.validator import validate_report, validate_sentence
from radio.schema.findings import (
    FindingStatus,
    Laterality,
    Measurement,
    Provenance,
    Recommendation,
    ReportSentence,
    SentenceKind,
)

from .conftest import make_finding, make_result


def _codes(text: str) -> set[str]:
    return {k for k, c in CONCEPTS.items() if any(t in text.lower() for t in c.terms)}


def test_normal_report_passes_and_discloses_not_assessed(normal_result):
    rep = generate_report(normal_result)
    assert rep.validation_passed, rep.validation_issues
    assert rep.impression[0].kind == SentenceKind.normal_summary
    text = rep.to_text()
    assert "Not assessed by the AI system: fracture" in text
    assert text.startswith("AI-GENERATED DRAFT")


def test_abnormal_report_orders_critical_first(abnormal_result):
    rep = generate_report(abnormal_result)
    assert rep.validation_passed, rep.validation_issues
    assert "pneumothorax" in rep.impression[0].text.lower()
    assert "critical" in rep.impression[0].text.lower()
    assert not any(s.kind == SentenceKind.normal_summary for s in rep.impression)
    assert "Possible pleural effusion" in rep.to_text()


def test_no_normal_summary_when_core_not_assessed():
    r = make_result({"pneumothorax": FindingStatus.absent, "pleural_effusion": FindingStatus.not_assessed})
    rep = generate_report(r)
    assert rep.validation_passed, rep.validation_issues
    assert all(s.kind != SentenceKind.normal_summary for s in rep.impression)


def test_property_report_never_mentions_unasserted_findings():
    """Random IRs: positive/uncertain sentences only mention concepts that are present/uncertain."""
    rng = random.Random(0)
    keys = list(CONCEPTS)
    for _ in range(300):
        chosen = rng.sample(keys, rng.randint(1, len(keys)))
        statuses = {k: rng.choice(list(FindingStatus)) for k in chosen}
        r = make_result(statuses)
        rep = generate_report(r)
        assert rep.validation_passed, (statuses, rep.validation_issues)
        asserted = {k for k, s in statuses.items() if s in (FindingStatus.present, FindingStatus.uncertain)}
        for s in rep.findings + rep.impression:
            if s.kind in (SentenceKind.positive, SentenceKind.uncertain):
                assert _codes(s.text) <= asserted, (s.text, asserted)
        for k, s in statuses.items():
            if s == FindingStatus.not_assessed:
                assert CONCEPTS[k].display in rep.to_text()


def test_validator_catches_hallucinated_mention(abnormal_result):
    pt = next(f for f in abnormal_result.findings if f.code == "pneumothorax")
    bad = ReportSentence(text="Pneumothorax and consolidation.", kind=SentenceKind.positive,
                         finding_ids=[pt.id], provenance=Provenance.model_inferred)
    codes = {i.code for i in validate_sentence(bad, abnormal_result)}
    assert "unsupported_mention" in codes


def test_validator_catches_wrong_side_and_invented_number(abnormal_result):
    pt = next(f for f in abnormal_result.findings if f.code == "pneumothorax")
    bad = ReportSentence(text="Left pneumothorax measuring 3 cm.", kind=SentenceKind.positive,
                         finding_ids=[pt.id], provenance=Provenance.model_inferred)
    codes = {i.code for i in validate_sentence(bad, abnormal_result)}
    assert {"laterality_unsupported", "unsupported_number"} <= codes


def test_validator_accepts_supported_side_and_measurement():
    f = make_finding("pleural_effusion", FindingStatus.present, laterality=Laterality.right,
                     measurements=[Measurement(name="depth", value=2.5, unit="cm", method="test",
                                               provenance=Provenance.radiologist_entered)])
    r = make_result({})
    r = r.model_copy(update={"findings": [f]})
    s = ReportSentence(text="Right pleural effusion, depth 2.5 cm.", kind=SentenceKind.positive,
                       finding_ids=[f.id], provenance=Provenance.model_inferred)
    assert validate_sentence(s, r) == []


def test_validator_requires_hedging_and_prior_for_comparison(abnormal_result):
    eff = next(f for f in abnormal_result.findings if f.code == "pleural_effusion")
    certain = ReportSentence(text="Pleural effusion.", kind=SentenceKind.uncertain, finding_ids=[eff.id],
                             provenance=Provenance.model_inferred)
    assert "uncertainty_not_preserved" in {i.code for i in validate_sentence(certain, abnormal_result)}
    pt = next(f for f in abnormal_result.findings if f.code == "pneumothorax")
    comp = ReportSentence(text="Pneumothorax, unchanged compared with prior.", kind=SentenceKind.positive,
                          finding_ids=[pt.id], provenance=Provenance.model_inferred)
    assert "unsupported_comparison" in {i.code for i in validate_sentence(comp, abnormal_result)}


def test_failed_draft_cannot_be_approved_as_is(abnormal_result):
    from radio.schema.findings import Review, ReviewAction

    rep = generate_report(abnormal_result).model_copy(update={"validation_passed": False})
    r = abnormal_result.model_copy(update={"report": rep})
    import pytest

    with pytest.raises(ValueError):
        r.apply_review(Review(reviewer_id="rad", action=ReviewAction.approve))
    r.apply_review(Review(reviewer_id="rad", action=ReviewAction.amend, final_report_text="Radiologist text."))


def test_validator_catches_negation_flip(abnormal_result):
    pt = next(f for f in abnormal_result.findings if f.code == "pneumothorax")
    bad = ReportSentence(text="No pneumothorax.", kind=SentenceKind.positive, finding_ids=[pt.id],
                         provenance=Provenance.model_inferred)
    assert "negation_in_positive" in {i.code for i in validate_sentence(bad, abnormal_result)}


def test_validator_rejects_unapproved_recommendation(abnormal_result):
    rep = generate_report(abnormal_result)
    pt = next(f for f in abnormal_result.findings if f.code == "pneumothorax")
    rep.recommendations.append(Recommendation(text="CT in 3 months.", guideline_id="made-up", finding_ids=[pt.id]))
    assert "unsupported_recommendation" in {i.code for i in validate_report(rep, abnormal_result)}


def test_omitted_and_critical_checks(abnormal_result):
    rep = generate_report(abnormal_result)
    rep.impression = []
    rep.findings = [s for s in rep.findings if s.kind != SentenceKind.positive]
    codes = {i.code for i in validate_report(rep, abnormal_result)}
    assert {"omitted_finding", "critical_not_in_impression"} <= codes


class _EvilRephraser:
    name = "evil"

    def rephrase(self, sentence, findings):
        return sentence.text.rstrip(".") + " with a 4 cm left mass."


class _PoliteRephraser:
    name = "polite"

    def rephrase(self, sentence, findings):
        return sentence.text.replace("identified by AI analysis", "seen on AI analysis")


def test_rephraser_cannot_add_content(abnormal_result):
    rep = generate_report(abnormal_result, rephraser=_EvilRephraser())
    assert rep.validation_passed
    assert "4 cm" not in rep.to_text()


def test_rephraser_can_change_wording(normal_result):
    rep = generate_report(normal_result, rephraser=_PoliteRephraser())
    assert rep.validation_passed
    assert "seen on AI analysis" in rep.to_text()
