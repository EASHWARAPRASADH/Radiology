"""WP 3.3: comparison against radiologist-verified prior findings."""

from __future__ import annotations

import pytest

from radio.report.comparison import PriorNotVerifiable, prior_from_result
from radio.report.generator import generate_report
from radio.report.validator import validate_report
from radio.schema.findings import (
    ClinicalContext,
    ComparisonChange,
    FindingOverride,
    FindingStatus,
    Laterality,
    PriorFinding,
    PriorStudy,
    Review,
    ReviewAction,
    SentenceKind,
)

from .conftest import make_finding, make_result

P, A, U = FindingStatus.present, FindingStatus.absent, FindingStatus.uncertain


def _prior(**statuses) -> PriorStudy:
    return PriorStudy(study_date="2026-03-01", findings_verified_by="dr_a",
                      findings=[PriorFinding(code=c, status=s) for c, s in statuses.items()])


def _current(prior: PriorStudy | None, **statuses):
    r = make_result({})
    findings = [make_finding(c, s) for c, s in statuses.items()]
    ctx = ClinicalContext(priors=[prior] if prior else [])
    return r.model_copy(update={"findings": findings, "clinical_context": ctx})


def test_new_persistent_resolved_and_no_change_statements():
    prior = _prior(pleural_effusion=A, cardiomegaly=P, pneumothorax=P, edema=A, nodule=P)
    r = _current(prior, pleural_effusion=P, cardiomegaly=P, pneumothorax=A, edema=A, nodule=U)
    rep = generate_report(r)
    assert rep.validation_passed, rep.validation_issues
    got = {(next(f.code for f in r.findings if f.id == s.finding_ids[0]), s.change) for s in rep.comparison}
    assert got == {("pleural_effusion", ComparisonChange.new), ("cardiomegaly", ComparisonChange.persistent),
                   ("pneumothorax", ComparisonChange.resolved)}  # edema absent->absent and uncertain nodule: silent
    text = rep.to_text()
    assert "COMPARISON:" in text and "new since the prior study of 2026-03-01" in text
    assert "radiologist-verified findings" in text


def test_no_comparison_without_verified_prior_or_on_side_mismatch():
    unverified = PriorStudy(study_date="2026-03-01", report_text="Small left effusion.")
    rep = generate_report(_current(unverified, pleural_effusion=P))
    assert rep.comparison == [] and "comparison was not performed" in rep.to_text()

    prior = PriorStudy(findings_verified_by="dr_a",
                       findings=[PriorFinding(code="pleural_effusion", status=P, laterality=Laterality.left)])
    r = _current(prior)
    r = r.model_copy(update={"findings": [make_finding("pleural_effusion", P, laterality=Laterality.right)]})
    rep = generate_report(r)
    assert rep.comparison == [] and rep.validation_passed


def test_validator_rejects_wrong_or_unsupported_comparison():
    prior = _prior(pleural_effusion=A)
    r = _current(prior, pleural_effusion=P)
    rep = generate_report(r)
    s = rep.comparison[0]

    def codes(report):
        return {i.code for i in validate_report(report, r)}

    flipped = rep.model_copy(deep=True)
    flipped.comparison[0] = s.model_copy(update={"change": ComparisonChange.persistent,
                                                 "text": s.text.replace("new since", "previously reported on")})
    assert "comparison_mismatch" in codes(flipped)
    text_only = rep.model_copy(deep=True)
    text_only.comparison[0] = s.model_copy(update={"text": s.text.replace("new since", "previously reported on")})
    assert "comparison_mismatch" in codes(text_only)
    dropped = rep.model_copy(deep=True)
    dropped.comparison = []
    assert "comparison_omitted" in codes(dropped)
    orphan = rep.model_copy(deep=True)
    orphan.comparison[0] = s.model_copy(update={"prior_id": "nope"})
    assert "comparison_without_verified_prior" in codes(orphan)
    # comparison wording is not allowed in ordinary finding sentences, even when a prior is linked
    stray = rep.model_copy(deep=True)
    stray.findings[0] = stray.findings[0].model_copy(update={"text": "Pleural effusion, unchanged."})
    assert "unsupported_comparison" in codes(stray)


def test_prior_from_result_requires_structured_radiologist_verification():
    earlier = make_result({"pleural_effusion": P, "pneumothorax": A, "nodule": U})
    with pytest.raises(PriorNotVerifiable, match="not radiologist-approved"):
        prior_from_result(earlier)
    approved = earlier.apply_review(Review(reviewer_id="dr_a", action=ReviewAction.approve))
    prior = prior_from_result(approved, "2026-03-01")
    assert {(f.code, f.status) for f in prior.findings} == {("pleural_effusion", P), ("pneumothorax", A)}
    assert prior.findings_verified_by == "dr_a" and prior.comparable

    eff = next(f for f in earlier.findings if f.code == "pleural_effusion")
    overridden = earlier.apply_review(Review(reviewer_id="dr_b", action=ReviewAction.amend,
                                             finding_overrides=[FindingOverride(finding_id=eff.id, status=A)]))
    assert ("pleural_effusion", A) in {(f.code, f.status) for f in prior_from_result(overridden).findings}

    free_text = earlier.apply_review(Review(reviewer_id="dr_c", action=ReviewAction.amend,
                                            final_report_text="Small right effusion. Possible nodule."))
    with pytest.raises(PriorNotVerifiable, match="free text"):
        prior_from_result(free_text)


def test_prior_schema_guards():
    with pytest.raises(ValueError, match="verified"):
        PriorStudy(findings=[PriorFinding(code="nodule", status=P)])
    with pytest.raises(ValueError, match="present or absent"):
        PriorStudy(findings_verified_by="x", findings=[PriorFinding(code="nodule", status=U)])


def test_comparison_sentences_are_not_rephrased():
    class Shouty:
        name = "shouty"

        def rephrase(self, sentence, findings):
            assert sentence.kind != SentenceKind.comparison
            return sentence.text

    generate_report(_current(_prior(pleural_effusion=A), pleural_effusion=P), rephraser=Shouty())
