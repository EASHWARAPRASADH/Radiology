"""WP 3.1: every concept x status x laterality x provenance x critical combination renders a draft that passes
the validator, names the finding, and carries the template library version."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from radio.ontology.cxr import CONCEPTS
from radio.report.generator import generate_report
from radio.report.templates import FINDING_PROVENANCES, TemplateError, TemplateLibrary, load_library
from radio.report.validator import _COMPARISON, _NEGATION, _SIDE
from radio.schema.findings import BoundingBox, FindingStatus, Laterality, Localization, LocalizationMethod, Provenance

from .conftest import make_finding, make_result

BOX = Localization(method=LocalizationMethod.detector, bbox=BoundingBox(x_min=1, y_min=1, x_max=9, y_max=9),
                   score=0.9, validated=True)


def _combos():
    for code, c in CONCEPTS.items():
        sides = ([Laterality.left, Laterality.right, Laterality.bilateral, Laterality.unknown]
                 if c.laterality_applicable else [Laterality.not_applicable])
        for status, side, prov, crit in itertools.product(
            list(FindingStatus), sides, FINDING_PROVENANCES, [False, True]
        ):
            yield code, status, side, prov, crit


def _finding(code, status, side, prov, crit):
    kw = dict(laterality=side, provenance=prov, critical=crit)
    if prov == Provenance.image_observed and status in (FindingStatus.present, FindingStatus.uncertain):
        kw["localizations"] = [BOX]
    if prov == Provenance.radiologist_entered:
        kw["evidence"] = None
    return make_finding(code, status, **kw)


def test_every_combination_renders_a_valid_draft():
    rendered = skipped = 0
    for code, status, side, prov, crit in _combos():
        try:
            f = _finding(code, status, side, prov, crit)
        except ValidationError:
            skipped += 1  # schema forbids the combination (e.g. critical + absent)
            continue
        r = make_result({})
        r = r.model_copy(update={"findings": [f]})
        rep = generate_report(r)
        assert rep.validation_passed, (code, status, side, prov, crit, rep.validation_issues)
        text = rep.to_text().lower()
        assert CONCEPTS[code].display in text
        if status in (FindingStatus.present, FindingStatus.uncertain) and side in (
            Laterality.left, Laterality.right, Laterality.bilateral
        ):
            assert f"{side.value} {CONCEPTS[code].display}" in text
        if status == FindingStatus.present:
            suffix = load_library().provenance_suffix[prov.value].strip()
            if suffix:
                assert suffix.lower() in text
        rendered += 1
    assert rendered == 954 and skipped > 0  # update if concepts or statuses change


def test_library_metadata_and_generator_tag():
    lib = load_library()
    assert lib.status == "engineering_draft" and lib.reviewed_by is None  # no clinical sign-off yet
    rep = generate_report(make_result({"pneumothorax": FindingStatus.absent}))
    assert rep.generator == lib.tag == "radio-cxr-templates@1.0.0"


def test_provenance_suffixes_cannot_trip_or_mask_validator_rules():
    for prov, suffix in load_library().provenance_suffix.items():
        for rx in (_NEGATION, _COMPARISON, _SIDE):
            assert not rx.search(suffix), (prov, rx.pattern)


def test_library_is_fail_closed():
    lib = load_library()
    raw = {"id": "x", "version": "0", "sentences": dict(lib.sentences),
           "provenance_suffix": dict(lib.provenance_suffix)}
    TemplateLibrary.from_dict(raw)
    bad = {**raw, "sentences": {k: v for k, v in lib.sentences.items() if k != "critical"}}
    with pytest.raises(TemplateError, match="critical"):
        TemplateLibrary.from_dict(bad)
    with pytest.raises(TemplateError, match="model_inferred"):
        TemplateLibrary.from_dict({**raw, "provenance_suffix": {"image_observed": "", "radiologist_entered": ""}})
    with pytest.raises(TemplateError, match="unknown concepts"):
        TemplateLibrary.from_dict({**raw, "concept_overrides": {"unicorn": {}}})
    f = make_finding("pleural_effusion", FindingStatus.present, provenance=Provenance.prior_report, evidence=None)
    with pytest.raises(TemplateError, match="prior_report"):
        generate_report(make_result({}).model_copy(update={"findings": [f]}))


def test_concept_override_is_used():
    lib = load_library()
    raw = {"id": "x", "version": "1", "sentences": dict(lib.sentences),
           "provenance_suffix": dict(lib.provenance_suffix),
           "concept_overrides": {"cardiomegaly": {"present_finding": "Enlarged cardiac silhouette{suffix}."}}}
    custom = TemplateLibrary.from_dict(raw)
    rep = generate_report(make_result({"cardiomegaly": FindingStatus.present}), library=custom)
    assert "Enlarged cardiac silhouette (AI-inferred" in rep.to_text() and rep.validation_passed
    assert rep.generator == "x@1"


def test_perturbation_suite_detects_every_injected_error_class():
    import random

    from radio.report.perturb import random_result, run_suite

    rng = random.Random(3)
    out = run_suite([random_result(rng) for _ in range(150)], seed=3)
    assert out["unperturbed_failures"] == 0
    for name, v in out["probes"].items():
        assert v["applied"] > 0, name
        if name == "synonym_mention":
            continue  # known limit of term matching; guarded on the rephraser path instead (below)
        assert v["detected_expected_code"] == 1.0, (name, v["missed_examples"])
    assert out["rephraser_synonym_injection"]["synonyms_in_final_draft"] == 0


def test_side_is_checked_per_finding_in_grouped_sentences():
    from radio.report.validator import validate_sentence
    from radio.schema.findings import ReportSentence, SentenceKind

    a = make_finding("lung_opacity", FindingStatus.uncertain, laterality=Laterality.left)
    b = make_finding("nodule", FindingStatus.uncertain, laterality=Laterality.bilateral)
    r = make_result({}).model_copy(update={"findings": [a, b]})

    def check(text):
        s = ReportSentence(text=text, kind=SentenceKind.uncertain, finding_ids=[a.id, b.id],
                           provenance=Provenance.model_inferred)
        return {i.code for i in validate_sentence(s, r)}

    assert not check("Possible left lung opacity and bilateral nodule; indeterminate.")
    # Both side words are individually allowed in this sentence, but bound to the wrong findings.
    assert "laterality_unsupported" in check("Possible bilateral lung opacity and left nodule; indeterminate.")


def test_rephraser_vocabulary_is_closed():
    from radio.report.validator import novel_words

    assert not novel_words("No pneumothorax identified by AI analysis.", "No pneumothorax is seen on AI analysis.")
    assert novel_words("Pleural effusion.", "Pleural effusion with volume loss.") == {"volume", "loss"}
