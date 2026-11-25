"""Synthetic perturbation suite for the report validator (Phase 3, WP 3.2).

Each perturbation injects one known class of report error into a valid template draft (a wrong side, a
flipped negation, an invented finding, a dropped critical finding...). The suite measures, per class, how
often the validator raises an error, and how often it raises the *expected* issue code. Any class below 100%
is a validator gap; the ``synonym_*`` classes probe free-text mentions that a constrained LLM rephraser or a
radiologist edit could plausibly produce.

The suite measures the validator only. It says nothing about whether the IR itself is right.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from radio.ontology.cxr import CONCEPTS
from radio.schema.findings import (
    AnalysisResult,
    BoundingBox,
    ClinicalContext,
    ComparisonChange,
    CriticalStatus,
    DraftReport,
    Finding,
    FindingStatus,
    ImageQuality,
    Laterality,
    Localization,
    LocalizationMethod,
    Measurement,
    ModelEvidence,
    PriorFinding,
    PriorStudy,
    Provenance,
    QualityStatus,
    Recommendation,
    ReportSentence,
    SentenceKind,
    StudyInfo,
    UncertaintyStatus,
)

_P = {FindingStatus.present: 0.9, FindingStatus.uncertain: 0.5, FindingStatus.absent: 0.02}


def random_result(rng: random.Random, p_status: tuple[float, float, float, float] = (0.15, 0.15, 0.4, 0.3),
                  p_prior: float = 0.5) -> AnalysisResult:
    """A random but schema-valid IR over all concepts (statuses, sides, provenance, critical, measurements),
    with a radiologist-verified prior attached with probability ``p_prior``."""
    findings = []
    statuses = list(FindingStatus)  # present, absent, uncertain, not_assessed (enum order)
    weights = {FindingStatus.present: p_status[0], FindingStatus.uncertain: p_status[1],
               FindingStatus.absent: p_status[2], FindingStatus.not_assessed: p_status[3]}
    for code, c in CONCEPTS.items():
        st = rng.choices(statuses, weights=[weights[s] for s in statuses])[0]
        kw: dict = {}
        if st == FindingStatus.not_assessed:
            kw["not_assessed_reason"] = "insufficient_validation_data"
            prov = Provenance.model_inferred
        else:
            kw["evidence"] = ModelEvidence(model_id="synthetic", model_version="0", raw_probability=_P[st],
                                           calibrated_probability=_P[st], threshold_absent=0.1, threshold_present=0.7)
            positive = st in (FindingStatus.present, FindingStatus.uncertain)
            prov = Provenance.image_observed if positive and rng.random() < 0.4 else Provenance.model_inferred
            if prov == Provenance.image_observed:
                kw["localizations"] = [Localization(method=LocalizationMethod.detector, score=0.9, validated=True,
                                                    bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=60))]
            if positive and c.laterality_applicable:
                kw["laterality"] = rng.choice([Laterality.left, Laterality.right, Laterality.bilateral,
                                               Laterality.unknown])
            if st == FindingStatus.present and rng.random() < 0.15:
                kw["measurements"] = [Measurement(name="size", value=float(rng.randint(5, 60)), unit="mm",
                                                  method="synthetic", provenance=Provenance.model_inferred)]
            if positive and c.critical_candidate and rng.random() < 0.7:
                kw["critical"] = True
            if st == FindingStatus.uncertain:
                kw["uncertainty_reasons"] = ["between_thresholds"]
        findings.append(Finding(code=code, display=c.display, status=st, provenance=prov, **kw))
    crit = CriticalStatus.suspected if any(f.critical for f in findings) else CriticalStatus.none_detected
    priors = []
    if rng.random() < p_prior:
        pf = [PriorFinding(code=f.code, status=rng.choice([FindingStatus.present, FindingStatus.absent]),
                           laterality=(f.laterality if rng.random() < 0.8 else Laterality.unknown))
              for f in findings if rng.random() < 0.8]
        priors.append(PriorStudy(study_date="2026-03-01", findings=pf, findings_verified_by="dr_synthetic"))
    return AnalysisResult(
        study=StudyInfo(modality="DX", anatomical_region="chest", view_position="PA", source_format="dicom"),
        clinical_context=ClinicalContext(priors=priors), image_quality=ImageQuality(status=QualityStatus.adequate),
        findings=findings, critical_status=crit, uncertainty_status=UncertaintyStatus.low)


# --- perturbations ---------------------------------------------------------------------------------------
# Each takes (report copy, result, rng) and returns the mutated report, or None if it does not apply.
Perturb = Callable[[DraftReport, AnalysisResult, random.Random], DraftReport | None]

_SWAP = {"left": "right", "right": "left", "bilateral": "left"}
_HEDGES = re.compile(r"\b(possible|possibly|may|might|indeterminate|suspected|probable)\b ?", re.I)


def _pick(rng, sents, pred):
    idx = [i for i, s in enumerate(sents) if pred(s)]
    return rng.choice(idx) if idx else None


def _cited_codes(s: ReportSentence, r: AnalysisResult) -> set[str]:
    by_id = {f.id: f.code for f in r.findings}
    return {by_id[i] for i in s.finding_ids if i in by_id}


def _edit(rep: DraftReport, section: str, i: int, text: str) -> DraftReport:
    getattr(rep, section)[i] = getattr(rep, section)[i].model_copy(update={"text": text})
    return rep


def laterality_swap(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind in (SentenceKind.positive, SentenceKind.uncertain)
              and re.search(r"\b(left|right|bilateral)\b", s.text))
    if i is None:
        return None
    t = re.sub(r"\b(left|right|bilateral)\b", lambda m: _SWAP[m.group(1).lower()], rep.findings[i].text, count=1)
    return _edit(rep, "findings", i, t)


def laterality_insert(rep, r, rng):
    by_id = {f.id: f for f in r.findings}
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.positive and len(s.finding_ids) == 1
              and by_id[s.finding_ids[0]].laterality == Laterality.unknown
              and CONCEPTS[by_id[s.finding_ids[0]].code].laterality_applicable)
    if i is None:
        return None
    display = CONCEPTS[by_id[rep.findings[i].finding_ids[0]].code].display
    t = re.sub(re.escape(display), f"left {display}", rep.findings[i].text, count=1, flags=re.I)
    return _edit(rep, "findings", i, t)


def negate_positive(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.positive)
    return None if i is None else _edit(rep, "findings", i, "No " + rep.findings[i].text[:1].lower()
                                        + rep.findings[i].text[1:])


def drop_negation(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.negative)
    if i is None:
        return None
    t = re.sub(r"^No (.*) identified by AI analysis\.$", r"\1 identified by AI analysis.", rep.findings[i].text)
    return _edit(rep, "findings", i, t[:1].upper() + t[1:])


def drop_hedge(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.uncertain)
    if i is None:
        return None
    t = _HEDGES.sub("", rep.findings[i].text.split(";")[0]) + "."
    return _edit(rep, "findings", i, t[:1].upper() + t[1:])


def _uncited_concept(s: ReportSentence, r: AnalysisResult, rng) -> str | None:
    cited = _cited_codes(s, r)
    pool = [c for c in CONCEPTS if c not in cited and CONCEPTS[c].terms]
    return rng.choice(pool) if pool else None


def insert_unsupported_finding(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind in (SentenceKind.positive, SentenceKind.uncertain,
                                                       SentenceKind.negative))
    if i is None:
        return None
    c = _uncited_concept(rep.findings[i], r, rng)
    return None if c is None else _edit(rep, "findings", i,
                                        rep.findings[i].text + f" There is also {CONCEPTS[c].terms[0]}.")


def insert_comparison(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.positive)
    return None if i is None else _edit(rep, "findings", i, rep.findings[i].text.rstrip(".") +
                                        ", unchanged from the prior study.")


def insert_number(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.positive)
    return None if i is None else _edit(rep, "findings", i, rep.findings[i].text.rstrip(".") + ", measuring 997 mm.")


def drop_present_finding(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.positive)
    if i is None:
        return None
    del rep.findings[i]
    return rep


def drop_critical_from_impression(rep, r, rng):
    crit = {f.id for f in r.findings if f.critical}
    keep = [s for s in rep.impression if not (set(s.finding_ids) & crit)]
    if len(keep) == len(rep.impression):
        return None
    rep.impression = keep
    return rep


def drop_not_assessed(rep, r, rng):
    keep = [s for s in rep.findings if s.kind != SentenceKind.not_assessed]
    if len(keep) == len(rep.findings):
        return None
    rep.findings = keep
    return rep


def unapproved_recommendation(rep, r, rng):
    pos = [f for f in r.findings if f.status in (FindingStatus.present, FindingStatus.uncertain)]
    if not pos:
        return None
    rep.recommendations.append(Recommendation(text="Follow-up CT is recommended.", guideline_id="invented-v1",
                                              finding_ids=[rng.choice(pos).id]))
    return rep


def tamper_disclaimer(rep, r, rng):
    rep.disclaimer = "Report generated automatically."
    return rep


def false_normal(rep, r, rng):
    pos = [f for f in r.findings if f.status in (FindingStatus.present, FindingStatus.uncertain)]
    if not pos:
        return None
    rep.impression.append(ReportSentence(
        text="No acute cardiopulmonary abnormality identified by AI analysis among the assessed findings.",
        kind=SentenceKind.normal_summary, finding_ids=[pos[0].id], provenance=Provenance.model_inferred))
    return rep


def ungrounded_sentence(rep, r, rng):
    rep.findings.append(ReportSentence(text="The lungs appear clear.", kind=SentenceKind.positive,
                                       provenance=Provenance.model_inferred))
    return rep


def kind_swap(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind == SentenceKind.positive)
    if i is None:
        return None
    rep.findings[i] = rep.findings[i].model_copy(update={"kind": SentenceKind.negative})
    return rep


_MARK = {ComparisonChange.new: "new since", ComparisonChange.persistent: "previously reported on",
         ComparisonChange.resolved: "not identified by AI analysis on"}


def comparison_change_flip(rep, r, rng):
    """Claim a different change (field and wording together) than the IR and prior support."""
    if not rep.comparison:
        return None
    i = rng.randrange(len(rep.comparison))
    s = rep.comparison[i]
    new = rng.choice([c for c in ComparisonChange if c != s.change])
    text = s.text.replace(_MARK[s.change], _MARK[new]) if _MARK[s.change] in s.text else s.text
    rep.comparison[i] = s.model_copy(update={"change": new, "text": text})
    return rep


def comparison_text_flip(rep, r, rng):
    """Keep the structured change but word it as a different one."""
    i = _pick(rng, rep.comparison, lambda s: _MARK[s.change] in s.text)
    if i is None:
        return None
    s = rep.comparison[i]
    new = rng.choice([c for c in ComparisonChange if c != s.change])
    return _edit(rep, "comparison", i, s.text.replace(_MARK[s.change], _MARK[new]))


def comparison_drop(rep, r, rng):
    if not rep.comparison:
        return None
    del rep.comparison[rng.randrange(len(rep.comparison))]
    return rep


# Free-text synonyms a rephraser or a human edit could use (not the validator's canonical terms).
SYNONYM_PROBES: dict[str, tuple[str, ...]] = {
    "pneumothorax": ("air in the pleural space", "collapsed lung"),
    "pleural_effusion": ("pleural fluid", "fluid in the pleural space", "blunting of the costophrenic angle"),
    "cardiomegaly": ("enlarged heart", "the heart is enlarged", "increased cardiothoracic ratio"),
    "edema": ("pulmonary venous congestion", "interstitial fluid"),
    "consolidation": ("airspace disease", "airspace opacification"),
    "atelectasis": ("volume loss", "lobar collapse"),
    "nodule": ("small rounded opacity",),
    "mass": ("large rounded lesion",),
    "fracture": ("cortical break", "broken rib"),
    "emphysema": ("hyperinflated lungs", "bullae"),
}


def synonym_mention(rep, r, rng):
    i = _pick(rng, rep.findings, lambda s: s.kind in (SentenceKind.positive, SentenceKind.uncertain,
                                                       SentenceKind.negative))
    if i is None:
        return None
    cited = _cited_codes(rep.findings[i], r)
    pool = [(c, t) for c, ts in SYNONYM_PROBES.items() if c not in cited for t in ts]
    if not pool:
        return None
    _, term = rng.choice(pool)
    return _edit(rep, "findings", i, rep.findings[i].text + f" There is also {term}.")


@dataclass
class Probe:
    name: str
    fn: Perturb
    expected: tuple[str, ...]  # any of these issue codes counts as the expected detection


PROBES: list[Probe] = [
    Probe("laterality_swap", laterality_swap, ("laterality_unsupported",)),
    Probe("laterality_insert", laterality_insert, ("laterality_unsupported",)),
    Probe("negate_positive", negate_positive, ("negation_in_positive",)),
    Probe("drop_negation", drop_negation, ("missing_negation",)),
    Probe("drop_hedge", drop_hedge, ("uncertainty_not_preserved",)),
    Probe("insert_unsupported_finding", insert_unsupported_finding, ("unsupported_mention",)),
    Probe("insert_comparison", insert_comparison, ("unsupported_comparison",)),
    Probe("insert_number", insert_number, ("unsupported_number",)),
    Probe("drop_present_finding", drop_present_finding, ("omitted_finding",)),
    Probe("drop_critical_from_impression", drop_critical_from_impression, ("critical_not_in_impression",)),
    Probe("drop_not_assessed", drop_not_assessed, ("not_assessed_undisclosed",)),
    Probe("unapproved_recommendation", unapproved_recommendation, ("unsupported_recommendation",)),
    Probe("tamper_disclaimer", tamper_disclaimer, ("missing_disclaimer",)),
    Probe("false_normal", false_normal, ("normal_with_positive", "kind_status_mismatch")),
    Probe("ungrounded_sentence", ungrounded_sentence, ("ungrounded_sentence",)),
    Probe("kind_swap", kind_swap, ("kind_status_mismatch",)),
    Probe("comparison_change_flip", comparison_change_flip, ("comparison_mismatch",)),
    Probe("comparison_text_flip", comparison_text_flip, ("comparison_mismatch",)),
    Probe("comparison_drop", comparison_drop, ("comparison_omitted",)),
    Probe("synonym_mention", synonym_mention, ("unsupported_mention",)),
]


@dataclass
class ProbeStats:
    applied: int = 0
    detected: int = 0
    expected: int = 0
    missed_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        def rate(k: int) -> float:
            return k / self.applied if self.applied else float("nan")

        return {"applied": self.applied, "detected_any_error": rate(self.detected),
                "detected_expected_code": rate(self.expected), "missed_examples": self.missed_examples[:5]}


def run_suite(results: list[AnalysisResult], seed: int = 0, probes: list[Probe] | None = None) -> dict:
    """Apply every probe to every result's template draft. Also checks the unperturbed drafts pass."""
    from radio.report.generator import generate_report
    from radio.report.validator import validate_report

    rng = random.Random(seed)
    stats = {p.name: ProbeStats() for p in probes or PROBES}
    clean_fail = 0
    for r in results:
        base = generate_report(r)
        if not base.validation_passed:
            clean_fail += 1
            continue
        for p in probes or PROBES:
            mutated = p.fn(base.model_copy(deep=True), r, rng)
            if mutated is None:
                continue
            st = stats[p.name]
            st.applied += 1
            codes = {i.code for i in validate_report(mutated, r) if i.severity == "error"}
            st.detected += bool(codes)
            st.expected += bool(codes & set(p.expected))
            if not codes and len(st.missed_examples) < 5:
                changed = [s.text for s in mutated.findings if s.text not in {b.text for b in base.findings}]
                st.missed_examples.append(changed[0] if changed else p.name)
    return {"n_results": len(results), "unperturbed_failures": clean_fail,
            "probes": {k: v.as_dict() for k, v in stats.items()},
            "rephraser_synonym_injection": rephraser_probe(results, seed)}


class _SynonymInjector:
    """A hostile rephraser that appends a finding synonym to every sentence it is offered."""

    name = "synonym-injector"

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.terms = [t for ts in SYNONYM_PROBES.values() for t in ts]

    def rephrase(self, sentence: ReportSentence, findings: list[Finding]) -> str:
        return sentence.text.rstrip(".") + f"; there is also {self.rng.choice(self.terms)}."


def rephraser_probe(results: list[AnalysisResult], seed: int = 0) -> dict:
    """End-to-end: how many injected synonyms survive the generator's rephraser guards into the draft."""
    from radio.report.generator import generate_report

    inj = _SynonymInjector(random.Random(seed))
    offered = leaked = 0
    for r in results:
        rep = generate_report(r, rephraser=inj)
        for s in rep.findings + rep.impression:
            offered += 1
            leaked += any(t in s.text.lower() for t in inj.terms)
    return {"sentences_offered": offered, "synonyms_in_final_draft": leaked,
            "leak_rate": leaked / offered if offered else float("nan")}
