"""WP 3.4: report-level scoring against radiologist image labels."""

from __future__ import annotations

import numpy as np

from radio.eval.report_eval import evaluate_reports
from radio.report.generator import generate_report
from radio.schema.findings import (
    Abstention,
    BoundingBox,
    FindingStatus,
    Localization,
    LocalizationMethod,
    Provenance,
)

from .conftest import make_finding, make_result

P, A, U, N = FindingStatus.present, FindingStatus.absent, FindingStatus.uncertain, FindingStatus.not_assessed
CONCEPTS = ["pleural_effusion", "pneumothorax", "nodule"]
CORE = {"pneumothorax": A, "pleural_effusion": A, "consolidation": A, "edema": A, "lung_opacity": A}


def _with_report(r):
    return r.model_copy(update={"report": generate_report(r)})


def test_counts_false_absent_false_normal_and_box_hits():
    box = Localization(method=LocalizationMethod.detector, validated=True, score=0.9,
                       bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50))
    eff = make_finding("pleural_effusion", P, provenance=Provenance.image_observed, localizations=[box])
    r1 = make_result({"pneumothorax": A, "nodule": U, "edema": P})  # edema: outside the scored label space
    r1 = _with_report(r1.model_copy(update={"findings": [eff, *r1.findings]}))
    # r2: everything core absent -> normal summary, but the radiologist marked a nodule (false normal) and the
    # report calls pneumothorax absent while the radiologist marked it (false absent).
    r2 = _with_report(make_result({**CORE, "nodule": N}))
    r3 = make_result({"nodule": N}).model_copy(update={
        "findings": [make_finding("nodule", N)],
        "abstention": Abstention(reason="out_of_distribution", detail="test")})
    items = [
        ("a", r1, {"pleural_effusion"}, {"pleural_effusion": np.array([[12, 12, 48, 48]], np.float32)}),
        ("b", r2, {"pneumothorax", "nodule"}, {}),
        ("c", r3, {"nodule"}, {}),
    ]
    res = evaluate_reports(items, CONCEPTS)
    s = res["summary"]
    assert s["images"] == 3 and s["abstained"] == 1 and s["abstention_reasons"] == {"out_of_distribution": 1}
    assert s["false_normal_reports"] == 1 and s["reports_with_false_absent"] == 1
    assert s["validation_pass_rate"] == 1.0
    pc = res["per_concept"]
    assert pc["pneumothorax"]["false_absent_rate"] == 1.0
    assert pc["pleural_effusion"]["sensitivity_flagged"] == 1.0 and pc["pleural_effusion"]["ppv_present"] == 1.0
    assert pc["pleural_effusion"]["localised_claims"] == 1
    assert pc["pleural_effusion"]["localised_claim_hit_rate"] == 1.0
    assert pc["nodule"]["ref_positive"]["not_assessed"] == 1.0  # r2 nodule not assessed; r3 abstained (excluded)
    # Effusion is a true positive and the uncertain nodule in r1 a false positive; edema (present, but outside
    # the scored label space) is ignored rather than counted as a third assertion.
    assert s["finding_precision"] == 0.5
