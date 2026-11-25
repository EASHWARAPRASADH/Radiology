"""Report-level evaluation against radiologist image labels (Phase 3, WP 3.4).

Runs the full pipeline (QC, OOD, classifier, detector fusion, templates, validator) on images with radiologist
annotations and scores what the *draft report* asserts, per concept and per report. The reference is the
ChestX-Det box annotation: a concept is present in an image if a radiologist drew a box for it, otherwise
absent. Only the 12 ChestX-Det concepts are scored; anything else the report says is outside the reference.

Safety-relevant outputs:
* ``false_absent``: the report says "no X" but a radiologist marked X (the miss that matters most).
* ``false_normal``: the impression uses the normal summary while the image has any reference finding.
* ``localised_claims``: of ``image_observed`` claims (validated boxes), how many boxes hit a reference box.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

from radio.eval.detection import iou_matrix
from radio.eval.report_metrics import ReportScores, score_report
from radio.schema.findings import AnalysisResult, FindingStatus, Provenance, SentenceKind

STATUSES = [s.value for s in FindingStatus]


def _box_hits(result: AnalysisResult, code: str, ref_boxes: np.ndarray, hit_iou: float) -> bool | None:
    f = next((x for x in result.findings if x.code == code), None)
    if f is None or f.provenance != Provenance.image_observed:
        return None
    boxes = [loc.bbox for loc in f.localizations if loc.validated and loc.bbox is not None]
    if not boxes:
        return None
    if len(ref_boxes) == 0:
        return False
    pred = np.array([[b.x_min, b.y_min, b.x_max, b.y_max] for b in boxes], dtype=np.float32)
    return bool((iou_matrix(pred, ref_boxes) >= hit_iou).any())


def evaluate_reports(items: Iterable[tuple[str, AnalysisResult, set[str], dict[str, np.ndarray]]],
                     concepts: list[str], hit_iou: float = 0.25) -> dict:
    """``items`` yields (image id, pipeline result, reference-present concepts, reference boxes per concept)."""
    per = {c: {"ref_pos": Counter(), "ref_neg": Counter()} for c in concepts}
    loc = {c: Counter() for c in concepts}
    scores = ReportScores()
    n = abstained = passed = normal = false_normal = 0
    abst_reasons: Counter = Counter()
    asserted, uncertain_load, rows = [], [], []
    for image_id, r, ref_present, ref_boxes in items:
        n += 1
        if r.abstention is not None:
            abstained += 1
            abst_reasons[r.abstention.reason] += 1
            rows.append({"image": image_id, "abstained": r.abstention.reason})
            continue
        by_code = {f.code: f for f in r.findings}
        row = {"image": image_id, "reference": sorted(ref_present), "statuses": {}}
        for c in concepts:
            st = by_code[c].status.value if c in by_code else FindingStatus.not_assessed.value
            per[c]["ref_pos" if c in ref_present else "ref_neg"][st] += 1
            row["statuses"][c] = st
            hit = _box_hits(r, c, ref_boxes.get(c, np.zeros((0, 4), np.float32)), hit_iou)
            if hit is not None:
                loc[c]["claims"] += 1
                loc[c]["hits"] += int(hit)
        reference = {c: (FindingStatus.present if c in ref_present else FindingStatus.absent) for c in concepts}
        score_report(r, reference, scores=scores, label_space=set(concepts))
        rep = r.report
        passed += int(rep is not None and rep.validation_passed)
        is_normal = rep is not None and any(s.kind == SentenceKind.normal_summary for s in rep.impression)
        normal += int(is_normal)
        false_normal += int(is_normal and bool(ref_present & set(concepts)))
        asserted.append(sum(f.status in (FindingStatus.present, FindingStatus.uncertain) for f in r.findings))
        uncertain_load.append(sum(f.status == FindingStatus.uncertain for f in r.findings))
        rows.append(row)

    def frac(cnt: Counter) -> dict[str, float]:
        tot = sum(cnt.values())
        return {s: (cnt[s] / tot if tot else float("nan")) for s in STATUSES} | {"n": tot}

    per_concept = {}
    for c in concepts:
        pos, neg = per[c]["ref_pos"], per[c]["ref_neg"]
        called_pos = pos["present"] + pos["uncertain"]
        per_concept[c] = {
            "ref_positive": frac(pos), "ref_negative": frac(neg),
            "sensitivity_flagged": called_pos / sum(pos.values()) if sum(pos.values()) else float("nan"),
            "false_absent_rate": pos["absent"] / sum(pos.values()) if sum(pos.values()) else float("nan"),
            "ppv_present": (pos["present"] / (pos["present"] + neg["present"])
                            if pos["present"] + neg["present"] else float("nan")),
            "localised_claims": loc[c]["claims"],
            "localised_claim_hit_rate": loc[c]["hits"] / loc[c]["claims"] if loc[c]["claims"] else float("nan"),
        }
    scored = n - abstained
    summary = {
        "images": n, "abstained": abstained, "abstention_reasons": dict(abst_reasons),
        "validation_pass_rate": passed / scored if scored else float("nan"),
        "normal_summary_rate": normal / scored if scored else float("nan"),
        "false_normal_reports": false_normal,
        "mean_asserted_findings": float(np.mean(asserted)) if asserted else float("nan"),
        "mean_uncertain_findings": float(np.mean(uncertain_load)) if uncertain_load else float("nan"),
        "reports_with_false_absent": sum(
            any(row.get("statuses", {}).get(c) == "absent" and c in row.get("reference", []) for c in concepts)
            for row in rows),
        **scores.summary(),
    }
    return {"summary": summary, "per_concept": per_concept, "rows": rows, "hit_iou": hit_iou,
            "reference": "ChestX-Det radiologist boxes (image-level presence)"}
