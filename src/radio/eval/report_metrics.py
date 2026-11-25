"""Report-level clinical accuracy metrics computed against a reference finding set.

Reference = radiologist-adjudicated statuses per concept (e.g. from a reader study). Generated statuses
come from the IR / report sentences. These complement (not replace) RadGraph-F1 / RaTEScore / RadFact,
which need reference free text; BLEU/ROUGE are never used alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from radio.schema.findings import AnalysisResult, FindingStatus, Laterality, SentenceKind

POSITIVE = {FindingStatus.present, FindingStatus.uncertain}


@dataclass
class ReportScores:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    contradictions: int = 0
    laterality_errors: int = 0
    laterality_checked: int = 0
    unsupported_recommendations: int = 0
    reports: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def hallucination_rate(self) -> float:
        """Fraction of asserted (present/uncertain) findings not in the reference."""
        return self.fp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    def summary(self) -> dict[str, float]:
        return {
            "finding_precision": self.precision,
            "finding_recall": self.recall,
            "hallucination_rate": self.hallucination_rate,
            "contradiction_rate": self.contradictions / self.reports if self.reports else float("nan"),
            "laterality_error_rate": (self.laterality_errors / self.laterality_checked
                                      if self.laterality_checked else float("nan")),
            "unsupported_recommendation_rate": (self.unsupported_recommendations / self.reports
                                                if self.reports else float("nan")),
            "reports": float(self.reports),
        }


def score_report(
    result: AnalysisResult,
    reference: dict[str, FindingStatus],
    reference_laterality: dict[str, Laterality] | None = None,
    scores: ReportScores | None = None,
    approved_guidelines: set[str] | None = None,
    label_space: set[str] | None = None,
) -> ReportScores:
    """Accumulate finding-level agreement. With ``label_space``, generated findings outside it are ignored
    (the reference cannot say whether they are right) instead of counted as false positives."""
    s = scores or ReportScores()
    s.reports += 1
    gen = {f.code: f for f in result.findings}
    for code, ref_status in reference.items():
        g = gen.get(code)
        if g is None or g.status == FindingStatus.not_assessed:
            if ref_status == FindingStatus.present:
                s.fn += 1  # unassessed counts as a miss for recall (disclosed, but still a miss)
            continue
        asserted = g.status in POSITIVE
        if asserted and ref_status == FindingStatus.present:
            s.tp += 1
        elif asserted:
            s.fp += 1
        elif ref_status == FindingStatus.present:
            s.fn += 1
        if reference_laterality and code in reference_laterality and asserted and g.laterality in (
            Laterality.left, Laterality.right, Laterality.bilateral
        ):
            s.laterality_checked += 1
            if g.laterality != reference_laterality[code]:
                s.laterality_errors += 1
    for code in gen:
        if label_space is not None and code not in label_space:
            continue
        if code not in reference and gen[code].status in POSITIVE:
            s.fp += 1
    if result.report is not None:
        kinds: dict[str, set[SentenceKind]] = {}
        for sents in result.report.sections().values():
            for sent in sents:
                for fid in sent.finding_ids:
                    kinds.setdefault(fid, set()).add(sent.kind)
        if any({SentenceKind.positive, SentenceKind.negative} <= k for k in kinds.values()):
            s.contradictions += 1
        ok = approved_guidelines or set()
        s.unsupported_recommendations += int(any(r.guideline_id not in ok for r in result.report.recommendations))
    return s


def word_edit_distance(a: str, b: str) -> float:
    """Normalised word-level Levenshtein distance between AI draft and radiologist-final text."""
    x, y = a.split(), b.split()
    if not x and not y:
        return 0.0
    prev = list(range(len(y) + 1))
    for i, wx in enumerate(x, 1):
        cur = [i]
        for j, wy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wx != wy)))
        prev = cur
    return prev[-1] / max(len(x), len(y))
