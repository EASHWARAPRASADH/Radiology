"""Versioned report template library (Phase 3, WP 3.1).

Templates live in data (``templates/cxr_v1.yaml``) so radiologists can author and sign them off without code
changes; the generator records the library id and version in every draft. Loading is fail-closed: a missing
sentence key or provenance suffix raises instead of falling back to free text.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from radio.ontology.cxr import CONCEPTS
from radio.schema.findings import Finding, Laterality, Provenance

REQUIRED_SENTENCES = (
    "study", "view_known", "view_unknown", "indication", "history", "priors_not_compared", "no_priors",
    "quality_limited", "present_finding", "present_diagnosis", "uncertain_one", "uncertain_many", "absent_group",
    "not_assessed_group", "critical", "normal_summary", "partial_negative_summary", "priors_compared",
    "comparison_new", "comparison_persistent", "comparison_resolved",
)
# Provenances a current-study finding may carry in a generated report. clinical_history / prior_report /
# study_metadata describe context, not findings in this image, so the generator refuses them.
FINDING_PROVENANCES = (Provenance.image_observed, Provenance.model_inferred, Provenance.radiologist_entered)
DEFAULT_LIBRARY = "cxr_v1.yaml"


class TemplateError(ValueError):
    pass


@dataclass(frozen=True)
class TemplateLibrary:
    id: str
    version: str
    status: str
    reviewed_by: str | None
    sentences: dict[str, str]
    provenance_suffix: dict[str, str]
    concept_overrides: dict[str, dict[str, str]]

    @property
    def tag(self) -> str:
        return f"{self.id}@{self.version}"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TemplateLibrary:
        missing = [k for k in REQUIRED_SENTENCES if k not in (raw.get("sentences") or {})]
        if missing:
            raise TemplateError(f"template library missing sentences: {missing}")
        suffix = raw.get("provenance_suffix") or {}
        missing_p = [p.value for p in FINDING_PROVENANCES if p.value not in suffix]
        if missing_p:
            raise TemplateError(f"template library missing provenance_suffix for: {missing_p}")
        overrides = raw.get("concept_overrides") or {}
        unknown = sorted(set(overrides) - set(CONCEPTS))
        if unknown:
            raise TemplateError(f"concept_overrides for unknown concepts: {unknown}")
        review = raw.get("clinical_review") or {}
        return cls(id=raw["id"], version=str(raw["version"]), status=raw.get("status", "engineering_draft"),
                   reviewed_by=review.get("reviewed_by"), sentences=dict(raw["sentences"]),
                   provenance_suffix=dict(suffix), concept_overrides=dict(overrides))

    # --- rendering -------------------------------------------------------------------------------------
    def render(self, key: str, **kw: str) -> str:
        try:
            text = self.sentences[key].format(**kw)
        except KeyError as e:
            raise TemplateError(f"template '{key}' needs placeholder {e}") from None
        return text[:1].upper() + text[1:]

    def side(self, f: Finding) -> str:
        if not CONCEPTS[f.code].laterality_applicable:
            return ""
        return {Laterality.left: "left ", Laterality.right: "right ", Laterality.bilateral: "bilateral "}.get(
            f.laterality, "")

    def suffix(self, f: Finding) -> str:
        if f.provenance not in FINDING_PROVENANCES:
            raise TemplateError(f"{f.code}: provenance {f.provenance.value} cannot be reported as a current finding")
        return self.provenance_suffix[f.provenance.value]

    def present(self, f: Finding) -> str:
        c = CONCEPTS[f.code]
        key = "present_diagnosis" if c.category == "diagnosis" else "present_finding"
        tmpl = self.concept_overrides.get(f.code, {}).get(key, self.sentences[key])
        text = tmpl.format(side=self.side(f), display=c.display,
                           loc=f" in the {f.location}" if f.location else "",
                           meas="".join(f", {m.name} {m.value:g} {m.unit}" for m in f.measurements),
                           suffix=self.suffix(f))
        return text[:1].upper() + text[1:]

    def named(self, f: Finding) -> str:
        return f"{self.side(f)}{CONCEPTS[f.code].display}"


def load_library(name_or_path: str = DEFAULT_LIBRARY) -> TemplateLibrary:
    if "/" in name_or_path:
        with open(name_or_path) as fh:
            return TemplateLibrary.from_dict(yaml.safe_load(fh))
    return _load_packaged(name_or_path)


@lru_cache(maxsize=4)
def _load_packaged(name: str) -> TemplateLibrary:
    text = resources.files("radio.report").joinpath("templates", name).read_text()
    return TemplateLibrary.from_dict(yaml.safe_load(text))
