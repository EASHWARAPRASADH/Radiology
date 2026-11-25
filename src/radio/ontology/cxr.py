"""Internal chest X-ray finding vocabulary.

Each concept is a stable internal key. External codes (RadLex / SNOMED CT) are deliberately left
``None`` until they are looked up and verified against the official terminology release - we do not
guess codes. DICOM SR export uses the private coding scheme ``99RADIO`` for these keys until then.

Dataset label maps translate source labels into internal keys. Concepts a dataset does not label are
*not assessed* for that dataset (never implicitly negative).
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRIVATE_CODING_SCHEME = "99RADIO"


@dataclass(frozen=True)
class Concept:
    key: str
    display: str
    category: str  # "finding" (directly observable) or "diagnosis" (interpretive)
    laterality_applicable: bool
    critical_candidate: bool = False
    # Synonyms/terms used by the report validator to detect mentions in free text.
    terms: tuple[str, ...] = field(default_factory=tuple)
    radlex_id: str | None = None  # fill only from a verified RadLex lookup
    snomed_id: str | None = None  # fill only from a verified SNOMED CT lookup
    # Priority for impression ordering (lower = more important).
    priority: int = 50


CONCEPTS: dict[str, Concept] = {
    c.key: c
    for c in [
        Concept("pneumothorax", "pneumothorax", "finding", True, critical_candidate=True,
                terms=("pneumothorax", "pneumothoraces"), priority=1),
        Concept("device_malposition", "support device malposition", "finding", False,
                critical_candidate=True, terms=("malposition", "malpositioned"), priority=2),
        Concept("pleural_effusion", "pleural effusion", "finding", True,
                terms=("effusion", "effusions"), priority=10),
        Concept("consolidation", "consolidation", "finding", True, terms=("consolidation",), priority=11),
        Concept("edema", "pulmonary edema", "finding", False, terms=("edema", "oedema"), priority=12),
        Concept("lung_opacity", "lung opacity", "finding", True,
                terms=("opacity", "opacities", "infiltrate", "infiltrates", "infiltration"), priority=13),
        Concept("mass", "mass", "finding", True, terms=("mass", "masses"), priority=14),
        Concept("nodule", "nodule", "finding", True, terms=("nodule", "nodules", "nodular"), priority=15),
        Concept("pneumonia", "pneumonia", "diagnosis", True, terms=("pneumonia",), priority=16),
        Concept("cardiomegaly", "cardiomegaly", "finding", False,
                terms=("cardiomegaly", "enlarged cardiac silhouette"), priority=20),
        Concept("atelectasis", "atelectasis", "finding", True, terms=("atelectasis", "atelectatic"), priority=21),
        Concept("fracture", "fracture", "finding", True, terms=("fracture", "fractures"), priority=22),
        Concept("emphysema", "emphysema", "finding", False, terms=("emphysema", "emphysematous"), priority=30),
        Concept("fibrosis", "fibrosis", "finding", True, terms=("fibrosis", "fibrotic"), priority=31),
        Concept("pleural_thickening", "pleural thickening", "finding", True,
                terms=("pleural thickening",), priority=32),
        Concept("calcification", "calcification", "finding", True,
                terms=("calcification", "calcified"), priority=33),
        Concept("hernia", "hiatal hernia", "finding", False, terms=("hernia",), priority=34),
    ]
}

# Concepts whose absence must be explicitly assessed before the generator may state that a study shows
# "no acute cardiopulmonary abnormality".
CORE_ACUTE_CONCEPTS: tuple[str, ...] = (
    "pneumothorax", "pleural_effusion", "consolidation", "edema", "lung_opacity",
)

# NIH ChestX-ray14 (Wang et al., CVPR 2017) labels -> internal keys. Labels were NLP-mined from reports.
NIH_CXR14_LABEL_MAP: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "pleural_effusion",
    "Infiltration": "lung_opacity",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural_thickening",
    "Hernia": "hernia",
}

# ChestX-Det (Deepwise, NIH images, Apache-2.0 repo) box/mask classes -> internal keys.
CHESTX_DET_LABEL_MAP: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Calcification": "calcification",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Diffuse Nodule": "nodule",
    "Effusion": "pleural_effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Fracture": "fracture",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pleural Thickening": "pleural_thickening",
    "Pneumothorax": "pneumothorax",
}


def concept(key: str) -> Concept:
    try:
        return CONCEPTS[key]
    except KeyError as e:  # pragma: no cover - defensive
        raise KeyError(f"Unknown finding concept '{key}'") from e


def label_space_for(label_map: dict[str, str]) -> list[str]:
    """Ordered, de-duplicated internal keys covered by a dataset label map."""
    seen: list[str] = []
    for k in label_map.values():
        if k not in seen:
            seen.append(k)
    return seen
