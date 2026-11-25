"""Modality plug-in interface. CT/MRI analyzers implement the same contract and emit the same IR, so
report generation, validation, review, export and audit (stages 10-14) are shared.

The router picks the first analyzer that accepts the input; if none does, the study is unsupported and
no draft is produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from radio.ingest.dicom import SUPPORTED_MODALITIES
from radio.pipeline.run import AnalysisOutput
from radio.schema.findings import ClinicalContext


@runtime_checkable
class Analyzer(Protocol):
    modalities: frozenset[str]
    body_region: str
    version: str

    def accepts(self, modality: str, body_part: str | None) -> bool: ...

    def analyze(self, path: str | Path, *, context: ClinicalContext | None = None) -> AnalysisOutput: ...


class CXRAnalyzer:
    """Adapter exposing the chest X-ray pipeline through the Analyzer contract."""

    modalities = frozenset(SUPPORTED_MODALITIES)
    body_region = "CHEST"

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.version = pipeline.bundle.model_version

    def accepts(self, modality: str, body_part: str | None) -> bool:
        return modality.upper() in self.modalities and (body_part is None or "CHEST" in body_part.upper())

    def analyze(self, path: str | Path, *, context: ClinicalContext | None = None) -> AnalysisOutput:
        return self.pipeline.analyze(path, context=context)


def route(analyzers: list[Analyzer], modality: str, body_part: str | None) -> Analyzer | None:
    return next((a for a in analyzers if a.accepts(modality, body_part)), None)
