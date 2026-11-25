"""Clinical knowledge base of *approved* recommendation guidelines.

Intentionally empty in the research prototype: the system emits no follow-up recommendations until a
clinical safety officer approves specific guideline rules (e.g. a pulmonary-nodule follow-up guideline
with its exact applicability criteria) and records them here with version and source. The report
validator rejects any recommendation whose ``guideline_id`` is not in this registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Guideline:
    id: str
    title: str
    version: str
    source_url: str
    applies_to: tuple[str, ...]  # concept keys
    approved_by: str
    approved_on: str


APPROVED_GUIDELINES: dict[str, Guideline] = {}
