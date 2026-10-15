"""Typed outputs of the three agents. These are the objects that cross the handovers.

``root_cause`` and ``action`` are enums built from taxonomy.py so the schema, the generator and
the outcome check can never drift apart. Faults (milestone 5) work on the dict form of these
objects at the handovers, so the models here stay strict.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ..scenarios.taxonomy import ACTIONS, INCIDENT_TYPES, SERVICES
from ..tools.mock_apis import TOOL_NAMES

RootCause = StrEnum("RootCause", {n: n for n in INCIDENT_TYPES})
Action = StrEnum("Action", {a: a for a in ACTIONS.values()})
Service = StrEnum("Service", {s: s for s in SERVICES})
ToolName = StrEnum("ToolName", {t: t for t in TOOL_NAMES})

Confidence = Field(ge=0.0, le=1.0, description="0 = guess, 1 = certain")


class Finding(BaseModel):
    id: str
    source: str  # the tool the fact came from
    value: str


class EvidenceBundle(BaseModel):
    incident_id: str
    affected_service: str
    time_window: tuple[str, str]
    findings: list[Finding]
    evidence_confidence: float = Confidence


class Diagnosis(BaseModel):
    root_cause: RootCause
    suspected_service: Service
    reasoning_summary: str
    confidence: float = Confidence
    evidence_used: list[str]


class ActionPlan(BaseModel):
    action: Action
    target_service: Service
    justification: str
    confidence: float = Confidence


# What the evidence model is actually asked to write. incident_id and time_window are copied
# from the ticket by Python and finding ids are numbered by Python: a 3B model should not be
# trusted to copy those, and a copying slip would be an artefact, not a pipeline fault.
class FindingDraft(BaseModel):
    source: ToolName
    value: str = Field(description="one short fact with concrete numbers, times or names")


class EvidenceDraft(BaseModel):
    affected_service: Service = Field(
        description="the service where the evidence says the problem originates"
    )
    findings: list[FindingDraft] = Field(min_length=1, max_length=8)
    evidence_confidence: float = Confidence
