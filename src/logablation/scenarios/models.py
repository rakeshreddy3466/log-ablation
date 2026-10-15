"""Pydantic schema of a stored scenario (what lands in data/scenarios/*.json)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Alert(BaseModel):
    id: str
    time: str
    service: str
    severity: str
    name: str
    message: str


class MetricStore(BaseModel):
    """One shared one-minute time axis; series[service][metric] aligns with timestamps."""

    timestamps: list[str]
    series: dict[str, dict[str, list[float]]]


class Deploy(BaseModel):
    id: str
    time: str
    service: str
    kind: str  # deploy | config | infra
    version: str
    author: str
    description: str


class LogLine(BaseModel):
    time: str
    service: str
    level: str
    message: str


class Topology(BaseModel):
    services: list[str]
    dependencies: list[tuple[str, str]]  # (caller, callee)
    zones: dict[str, str]
    limits: dict[str, float]


class SensitiveField(BaseModel):
    """A planted sensitive value and exactly where it sits (so we can count and score later)."""

    id: str
    type: str  # email | ip | api_key | order_id
    value: str
    source: str  # alerts | logs | deploys
    index: int  # position within that list
    field: str  # message | author
    start: int  # character span inside the field text
    end: int


class GroundTruth(BaseModel):
    incident_type: str
    root_cause: str
    root_cause_service: str
    symptom_service: str
    correct_action: str
    action_target: str
    onset_time: str
    decisive_alert_id: str


class Scenario(BaseModel):
    scenario_id: str
    seed: int
    incident_type: str
    time_window: tuple[str, str]
    topology: Topology
    alerts: list[Alert]
    metrics: MetricStore
    deploys: list[Deploy]
    logs: list[LogLine]
    sensitive_fields: list[SensitiveField] = Field(default_factory=list)
    ground_truth: GroundTruth
