"""The incident ticket: what the on-call engineer receives, and the pipeline's starting input."""

from __future__ import annotations

from pydantic import BaseModel

from ..scenarios.models import Scenario
from .timeutil import parse, shift

WINDOW_BEFORE_MIN = 20
WINDOW_AFTER_MIN = 5


class IncidentTicket(BaseModel):
    incident_id: str
    reported_at: str
    symptom_service: str
    summary: str
    time_window: tuple[str, str]  # suggested investigation window, inclusive


def build_ticket(scenario: Scenario) -> IncidentTicket:
    """Build the ticket from the first non-info alert on the paged (symptom) service.

    Only the symptom alert's time, service, severity and name are used: never the root cause,
    the onset time, the decisive alert or the action. The summary deliberately states the
    symptom only.
    """
    # Who was paged is part of the ticket (it is a symptom, not the cause).
    sym = scenario.ground_truth.symptom_service
    paged = [a for a in scenario.alerts if a.service == sym and a.severity != "info"]
    if not paged:
        raise ValueError(f"{scenario.scenario_id}: no symptom alert on {sym}")
    first = min(paged, key=lambda a: parse(a.time))
    summary = (
        f"{first.severity.capitalize()} alert {first.name} fired on {first.service} at "
        f"{first.time}; customers are affected and the cause and fix are still unknown."
    )
    return IncidentTicket(
        incident_id="INC-" + scenario.scenario_id.split("-", 1)[1],
        reported_at=first.time,
        symptom_service=first.service,
        summary=summary,
        time_window=(shift(first.time, -WINDOW_BEFORE_MIN), shift(first.time, WINDOW_AFTER_MIN)),
    )
