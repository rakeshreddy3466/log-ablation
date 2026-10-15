"""Prompt text. Kept in one file so prompt rendering is testable without a model."""

from __future__ import annotations

from typing import Any

from ..scenarios.taxonomy import ACTIONS, INCIDENT_TYPES, SERVICES
from ..tools.ticket import IncidentTicket
from .structured import dump

# A short runbook: what each incident type looks like in the data. Shared by reasoning only.
SIGNATURES: dict[str, str] = {
    "db_pool_exhausted": "db_pool_in_use at its limit, checkout latency and errors rising, "
    "pool timeout log lines",
    "bad_deploy": "errors or latency start right after a deploy/config change on a service",
    "memory_leak": "memory_mb climbing steadily to its limit, then restarts or OOM log lines",
    "cross_zone_latency": "latency rises between services in different zones, no errors in "
    "the callee itself, no deploy",
    "expired_certificate": "tls_handshake_failures rise, cert_seconds_to_expiry below 0, "
    "certificate expired log lines",
    "cache_eviction_storm": "cache evictions and miss rate spike, then load and latency on "
    "the database and callers",
}

EVIDENCE_SYSTEM = (
    "You are the evidence agent in an incident-triage pipeline for an online shop. "
    "You are given an incident ticket and the raw results of monitoring tools. "
    "Write down the facts that matter for finding the cause: which metrics moved and when, "
    "which alerts fired, which log lines are errors, which deploys happened. "
    "Report only what the tool results show, with concrete numbers, times and service names. "
    "Do not name a root cause. Output only JSON."
)

REASONING_SYSTEM = (
    "You are the reasoning agent in an incident-triage pipeline for an online shop. "
    "You receive an evidence bundle and must decide the single root cause. "
    "Use only the evidence bundle; cite the ids of the findings you relied on.\n"
    "Possible root causes and what they usually look like:\n"
    + "\n".join(f"- {k}: {SIGNATURES[k]}" for k in INCIDENT_TYPES)
    + "\nservices: "
    + ", ".join(SERVICES)
    + ". Output only JSON."
)

ACTION_SYSTEM = (
    "You are the action agent in an incident-triage pipeline for an online shop. "
    "You receive a diagnosis and must choose the one remediation action for it, "
    "using only the diagnosis.\nRunbook (root cause -> action):\n"
    + "\n".join(f"- {k} -> {ACTIONS[k]}" for k in INCIDENT_TYPES)
    + "\nservices: "
    + ", ".join(SERVICES)
    + ". target_service is the service the action is applied to. Output only JSON."
)


def render_evidence_user(ticket: IncidentTicket, calls: list[dict[str, Any]]) -> str:
    parts = [
        f"Incident ticket: {ticket.incident_id}",
        f"Summary: {ticket.summary}",
        f"Symptom service: {ticket.symptom_service}",
        f"Investigation window: {ticket.time_window[0]} to {ticket.time_window[1]}",
        "",
        "Tool results:",
    ]
    for c in calls:
        args = ", ".join(f"{k}={v}" for k, v in c["args"].items() if v is not None)
        parts.append(f"## {c['tool']}({args})\n{dump(c['result'])}")
    parts.append(
        "\nWrite up to 8 findings (most important first), the service where the problem "
        "originates, and your confidence in the evidence (0 to 1)."
    )
    return "\n".join(parts)


def render_reasoning_user(bundle: dict[str, Any]) -> str:
    return (
        f"Evidence bundle:\n{dump(bundle)}\n\nGive the root cause, the suspected service, a "
        "short reasoning summary, your confidence (0 to 1) and the finding ids you used."
    )


def render_action_user(diagnosis: dict[str, Any]) -> str:
    return (
        f"Diagnosis:\n{dump(diagnosis)}\n\nChoose the action, its target service, a short "
        "justification and your confidence (0 to 1)."
    )
