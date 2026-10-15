"""Prompt text. Kept in one file so prompt rendering is testable without a model."""

from __future__ import annotations

from typing import Any

from ..scenarios.taxonomy import ACTIONS, INCIDENT_TYPES, SERVICES
from ..tools.ticket import IncidentTicket
from .schemas import MAX_FINDING_CHARS, MAX_FINDINGS
from .structured import dump

# Runbook: for each incident type, the evidence that must be present and what must be absent.
# Same shape and length for all six; no two share a distinguishing signal. Used by reasoning only.
SIGNATURES: dict[str, str] = {
    "db_pool_exhausted": "Present: checkout db_pool_in_use pinned at its limit, pool-timeout "
    "errors in checkout logs, payment-db connections near max. "
    "Absent: deploy shortly before onset, cache evictions, TLS errors, memory growth.",
    "bad_deploy": "Present: a code or config deploy on a service 1-3 minutes before errors or "
    "latency step up, exceptions naming the new build. "
    "Absent: pool at its limit, cache evictions, TLS errors, memory growth.",
    "memory_leak": "Present: memory_mb rising steadily toward its limit across the window, "
    "heap or GC warnings, slow latency drift. "
    "Absent: sudden step change, deploy before onset, pool at its limit, TLS errors.",
    "cross_zone_latency": "Present: payment_db_rpc_latency_ms jumps after an infra reschedule "
    "moved a pod to another zone, call logs show different source and destination zones. "
    "Absent: code deploy, pool at its limit, TLS errors, memory growth.",
    "expired_certificate": "Present: tls_handshake_failures rising, cert_seconds_to_expiry "
    "below 0, x509 certificate-expired errors. "
    "Absent: deploy before onset, pool at its limit, cache evictions, memory growth.",
    "cache_eviction_storm": "Present: cache evictions_per_s spike and hit_ratio drop, then "
    "catalog errors and payment-db connections rise, db_pool_in_use up but below its limit. "
    "Absent: deploy before onset, TLS errors, memory growth.",
}

EVIDENCE_SYSTEM = (
    "You are the evidence agent in an incident-triage pipeline for an online shop. "
    "You are given an incident ticket and the raw results of monitoring tools. "
    "Write down the facts that matter for finding the cause: which metrics moved and when, "
    "which alerts fired, which log lines are errors, which deploys happened. "
    f"Write at most {MAX_FINDINGS} findings, each at most {MAX_FINDING_CHARS} characters, "
    "each a different fact. A finding is a bare fact with concrete numbers, times and service "
    "names, copied from the tool results. No explanations, no opinions, no confidence "
    "statements inside findings. Do not name a root cause. Output only JSON."
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
        f"\nWrite up to {MAX_FINDINGS} findings (most important first, facts only, no repeats), "
        "the service where the problem originates, and your confidence in the evidence (0 to 1)."
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
