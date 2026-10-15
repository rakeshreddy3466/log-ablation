"""Rule-based attribution checker: the deterministic reference ceiling.

Input is one run's (possibly category-filtered) list of trace events and nothing else: the
function never receives a label of any kind (which fault was injected, how the run ended, what
the truth is) and never looks at the run identifier, which in this repo happens to end in the
fault's name. Each rule needs one category;
when that category is filtered out the rule cannot fire, so the answer degrades to "unknown".

Rules are checked from the earliest pipeline step to the latest and the first one that fires
names the fault. first_error_step then follows from the variant (taxonomy.FIRST_ERROR_STEP).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..agents.prompts import REASONING_SYSTEM
from ..scenarios.taxonomy import FIRST_ERROR_STEP

# R2 fallback when the prompt options (C3) are not logged: the reasoning answer fields' mean
# logprob (minimum over fields) below this value. Set on the design side only
# (results/rules_calibration.json).
R2_LOGPROB_THRESHOLD = -0.675849

Events = list[dict[str, Any]]
STEP_ORDER = ("evidence_agent", "handover_1", "reasoning_agent", "handover_2", "action_agent")
HANDOVER_KEYS = ("incident_id", "affected_service", "time_window", "findings")


def _of(events: Events, category: str, step: str | None = None) -> Events:
    return [e for e in events if e["category"] == category and (step is None or e["step"] == step)]


def _tool_calls(events: Events) -> list[dict[str, Any]]:
    return [e["payload"] for e in _of(events, "tool_calls")]


# ---- evidence rules (C2) ---------------------------------------------------------------------


def rule_e1(events: Events) -> bool:
    """A get_metrics result window differs from the window that was requested."""
    for c in _tool_calls(events):
        r = c.get("result")
        if c.get("tool") == "get_metrics" and isinstance(r, dict) and "window" in r:
            if list(r["window"]) != [c["args"].get("start"), c["args"].get("end")]:
                return True
    return False


def rule_e2(events: Events) -> bool:
    """A tool result is about another service than the one that was asked for."""
    for c in _tool_calls(events):
        r, a = c.get("result"), c.get("args", {})
        if (
            isinstance(r, dict)
            and "service" in r
            and "service" in a
            and r["service"] != a["service"]
        ):
            return True
    return False


# Metrics that have a dedicated alert type in the monitoring setup (db pool saturation, cache
# eviction rate, certificate expiry, memory near limit, cross-zone rpc latency). A service that
# moves one of these without any alert on it is missing its alert.
ALERTED_METRICS = frozenset(
    {
        "db_pool_in_use",
        "evictions_per_s",
        "cert_seconds_to_expiry",
        "memory_mb",
        "payment_db_rpc_latency_ms",
    }
)


def _anomalies_without_alert(events: Events, *, only_alerted_metrics: bool) -> bool:
    calls = _tool_calls(events)
    alerted: set[str] = set()
    saw_alerts = False
    deployed: set[str] = set()
    for c in calls:
        if c.get("tool") == "get_alerts" and isinstance(c.get("result"), list):
            saw_alerts = True
            alerted |= {a.get("service") for a in c["result"]}
        if c.get("tool") == "get_deploys" and isinstance(c.get("result"), list):
            deployed |= {d.get("service") for d in c["result"]}
    if not saw_alerts:
        return False
    for c in calls:
        r = c.get("result")
        if c.get("tool") != "get_metrics" or not isinstance(r, dict) or r.get("service") in alerted:
            continue
        for name, m in r.get("metrics", {}).items():
            if not (isinstance(m, dict) and m.get("first_crossed")):
                continue
            if not only_alerted_metrics:
                return True
            if name in ALERTED_METRICS or (name == "error_rate" and r.get("service") in deployed):
                return True
    return False


def rule_e3_literal(events: Events) -> bool:
    """As first specified: any service with an anomaly (first_crossed) and no alert at all.
    Kept for the report; it also fires on downstream victims in clean runs."""
    return _anomalies_without_alert(events, only_alerted_metrics=False)


def rule_e3(events: Events) -> bool:
    """A service moves a metric that has its own alert type (or error_rate right after a deploy
    on it) while get_alerts has no alert for that service."""
    return _anomalies_without_alert(events, only_alerted_metrics=True)


# ---- reasoning rules (C3, with a C4 fallback for R2) -----------------------------------------


def rule_r1(events: Events) -> bool:
    """The reasoning system prompt differs from the frozen v1 template."""
    for e in _of(events, "prompt_shown", "reasoning_agent"):
        msgs = e["payload"].get("messages") or []
        sys_msgs = [m["content"] for m in msgs if m.get("role") == "system"]
        if sys_msgs and sys_msgs[0] != REASONING_SYSTEM:
            return True
    return False


def rule_r2(events: Events) -> bool:
    """Reasoning ran with temperature > 0 (C3); without C3, unusually low answer logprobs (C4)."""
    prompts = _of(events, "prompt_shown", "reasoning_agent")
    if prompts:
        return any((e["payload"].get("options") or {}).get("temperature", 0) > 0 for e in prompts)
    for e in _of(events, "confidence", "reasoning_agent"):
        means = [
            f["mean_logprob"]
            for f in (e["payload"].get("fields") or {}).values()
            if f and f.get("mean_logprob") is not None
        ]
        if means and min(means) < R2_LOGPROB_THRESHOLD:
            return True
    return False


# ---- interface rules (C6, or C1 by comparing the neighbouring steps) -------------------------


def _handover_pair(events: Events, handover: str, producer: str, consumer: str) -> tuple[Any, Any]:
    """(sent, received) objects of a handover: C6 if logged, else the producer's output and the
    consumer's input from C1. (None, None) when neither category is present."""
    h = _of(events, "handover_state", handover)
    if h:
        return h[0]["payload"].get("sent"), h[0]["payload"].get("received")
    out, inp = _of(events, "step_io", producer), _of(events, "step_io", consumer)
    if out and inp:
        return out[0]["payload"].get("output"), inp[0]["payload"].get("input")
    return None, None


def _removed_fields(sent: Any, received: Any) -> list[str]:
    if not isinstance(sent, dict) or not isinstance(received, dict):
        return []
    return [k for k, v in sent.items() if v is not None and received.get(k) is None]


def rule_i1(events: Events) -> bool:
    """Handover 1 dropped a field (affected_service, time_window, ...)."""
    sent, received = _handover_pair(events, "handover_1", "evidence_agent", "reasoning_agent")
    return any(k != "findings" for k in _removed_fields(sent, received))


def rule_i2(events: Events) -> bool:
    """Handover 1 passed on fewer findings than it received."""
    sent, received = _handover_pair(events, "handover_1", "evidence_agent", "reasoning_agent")
    if not isinstance(sent, dict) or not isinstance(received, dict):
        return False
    s, r = sent.get("findings"), received.get("findings")
    return isinstance(s, list) and isinstance(r, list) and len(r) < len(s)


def rule_i3(events: Events) -> bool:
    """Handover 2 removed suspected_service."""
    sent, received = _handover_pair(events, "handover_2", "reasoning_agent", "action_agent")
    return "suspected_service" in _removed_fields(sent, received)


# Earliest step first. I1 before I2: a dropped field is the more specific change.
RULES: list[tuple[str, str, Callable[[Events], bool]]] = [
    ("E1", "evidence_agent", rule_e1),
    ("E2", "evidence_agent", rule_e2),
    ("E3", "evidence_agent", rule_e3),
    ("I1", "handover_1", rule_i1),
    ("I2", "handover_1", rule_i2),
    ("R1", "reasoning_agent", rule_r1),
    ("R2", "reasoning_agent", rule_r2),
    ("I3", "handover_2", rule_i3),
]
# category each rule needs (the C1 / C6 alternatives for I1-I3 are both listed)
NEEDS: dict[str, tuple[str, ...]] = {
    "E1": ("tool_calls",),
    "E2": ("tool_calls",),
    "E3": ("tool_calls",),
    "R1": ("prompt_shown",),
    "R2": ("prompt_shown", "confidence"),
    "I1": ("handover_state", "step_io"),
    "I2": ("handover_state", "step_io"),
    "I3": ("handover_state", "step_io"),
}


def attribute(events: Events) -> dict[str, Any]:
    """{first_error_step, root_cause_variant | "unknown", fired_rules}."""
    fired = [name for name, _step, fn in RULES if fn(events)]
    if not fired:
        return {"first_error_step": "unknown", "root_cause_variant": "unknown", "fired_rules": []}
    # rules are ordered by pipeline step, so the first fired rule is the earliest error
    first = fired[0]
    assert FIRST_ERROR_STEP[first] in STEP_ORDER
    return {
        "first_error_step": FIRST_ERROR_STEP[first],
        "root_cause_variant": first,
        "fired_rules": fired,
    }
