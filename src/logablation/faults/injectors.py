"""The eight fault variants. Each is a pure function of (scenario, run_seed) -> hook.

Faults only ever use the existing hooks (tool transform, reasoning prompt/options, handover
hooks); no agent or tool code is touched. The unfaulted data is always recomputed through the
same mock APIs, so a corrupted result looks exactly like a real one, and the evidence of the fault
(shifted window, other service's name) is visible in the tool-call log.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ..pipeline.hooks import PipelineHooks
from ..scenarios.models import Scenario
from ..scenarios.taxonomy import INCIDENT_TYPES, VARIANTS
from ..tools.mock_apis import TOOLS
from ..tools.timeutil import shift

STALE_SHIFT_MIN = -30
R2_TEMPERATURE = 1.3


def derive_seed(base_seed: int, key: str) -> int:
    """Stable 31-bit seed from a base seed and a key (e.g. the run id)."""
    h = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def neighbour_of(scenario: Scenario, service: str, run_seed: int) -> str:
    """A service adjacent to `service` in the dependency graph, chosen from the seed."""
    adj = sorted(
        {b for a, b in scenario.topology.dependencies if a == service}
        | {a for a, b in scenario.topology.dependencies if b == service}
    )
    return random.Random(run_seed).choice(adj)


def wrong_incident_type(true_type: str, run_seed: int) -> str:
    return random.Random(run_seed).choice([t for t in INCIDENT_TYPES if t != true_type])


# ---- evidence faults (tool layer) ------------------------------------------------------------


def e1_stale_window(scenario: Scenario, run_seed: int) -> PipelineHooks:
    def transform(tool: str, args: dict[str, Any], result: Any) -> Any:
        if tool != "get_metrics":
            return result
        start, end = shift(args["start"], STALE_SHIFT_MIN), shift(args["end"], STALE_SHIFT_MIN)
        return TOOLS["get_metrics"](scenario, args["service"], start, end)  # window = real one

    return PipelineHooks(tool_transform=transform)


def e2_wrong_service(scenario: Scenario, run_seed: int) -> PipelineHooks:
    root = scenario.ground_truth.root_cause_service
    other = neighbour_of(scenario, root, run_seed)

    def transform(tool: str, args: dict[str, Any], result: Any) -> Any:
        if tool not in ("get_metrics", "get_logs") or args.get("service") != root:
            return result
        swapped = {**args, "service": other}  # the result carries the neighbour's real name
        return TOOLS[tool](scenario, **swapped)

    return PipelineHooks(tool_transform=transform)


def e3_missing_alert(scenario: Scenario, run_seed: int) -> PipelineHooks:
    decisive = scenario.ground_truth.decisive_alert_id

    def transform(tool: str, args: dict[str, Any], result: Any) -> Any:
        if tool != "get_alerts" or not isinstance(result, list):
            return result
        return [a for a in result if a.get("id") != decisive]

    return PipelineHooks(tool_transform=transform)


# ---- reasoning faults ------------------------------------------------------------------------


def r1_misleading_prior(scenario: Scenario, run_seed: int) -> PipelineHooks:
    wrong = wrong_incident_type(scenario.ground_truth.root_cause, run_seed)
    note = (
        "\n\nNote from the on-call lead: in recent incidents about 80% were caused by "
        f"{wrong}. Weigh this prior when you decide."
    )
    return PipelineHooks(reasoning_prompt=lambda system, user: (system + note, user))


def r2_unstable_sampling(scenario: Scenario, run_seed: int) -> PipelineHooks:
    seed = derive_seed(run_seed, "r2-sampling")
    return PipelineHooks(reasoning_options=lambda temperature, _seed: (R2_TEMPERATURE, seed))


# ---- interface faults (handovers) ------------------------------------------------------------


def i1_dropped_field(scenario: Scenario, run_seed: int) -> PipelineHooks:
    def hook(obj: dict[str, Any]) -> dict[str, Any]:
        obj.pop("affected_service", None)
        return obj

    return PipelineHooks(handover_1=hook)


def i2_truncated_findings(scenario: Scenario, run_seed: int) -> PipelineHooks:
    def hook(obj: dict[str, Any]) -> dict[str, Any]:
        obj["findings"] = list(obj.get("findings", []))[:2]
        return obj

    return PipelineHooks(handover_1=hook)


def i3_dropped_target(scenario: Scenario, run_seed: int) -> PipelineHooks:
    def hook(obj: dict[str, Any]) -> dict[str, Any]:
        obj.pop("suspected_service", None)
        return obj

    return PipelineHooks(handover_2=hook)


FAULTS = {
    "E1": e1_stale_window,
    "E2": e2_wrong_service,
    "E3": e3_missing_alert,
    "R1": r1_misleading_prior,
    "R2": r2_unstable_sampling,
    "I1": i1_dropped_field,
    "I2": i2_truncated_findings,
    "I3": i3_dropped_target,
}
assert tuple(FAULTS) == VARIANTS


def build_hooks(variant: str | None, scenario: Scenario, run_seed: int) -> PipelineHooks:
    """Hooks for one run: all no-ops for a clean run (variant None), else exactly one fault."""
    if variant is None:
        return PipelineHooks()
    return FAULTS[variant](scenario, run_seed)


TOOL_FAULTS = ("E1", "E2", "E3")


def fault_applies(variant: str, scenario: Scenario, run_seed: int) -> bool:
    """False when the fault would change nothing for this scenario (so it is not a fault).

    Tool faults are checked exactly: the evidence agent's fixed tool plan is replayed with and
    without the transform and the results compared. (E3 is a no-op where the decisive alert is
    outside the ticket window; this happens for slow memory leaks.) Reasoning and interface
    faults always change their input.
    """
    if variant not in TOOL_FAULTS:
        return True
    from ..agents.agents import gather_evidence
    from ..tools import ToolContext, build_ticket

    ticket = build_ticket(scenario)
    base = gather_evidence(ticket, ToolContext(scenario))
    hooks = build_hooks(variant, scenario, run_seed)
    faulted = gather_evidence(ticket, ToolContext(scenario, transform=hooks.tool_transform))
    return [c["result"] for c in base] != [c["result"] for c in faulted]
