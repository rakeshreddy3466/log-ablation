"""The three agents as plain functions. Each returns an AgentResult; none raises on bad output.

Faults never edit this file: tool faults go through ToolContext.transform, prompt/sampling
faults wrap the LLMClient, handover faults act on the objects between these functions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..tools import IncidentTicket, ToolContext
from .llm import LLMClient
from .prompts import (
    ACTION_SYSTEM,
    EVIDENCE_SYSTEM,
    REASONING_SYSTEM,
    render_action_user,
    render_evidence_user,
    render_reasoning_user,
)
from .schemas import ActionPlan, Diagnosis, EvidenceBundle, EvidenceDraft, Finding
from .structured import StructuredResult, generate_structured

LOGS_PER_SERVICE = 5


@dataclass
class AgentResult:
    output: BaseModel | None  # the validated object, or None = recorded failure
    llm: StructuredResult
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.output is not None


def _as_dict(obj: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    # Handover faults may hand over a partial object that no longer validates; accept the dict.
    return obj.model_dump(mode="json") if isinstance(obj, BaseModel) else dict(obj)


def plan_services(ticket: IncidentTicket, topology: dict[str, Any]) -> list[str]:
    """Symptom service first, then everything it depends on, nearest first."""
    edges: dict[str, list[str]] = {}
    for dep in topology.get("dependencies", []):
        a, _, b = dep.partition(" -> ")
        edges.setdefault(a, []).append(b)
    order, seen, queue = [], {ticket.symptom_service}, deque([ticket.symptom_service])
    while queue:
        svc = queue.popleft()
        order.append(svc)
        for nxt in edges.get(svc, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order


def gather_evidence(ticket: IncidentTicket, tools: ToolContext) -> list[dict[str, Any]]:
    """The fixed tool plan: topology, alerts, then metrics+logs per service, then deploys."""
    start, end = ticket.time_window
    calls: list[dict[str, Any]] = []

    def call(tool: str, **args: Any) -> Any:
        result = getattr(tools, tool)(**args)
        calls.append({"tool": tool, "args": args, "result": result})
        return result

    topology = call("get_topology")
    call("get_alerts", start=start, end=end)
    services = plan_services(ticket, topology if isinstance(topology, dict) else {})
    for svc in services:
        call("get_metrics", service=svc, start=start, end=end)
        call("get_logs", service=svc, start=start, end=end, limit=LOGS_PER_SERVICE)
    call("get_deploys", start=start, end=end)
    return calls


def evidence_agent(ticket: IncidentTicket, tools: ToolContext, llm: LLMClient) -> AgentResult:
    calls = gather_evidence(ticket, tools)
    res = generate_structured(
        llm, EvidenceDraft, EVIDENCE_SYSTEM, render_evidence_user(ticket, calls)
    )
    bundle = None
    if isinstance(res.parsed, EvidenceDraft):
        d = res.parsed
        bundle = EvidenceBundle(
            incident_id=ticket.incident_id,
            affected_service=str(d.affected_service),
            time_window=ticket.time_window,
            findings=[
                Finding(id=f"f{i}", source=str(f.source), value=f.value)
                for i, f in enumerate(d.findings, 1)
            ],
            evidence_confidence=d.evidence_confidence,
        )
    return AgentResult(output=bundle, llm=res, tool_calls=calls)


def reasoning_agent(bundle: EvidenceBundle | Mapping[str, Any], llm: LLMClient) -> AgentResult:
    data = _as_dict(bundle)
    known = {f.get("id") for f in data.get("findings", []) if isinstance(f, dict)}

    def check(d: Diagnosis) -> str | None:
        unknown = [i for i in d.evidence_used if i not in known]
        return f"evidence_used cites unknown finding ids {unknown}" if unknown else None

    res = generate_structured(
        llm, Diagnosis, REASONING_SYSTEM, render_reasoning_user(data), check=check
    )
    return AgentResult(output=res.parsed, llm=res)


def action_agent(diagnosis: Diagnosis | Mapping[str, Any], llm: LLMClient) -> AgentResult:
    res = generate_structured(
        llm, ActionPlan, ACTION_SYSTEM, render_action_user(_as_dict(diagnosis))
    )
    return AgentResult(output=res.parsed, llm=res)
