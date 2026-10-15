"""The LangGraph pipeline: evidence -> handover_1 -> reasoning -> handover_2 -> action -> outcome.

Handovers are real nodes, so a fault there is injected and logged at a place that exists in the
graph. If an agent produces no valid output the run stops being fed forward (conditional edge)
and goes straight to outcome_check, which records the failure.
"""

from __future__ import annotations

import copy
import operator
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from ..agents import (
    Diagnosis,
    EvidenceBundle,
    LLMClient,
    action_agent,
    evidence_agent,
    reasoning_agent,
)
from ..agents.agents import AgentResult
from ..agents.llm import NUM_CTX
from ..scenarios.models import Scenario
from ..scenarios.taxonomy import FIRST_ERROR_STEP
from ..tools import ToolContext, build_ticket
from .diff import field_diff
from .hooks import Event, HookedLLM, PipelineHooks, TimedLLM, now
from .record import HandoverRecord, RunRecord, StepRecord, label_outcome, outcome_success


class State(TypedDict, total=False):
    evidence_bundle: dict[str, Any] | None
    handover_1_received: dict[str, Any] | None
    diagnosis: dict[str, Any] | None
    handover_2_received: dict[str, Any] | None
    action_plan: dict[str, Any] | None
    handovers: Annotated[list[HandoverRecord], operator.add]
    steps: Annotated[list[StepRecord], operator.add]
    timings: Annotated[list[tuple[str, float]], operator.add]
    halted: bool
    failed_step: str | None
    root_cause_correct: bool
    action_correct: bool
    target_correct: bool
    success: bool


def _step_record(step: str, res: AgentResult, wall: float) -> StepRecord:
    t = res.llm.timings
    return StepRecord(
        step=step,
        ok=res.ok,
        error=res.llm.error,
        attempts=len(res.llm.attempts),
        wall_s=round(wall, 3),
        prompt_tokens=t.get("prompt_eval_count"),
        eval_tokens=t.get("eval_count"),
    )


def build_graph(
    run_id: str,
    scenario: Scenario,
    llm: LLMClient,
    hooks: PipelineHooks,
) -> Any:
    """Compile the graph for one run. Nodes close over the run's scenario, llm and hooks."""
    ticket = build_ticket(scenario)
    tools = ToolContext(scenario, transform=hooks.tool_transform)
    timed = TimedLLM(llm)  # every agent's model calls pass through it (timing only)
    reasoning_llm = HookedLLM(timed, hooks.reasoning_prompt, hooks.reasoning_options)
    truth = scenario.ground_truth

    def emit(step: str, phase: str, **data: Any) -> None:
        hooks.on_event(Event(run_id, step, phase, now(), data))

    def agent_node(
        step: str,
        key: str,
        call: Callable[[State], AgentResult],
        inputs: Callable[[State], dict[str, Any]],
    ) -> Callable[[State], State]:
        def node(state: State) -> State:
            emit(step, "start", input=inputs(state))
            t0 = time.perf_counter()
            res = call(state)
            wall = time.perf_counter() - t0
            out = res.output.model_dump(mode="json") if isinstance(res.output, BaseModel) else None
            emit(
                step,
                "end",
                output=out,
                agent_result=res,
                wall_s=wall,
                model_timings=timed.drain(),
                tool_timings=list(tools.timings) if step == "evidence_agent" else [],
            )
            update: State = {
                key: out,  # type: ignore[misc]
                "steps": [_step_record(step, res, wall)],
                "timings": [(step, wall)],
            }
            if out is None:
                update["halted"] = True
                update["failed_step"] = step
            return update

        return node

    def handover_node(
        name: str, source: str, hook: Callable[[dict[str, Any]], dict[str, Any]], schema: type
    ) -> Callable[[State], State]:
        def node(state: State) -> State:
            sent = copy.deepcopy(state[source])  # type: ignore[literal-required]
            emit(name, "start", sent=sent)
            t0 = time.perf_counter()
            received = hook(copy.deepcopy(sent))
            wall = time.perf_counter() - t0
            schema_error: str | None = None
            try:
                schema.model_validate(received)
            except ValidationError as e:
                schema_error = str(e)[:400]
            rec = HandoverRecord(
                name=name,
                sent=sent,
                received=received,
                expected_schema=schema.__name__,
                schema_ok=schema_error is None,
                schema_error=schema_error,
                diff=field_diff(sent, received),
            )
            emit(name, "end", record=rec, wall_s=wall)
            return {  # type: ignore[return-value]
                f"{name}_received": received,
                "handovers": [rec],
                "steps": [StepRecord(step=name, ok=True, wall_s=round(wall, 6))],
                "timings": [(name, wall)],
            }

        return node

    def outcome_check(state: State) -> State:
        emit("outcome_check", "start")
        dg, ap = state.get("diagnosis"), state.get("action_plan")
        rc = bool(dg) and dg["root_cause"] == truth.root_cause  # type: ignore[index]
        ac = bool(ap) and ap["action"] == truth.correct_action  # type: ignore[index]
        tc = bool(ap) and ap["target_service"] == truth.action_target  # type: ignore[index]
        ok = outcome_success(rc, ac, tc)
        emit("outcome_check", "end", success=ok)
        return {"root_cause_correct": rc, "action_correct": ac, "target_correct": tc, "success": ok}

    def route(next_node: str) -> Callable[[State], str]:
        return lambda state: "outcome_check" if state.get("halted") else next_node

    g = StateGraph(State)
    g.add_node(
        "evidence_agent",
        agent_node(
            "evidence_agent",
            "evidence_bundle",
            lambda s: evidence_agent(ticket, tools, timed),
            lambda s: ticket.model_dump(mode="json"),
        ),
    )
    g.add_node(
        "handover_1",
        handover_node("handover_1", "evidence_bundle", hooks.handover_1, EvidenceBundle),
    )
    g.add_node(
        "reasoning_agent",
        agent_node(
            "reasoning_agent",
            "diagnosis",
            lambda s: reasoning_agent(s["handover_1_received"], reasoning_llm),  # type: ignore[typeddict-item]
            lambda s: s["handover_1_received"],
        ),  # type: ignore[typeddict-item]
    )
    g.add_node("handover_2", handover_node("handover_2", "diagnosis", hooks.handover_2, Diagnosis))
    g.add_node(
        "action_agent",
        agent_node(
            "action_agent",
            "action_plan",
            lambda s: action_agent(s["handover_2_received"], timed),  # type: ignore[typeddict-item]
            lambda s: s["handover_2_received"],
        ),  # type: ignore[typeddict-item]
    )
    g.add_node("outcome_check", outcome_check)

    g.add_edge(START, "evidence_agent")
    g.add_conditional_edges("evidence_agent", route("handover_1"), ["handover_1", "outcome_check"])
    g.add_edge("handover_1", "reasoning_agent")
    g.add_conditional_edges("reasoning_agent", route("handover_2"), ["handover_2", "outcome_check"])
    g.add_edge("handover_2", "action_agent")
    g.add_edge("action_agent", "outcome_check")
    g.add_edge("outcome_check", END)
    return g.compile()


def run_pipeline(
    scenario: Scenario,
    llm: LLMClient,
    *,
    run_id: str,
    variant: str | None = None,
    twin_run_id: str | None = None,
    twin_success: bool | None = None,
    hooks: PipelineHooks | None = None,
) -> RunRecord:
    """Run one scenario through the graph and return its record (outcome label included)."""
    hooks = hooks or PipelineHooks()
    started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    hooks.on_event(Event(run_id, "run", "start", now(), {"scenario_id": scenario.scenario_id}))
    final: State = build_graph(run_id, scenario, llm, hooks).invoke(
        {"handovers": [], "steps": [], "timings": []}
    )
    total = time.perf_counter() - t0
    rc_ok = bool(final.get("root_cause_correct"))
    ac_ok = bool(final.get("action_correct"))
    tc_ok = bool(final.get("target_correct"))
    outcome = label_outcome(
        variant, root_cause_ok=rc_ok, action_ok=ac_ok, target_ok=tc_ok, twin_success=twin_success
    )
    success = outcome in ("clean_success", "masked")
    timings = {name: round(sec, 3) for name, sec in final.get("timings", [])}
    timings["total_s"] = round(total, 3)
    rec = RunRecord(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        incident_type=scenario.incident_type,
        variant=variant,
        twin_run_id=twin_run_id,
        model=getattr(llm, "model", "unknown"),
        num_ctx=getattr(llm, "num_ctx", NUM_CTX),
        ground_truth={
            "root_cause": scenario.ground_truth.root_cause,
            "correct_action": scenario.ground_truth.correct_action,
        },
        evidence_bundle=final.get("evidence_bundle"),
        diagnosis=final.get("diagnosis"),
        action_plan=final.get("action_plan"),
        handovers=final.get("handovers", []),
        steps=final.get("steps", []),
        failed_step=final.get("failed_step"),
        root_cause_correct=rc_ok,
        action_correct=ac_ok,
        target_correct=tc_ok,
        success=success,
        outcome=outcome,
        first_error_step=FIRST_ERROR_STEP[variant] if outcome == "manifested" else None,  # type: ignore[index]
        root_cause_variant=variant if outcome == "manifested" else None,
        timings=timings,
        started_at=started,
    )
    hooks.on_event(Event(run_id, "run", "end", now(), {"success": success}))
    return rec
