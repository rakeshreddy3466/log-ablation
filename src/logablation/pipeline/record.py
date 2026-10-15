"""The run record: everything one pipeline run produced, as one validated, JSON-able object."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Outcome = Literal["clean_success", "natural_failure", "manifested", "masked"]
STEPS = ("evidence_agent", "handover_1", "reasoning_agent", "handover_2", "action_agent")


class StepRecord(BaseModel):
    step: str
    ok: bool
    error: str | None = None
    attempts: int = 0
    wall_s: float = 0.0
    prompt_tokens: int | None = None
    eval_tokens: int | None = None


class HandoverRecord(BaseModel):
    name: str  # handover_1 | handover_2
    sent: dict[str, Any]
    received: dict[str, Any]
    expected_schema: str  # the model the receiver expects
    schema_ok: bool
    schema_error: str | None = None
    diff: list[dict[str, Any]] = Field(default_factory=list)


class RunRecord(BaseModel):
    run_id: str
    scenario_id: str
    incident_type: str
    variant: str | None  # None = clean run
    twin_run_id: str | None  # the clean run of the same scenario (faulted runs only)
    model: str
    num_ctx: int
    ground_truth: dict[str, str]  # root_cause, correct_action
    evidence_bundle: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    action_plan: dict[str, Any] | None = None
    handovers: list[HandoverRecord] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    failed_step: str | None = None  # first agent that produced no valid output, if any
    root_cause_correct: bool = False
    action_correct: bool = False
    success: bool = False
    outcome: Outcome | None = None
    timings: dict[str, float] = Field(default_factory=dict)  # step -> seconds, plus total_s
    started_at: str = ""
    foreign_models: list[str] = Field(default_factory=list)  # /api/ps after the run

    @property
    def contaminated(self) -> bool:
        return bool(self.foreign_models)


def label_outcome(variant: str | None, success: bool, twin_success: bool | None = None) -> Outcome:
    """Deterministic outcome label.

    clean run: clean_success / natural_failure.
    faulted run: manifested (fails; its twin succeeded) / masked (still succeeds). A faulted run
    whose twin did not succeed is ineligible by design and is an error, never silently labelled.
    """
    if variant is None:
        return "clean_success" if success else "natural_failure"
    if twin_success is not True:
        raise ValueError("a faulted run needs a twin that succeeded (ineligible scenario)")
    return "masked" if success else "manifested"
