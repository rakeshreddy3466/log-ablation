"""Pipeline tests with a fake LLM: graph, handover records, hooks, resume, outcome labels."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from logablation.agents import LLMReply
from logablation.agents.llm import NUM_CTX, render_prompt
from logablation.pipeline import (
    Event,
    PipelineHooks,
    field_diff,
    finished_run_ids,
    label_outcome,
    plan_clean,
    run_clean_phase,
    run_pipeline,
)
from logablation.scenarios.generator import generate_scenario
from logablation.scenarios.models import Scenario
from logablation.scenarios.taxonomy import ACTIONS, INCIDENT_TYPES


@dataclass
class RoutedLLM:
    """Answers by requested schema. `root_cause` / `action` choose right or wrong answers."""

    root_cause: str
    action: str
    model: str = "fake"
    calls: list[tuple[str, str, str, float | None, int | None]] = field(default_factory=list)

    def chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMReply:
        title = schema["title"]
        self.calls.append((title, system, user, temperature, seed))
        if title == "EvidenceDraft":
            body: dict[str, Any] = {
                "affected_service": "checkout",
                "evidence_confidence": 0.7,
                "findings": [
                    {"source": "get_metrics", "value": f"finding number {i} value {i * 7}"}
                    for i in range(4)
                ],
            }
        elif title == "Diagnosis":
            body = {
                "root_cause": self.root_cause,
                "suspected_service": "checkout",
                "reasoning_summary": "because",
                "confidence": 0.8,
                "evidence_used": ["f1", "f2"],
            }
        else:
            body = {
                "action": self.action,
                "target_service": "checkout",
                "justification": "do it",
                "confidence": 0.8,
            }
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return LLMReply(
            text=json.dumps(body),
            model=self.model,
            messages=msgs,
            prompt=render_prompt(msgs),
            logprobs=[],
            timings={"prompt_eval_count": 10, "eval_count": 5},
            options={"num_ctx": NUM_CTX},
        )


def _right(sc: Scenario) -> RoutedLLM:
    return RoutedLLM(sc.ground_truth.root_cause, sc.ground_truth.correct_action)


def _wrong_rc(sc: Scenario) -> RoutedLLM:
    other = next(t for t in INCIDENT_TYPES if t != sc.ground_truth.root_cause)
    return RoutedLLM(other, sc.ground_truth.correct_action)


# ---- graph end to end ------------------------------------------------------------------------


def test_graph_runs_end_to_end_with_identity_handovers() -> None:
    sc = generate_scenario(1002)
    rec = run_pipeline(sc, _right(sc), run_id="r1")
    assert [s.step for s in rec.steps] == [
        "evidence_agent",
        "handover_1",
        "reasoning_agent",
        "handover_2",
        "action_agent",
    ]
    assert rec.evidence_bundle and rec.diagnosis and rec.action_plan
    assert rec.success and rec.outcome == "clean_success" and rec.failed_step is None
    assert rec.variant is None and rec.twin_run_id is None
    assert set(rec.timings) >= {"evidence_agent", "handover_1", "action_agent", "total_s"}
    # identity handovers: delivered object equals sent object, schema valid, diff empty
    assert [h.name for h in rec.handovers] == ["handover_1", "handover_2"]
    for h in rec.handovers:
        assert h.sent == h.received and h.diff == [] and h.schema_ok and h.schema_error is None
    assert rec.handovers[0].sent == rec.evidence_bundle
    assert rec.handovers[1].sent == rec.diagnosis
    # the record is plain JSON
    assert json.loads(rec.model_dump_json())["run_id"] == "r1"


def test_wrong_answer_is_natural_failure_and_flags_which_part() -> None:
    sc = generate_scenario(1002)
    rec = run_pipeline(sc, _wrong_rc(sc), run_id="r2")
    assert not rec.success and rec.outcome == "natural_failure"
    assert not rec.root_cause_correct and rec.action_correct


def test_agent_failure_halts_the_graph_and_is_recorded() -> None:
    sc = generate_scenario(1002)
    llm = RoutedLLM("not_a_cause", "x")  # Diagnosis never validates -> reasoning fails
    rec = run_pipeline(sc, llm, run_id="r3")
    assert rec.failed_step == "reasoning_agent" and rec.diagnosis is None
    assert rec.action_plan is None and not rec.success
    assert [s.step for s in rec.steps] == ["evidence_agent", "handover_1", "reasoning_agent"]
    assert [h.name for h in rec.handovers] == ["handover_1"]


# ---- hooks -----------------------------------------------------------------------------------


def test_handover_hook_changes_what_the_next_agent_gets_and_is_diffed() -> None:
    sc = generate_scenario(1002)

    def drop_two(obj: dict[str, Any]) -> dict[str, Any]:
        obj["findings"] = obj["findings"][:2]
        del obj["affected_service"]
        return obj

    llm = _right(sc)
    rec = run_pipeline(sc, llm, run_id="r4", hooks=PipelineHooks(handover_1=drop_two))
    h = rec.handovers[0]
    assert not h.schema_ok and "affected_service" in (h.schema_error or "")
    paths = {d["path"]: d["op"] for d in h.diff}
    assert paths == {
        "affected_service": "removed",
        "findings[2]": "removed",
        "findings[3]": "removed",
    }
    assert h.sent == rec.evidence_bundle  # the sent object is the true agent output
    reasoning_user = next(u for t, _, u, _, _ in llm.calls if t == "Diagnosis")
    assert "finding number 3" not in reasoning_user  # the reasoning agent saw the cut object


def test_reasoning_prompt_and_options_hooks_apply_only_to_the_reasoning_call() -> None:
    sc = generate_scenario(1002)
    hooks = PipelineHooks(
        reasoning_prompt=lambda s, u: (s, u + "\nHINT: it is a memory leak"),
        reasoning_options=lambda t, sd: (1.5, 99),
    )
    llm = _right(sc)
    run_pipeline(sc, llm, run_id="r5", hooks=hooks)
    by_title = {t: (u, temp, sd) for t, _, u, temp, sd in llm.calls}
    assert "HINT" in by_title["Diagnosis"][0] and by_title["Diagnosis"][1:] == (1.5, 99)
    for title in ("EvidenceDraft", "ActionPlan"):
        assert "HINT" not in by_title[title][0] and by_title[title][1] != 1.5


def test_tool_transform_hook_reaches_the_evidence_agent() -> None:
    sc = generate_scenario(1002)
    seen: list[str] = []

    def spy(name: str, args: dict[str, Any], result: Any) -> Any:
        seen.append(name)
        return result

    run_pipeline(sc, _right(sc), run_id="r6", hooks=PipelineHooks(tool_transform=spy))
    assert seen[0] == "get_topology" and seen[-1] == "get_deploys"


def test_event_hook_sees_every_step_boundary_in_order() -> None:
    sc = generate_scenario(1002)
    events: list[Event] = []
    run_pipeline(sc, _right(sc), run_id="r7", hooks=PipelineHooks(on_event=events.append))
    seq = [(e.step, e.phase) for e in events]
    expected = []
    for s in (
        "evidence_agent",
        "handover_1",
        "reasoning_agent",
        "handover_2",
        "action_agent",
        "outcome_check",
    ):
        expected += [(s, "start"), (s, "end")]
    assert seq == [("run", "start"), *expected, ("run", "end")]
    assert all(e.run_id == "r7" for e in events)
    end = next(e for e in events if e.step == "action_agent" and e.phase == "end")
    assert end.data["output"]["action"] in set(ACTIONS.values())


# ---- diff ------------------------------------------------------------------------------------


def test_field_diff_identity_and_each_op() -> None:
    a = {"x": 1, "y": [1, 2, 3], "z": {"k": "v"}}
    assert field_diff(a, {"x": 1, "y": [1, 2, 3], "z": {"k": "v"}}) == []
    d = field_diff(a, {"x": 2, "y": [1, 2], "z": {"k": "v", "n": 0}, "w": 5})
    assert {(i["path"], i["op"]) for i in d} == {
        ("x", "changed"),
        ("y[2]", "removed"),
        ("z.n", "added"),
        ("w", "added"),
    }


# ---- outcome labels (hand-written cases) -------------------------------------------------------


@pytest.mark.parametrize(
    ("variant", "success", "twin", "expected"),
    [
        (None, True, None, "clean_success"),
        (None, False, None, "natural_failure"),
        ("E1", False, True, "manifested"),
        ("E1", True, True, "masked"),
        ("R2", False, True, "manifested"),  # R2 is labelled like every other variant
    ],
)
def test_label_outcome(
    variant: str | None, success: bool, twin: bool | None, expected: str
) -> None:
    assert label_outcome(variant, success, twin) == expected


@pytest.mark.parametrize("twin", [None, False])
def test_faulted_run_without_a_successful_twin_is_an_error(twin: bool | None) -> None:
    with pytest.raises(ValueError):
        label_outcome("E2", True, twin)


def test_faulted_record_carries_twin_id_and_label() -> None:
    sc = generate_scenario(1002)
    rec = run_pipeline(
        sc, _wrong_rc(sc), run_id="r8", variant="E1", twin_run_id="r1", twin_success=True
    )
    assert rec.variant == "E1" and rec.twin_run_id == "r1" and rec.outcome == "manifested"


# ---- planning and resume -----------------------------------------------------------------------


def _write(tmp: Path, seeds: list[int]) -> list[str]:
    tmp.mkdir(parents=True, exist_ok=True)
    ids = []
    for s in seeds:
        sc = generate_scenario(s)
        (tmp / f"{sc.scenario_id}.json").write_text(sc.model_dump_json())
        ids.append(sc.scenario_id)
    return ids


def test_plan_clean_is_balanced_and_respects_first_seed(tmp_path: Path) -> None:
    _write(tmp_path, list(range(1000, 1030)))
    ids = plan_clean(tmp_path, 12, 1010)
    scs = [Scenario.model_validate_json((tmp_path / f"{i}.json").read_text()) for i in ids]
    assert len(set(ids)) == 12 and all(s.seed >= 1010 for s in scs)
    counts = {t: sum(s.incident_type == t for s in scs) for t in INCIDENT_TYPES}
    assert set(counts.values()) == {2}


def test_resume_skips_finished_runs_and_tolerates_a_torn_line(tmp_path: Path) -> None:
    ids = _write(tmp_path / "sc", [1002, 1003, 1004])
    out = tmp_path / "outcomes.jsonl"
    llm = RoutedLLM("bad_deploy", "rollback_deploy")
    first = run_clean_phase(ids[:2], llm, out, scen_dir=tmp_path / "sc", log=lambda m: None)
    assert [r.run_id for r in first] == [f"{i}-clean" for i in ids[:2]]
    with out.open("a") as fh:
        fh.write('{"run_id": "torn')  # crash mid-write
    n_calls = len(llm.calls)
    second = run_clean_phase(ids, llm, out, scen_dir=tmp_path / "sc", log=lambda m: None)
    assert [r.run_id for r in second] == [f"{ids[2]}-clean"]  # only the unfinished one ran
    assert len(llm.calls) == n_calls + 3  # three agent calls, nothing re-run
    assert finished_run_ids(out) == {f"{i}-clean" for i in ids}


def test_foreign_model_check_is_recorded_per_run(tmp_path: Path) -> None:
    ids = _write(tmp_path / "sc", [1002])
    out = tmp_path / "o.jsonl"
    recs = run_clean_phase(
        ids,
        RoutedLLM("bad_deploy", "rollback_deploy"),
        out,
        scen_dir=tmp_path / "sc",
        check_foreign=lambda: ["llama3.2:3b (18.0 GB, ctx 131072)"],
        log=lambda m: None,
    )
    assert recs[0].contaminated
    assert json.loads(out.read_text())["foreign_models"] == ["llama3.2:3b (18.0 GB, ctx 131072)"]
