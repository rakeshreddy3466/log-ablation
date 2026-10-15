"""Fault tests (no Ollama): each fault changes exactly what it should and nothing else."""

from __future__ import annotations

import json
from typing import Any

import pytest

from logablation.agents.agents import gather_evidence
from logablation.faults import build_hooks, derive_seed, plan_faulted
from logablation.faults.injectors import (
    R2_TEMPERATURE,
    STALE_SHIFT_MIN,
    neighbour_of,
    wrong_incident_type,
)
from logablation.pipeline import PipelineHooks, field_diff, run_pipeline
from logablation.scenarios.generator import generate_scenario
from logablation.scenarios.models import Scenario
from logablation.scenarios.taxonomy import (
    FIRST_ERROR_STEP,
    INCIDENT_TYPES,
    VARIANTS,
)
from logablation.tools import ToolContext, build_ticket
from logablation.tools.timeutil import shift

from .test_pipeline import RoutedLLM, _right

SEEDS = [1000, 1001, 1002, 1003, 1004, 1005, 1100, 1101]  # all six incident types covered


def _calls(sc: Scenario, variant: str | None, run_seed: int = 5) -> list[dict[str, Any]]:
    hooks = build_hooks(variant, sc, run_seed)
    return gather_evidence(build_ticket(sc), ToolContext(sc, transform=hooks.tool_transform))


def _pair(sc: Scenario, variant: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    base, faulted = _calls(sc, None), _calls(sc, variant)
    assert [c["tool"] for c in base] == [c["tool"] for c in faulted]  # same plan, same args
    assert [c["args"] for c in base] == [c["args"] for c in faulted]
    return list(zip(base, faulted, strict=True))


def test_covers_all_incident_types() -> None:
    assert {generate_scenario(s).incident_type for s in SEEDS} == set(INCIDENT_TYPES)


@pytest.mark.parametrize("seed", SEEDS)
def test_e1_only_get_metrics_changes_and_shows_the_shifted_window(seed: int) -> None:
    sc = generate_scenario(seed)
    changed = 0
    for base, f in _pair(sc, "E1"):
        if base["tool"] != "get_metrics":
            assert f["result"] == base["result"]
            continue
        start, end = base["args"]["start"], base["args"]["end"]
        assert f["result"]["window"] == [shift(start, STALE_SHIFT_MIN), shift(end, STALE_SHIFT_MIN)]
        assert f["result"]["service"] == base["result"]["service"]
        changed += f["result"] != base["result"]
    assert changed > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_e2_swaps_only_root_cause_service_metrics_and_logs(seed: int) -> None:
    sc = generate_scenario(seed)
    root = sc.ground_truth.root_cause_service
    other = neighbour_of(sc, root, 5)
    assert other != root
    assert (root, other) in sc.topology.dependencies or (other, root) in sc.topology.dependencies
    hit = 0
    for base, f in _pair(sc, "E2"):
        if base["tool"] in ("get_metrics", "get_logs") and base["args"]["service"] == root:
            assert f["result"]["service"] == other  # the neighbour's real name is visible
            assert f["result"] != base["result"]
            hit += 1
        else:
            assert f["result"] == base["result"]
    assert hit == 2  # one get_metrics and one get_logs for the root-cause service


@pytest.mark.parametrize("seed", SEEDS)
def test_e3_removes_exactly_the_decisive_alert(seed: int) -> None:
    sc = generate_scenario(seed)
    dec = sc.ground_truth.decisive_alert_id
    for base, f in _pair(sc, "E3"):
        if base["tool"] != "get_alerts":
            assert f["result"] == base["result"]
            continue
        # exactly the decisive alert is removed (when it is in the window; else nothing changes)
        assert [a for a in base["result"] if a["id"] != dec] == f["result"]


def test_r1_changes_only_the_reasoning_system_prompt_with_a_wrong_type() -> None:
    sc = generate_scenario(1002)
    h = build_hooks("R1", sc, 5)
    system, user = h.reasoning_prompt("SYS", "USER")
    assert user == "USER" and system.startswith("SYS") and len(system) > len("SYS")
    wrong = wrong_incident_type(sc.ground_truth.root_cause, 5)
    assert wrong != sc.ground_truth.root_cause and wrong in system
    assert sc.ground_truth.root_cause not in system
    assert h.reasoning_options(0.0, 7) == (0.0, 7)  # sampling untouched


def test_r2_changes_only_reasoning_sampling_and_is_seed_derived() -> None:
    sc = generate_scenario(1002)
    h = build_hooks("R2", sc, 5)
    assert h.reasoning_prompt("S", "U") == ("S", "U")
    t, s = h.reasoning_options(0.0, 7)
    assert t == R2_TEMPERATURE == 1.3 and s == build_hooks("R2", sc, 5).reasoning_options(0.0, 7)[1]
    assert s != build_hooks("R2", sc, 6).reasoning_options(0.0, 7)[1]


BUNDLE = {
    "incident_id": "INC-1",
    "affected_service": "checkout",
    "time_window": ["a", "b"],
    "findings": [{"id": f"f{i}", "source": "get_logs", "value": f"v{i}"} for i in range(1, 5)],
    "evidence_confidence": 0.8,
}
DIAG = {
    "root_cause": "bad_deploy",
    "suspected_service": "checkout",
    "reasoning_summary": "x",
    "confidence": 0.5,
    "evidence_used": ["f1"],
}


def _diff(hook: Any, obj: dict[str, Any]) -> set[tuple[str, str]]:
    import copy

    return {(d["path"], d["op"]) for d in field_diff(obj, hook(copy.deepcopy(obj)))}


def test_i1_i2_i3_diffs_are_exactly_the_documented_change() -> None:
    sc = generate_scenario(1002)
    assert _diff(build_hooks("I1", sc, 5).handover_1, BUNDLE) == {("affected_service", "removed")}
    assert _diff(build_hooks("I2", sc, 5).handover_1, BUNDLE) == {
        ("findings[2]", "removed"),
        ("findings[3]", "removed"),
    }
    assert _diff(build_hooks("I3", sc, 5).handover_2, DIAG) == {("suspected_service", "removed")}
    # the other handover is untouched
    assert _diff(build_hooks("I3", sc, 5).handover_1, BUNDLE) == set()
    assert _diff(build_hooks("I1", sc, 5).handover_2, DIAG) == set()


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_fault_sets_only_its_own_hooks(variant: str) -> None:
    sc = generate_scenario(1002)
    default = PipelineHooks()
    h = build_hooks(variant, sc, 5)
    own = {
        "E1": {"tool_transform"},
        "E2": {"tool_transform"},
        "E3": {"tool_transform"},
        "R1": {"reasoning_prompt"},
        "R2": {"reasoning_options"},
        "I1": {"handover_1"},
        "I2": {"handover_1"},
        "I3": {"handover_2"},
    }[variant]
    changed = {f for f in vars(default) if getattr(h, f) is not getattr(default, f)}
    assert changed == own
    assert vars(build_hooks(None, sc, 5)) == vars(default)


def test_i3_end_to_end_the_action_agent_never_sees_the_suspected_service() -> None:
    sc = generate_scenario(1002)
    llm = _right(sc)
    rec = run_pipeline(
        sc,
        llm,
        run_id="i3",
        variant="I3",
        twin_run_id="x",
        twin_success=True,
        hooks=build_hooks("I3", sc, 5),
    )
    action_user = next(u for t, _, u, _, _ in llm.calls if t == "ActionPlan")
    assert "suspected_service" not in action_user
    assert [d["path"] for d in rec.handovers[1].diff] == ["suspected_service"]
    assert not rec.handovers[1].schema_ok and rec.handovers[0].diff == []


def test_faulted_run_ground_truth_labels() -> None:
    sc = generate_scenario(1002)
    for v in VARIANTS:
        # a wrong answer under any fault is manifested and labelled with that fault's variant/step
        wrong = RoutedLLM("memory_leak" if sc.incident_type != "memory_leak" else "bad_deploy", "x")
        rec = run_pipeline(
            sc,
            wrong,
            run_id=f"r-{v}",
            variant=v,
            twin_run_id="t",
            twin_success=True,
            hooks=build_hooks(v, sc, 5),
        )
        assert rec.outcome == "manifested"
        assert rec.root_cause_variant == v and rec.first_error_step == FIRST_ERROR_STEP[v]
    assert FIRST_ERROR_STEP == {
        **dict.fromkeys(("E1", "E2", "E3"), "evidence_agent"),
        **dict.fromkeys(("R1", "R2"), "reasoning_agent"),
        **dict.fromkeys(("I1", "I2"), "handover_1"),
        "I3": "handover_2",
    }
    ok = run_pipeline(
        sc,
        _right(sc),
        run_id="ok",
        variant="E3",
        twin_run_id="t",
        twin_success=True,
        hooks=build_hooks("E3", sc, 5),
    )
    assert ok.outcome == "masked" and ok.first_error_step is None and ok.root_cause_variant is None


def test_derive_seed_is_stable_and_key_dependent() -> None:
    assert derive_seed(11, "sc-1-E1") == derive_seed(11, "sc-1-E1")
    assert derive_seed(11, "sc-1-E1") != derive_seed(11, "sc-1-E2")
    assert derive_seed(11, "a") != derive_seed(12, "a")
    assert 0 <= derive_seed(11, "a") < 2**31


# ---- planner ---------------------------------------------------------------------------------


def _clean(n: int, fail_every: int = 5) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        sid = f"sc-{1100 + i:05d}"
        out.append(
            {
                "run_id": f"{sid}-clean",
                "scenario_id": sid,
                "incident_type": INCIDENT_TYPES[i % 6],
                "variant": None,
                "success": i % fail_every != 0,
            }
        )
    return out


def test_planner_uses_only_eligible_scenarios_and_balances_variants() -> None:
    clean = _clean(60)
    plan = plan_faulted(clean, per_variant=30, seed=11)
    ok = {r["scenario_id"] for r in clean if r["success"]}
    assert len(ok) == 48
    assert {p.scenario_id for p in plan.runs} <= ok  # never an ineligible scenario
    counts = {v: sum(p.variant == v for p in plan.runs) for v in VARIANTS}
    assert set(counts.values()) == {30} and plan.shortfall == {}
    keys = [(p.scenario_id, p.variant) for p in plan.runs]
    assert len(keys) == len(set(keys))  # no scenario gets the same variant twice
    assert len({p.run_id for p in plan.runs}) == 240
    twin = {r["scenario_id"]: r["run_id"] for r in clean}
    assert all(p.twin_run_id == twin[p.scenario_id] for p in plan.runs)
    load = [sum(p.scenario_id == s for p in plan.runs) for s in ok]
    assert max(load) - min(load) <= 1  # scenarios share the 240 runs evenly (5 each)


def test_planner_is_deterministic_and_seed_sensitive() -> None:
    clean = _clean(60)
    a = plan_faulted(clean, per_variant=30, seed=11)
    assert a.runs == plan_faulted(clean, per_variant=30, seed=11).runs
    assert a.runs != plan_faulted(clean, per_variant=30, seed=12).runs


def test_planner_reports_shortfall_and_eligibility() -> None:
    clean = _clean(12)  # i = 0, 5, 10 fail -> 9 eligible
    plan = plan_faulted(clean, per_variant=30, seed=11)
    assert plan.shortfall == dict.fromkeys(VARIANTS, 21)  # no repeats -> at most 9 per variant
    assert len(plan.runs) == 72
    assert sum(ok for ok, _ in plan.eligibility.values()) == 9
    assert sum(n for _, n in plan.eligibility.values()) == 12
    assert plan_faulted([], per_variant=3, seed=1).runs == []


def test_planner_ignores_faulted_records_in_the_input() -> None:
    clean = _clean(6) + [
        {
            "run_id": "sc-01101-E1",
            "scenario_id": "sc-01101",
            "incident_type": "x",
            "variant": "E1",
            "success": True,
        }
    ]
    plan = plan_faulted(clean, per_variant=2, seed=1)
    assert all(p.twin_run_id.endswith("-clean") for p in plan.runs)
    assert json.dumps(sum(n for _, n in plan.eligibility.values())) == "6"


def test_e3_is_a_noop_exactly_when_the_decisive_alert_is_outside_the_window() -> None:
    from logablation.faults import fault_applies

    sc = generate_scenario(1004)  # memory_leak: decisive alert A3 not in the ticket window
    assert not fault_applies("E3", sc, 5)
    assert fault_applies("R1", sc, 5) and fault_applies("I2", sc, 5)
    assert fault_applies("E3", generate_scenario(1002), 5)


def test_planner_never_assigns_an_inapplicable_pair() -> None:
    clean = _clean(30)
    banned = {(r["scenario_id"], "E3") for r in clean[::2]}
    plan = plan_faulted(clean, per_variant=10, seed=3, applicable=lambda s, v: (s, v) not in banned)
    assert not {(p.scenario_id, p.variant) for p in plan.runs} & banned
    assert sum(p.variant == "E3" for p in plan.runs) == 10
