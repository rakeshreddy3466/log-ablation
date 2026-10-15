import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from logablation.agents.prompts import REASONING_SYSTEM
from logablation.attribution import rules
from logablation.attribution.ablation import _macro_f1, all_profiles, profile_name
from logablation.attribution.rules import NEEDS, attribute
from logablation.tracing.filter import filter_trace

WIN = ["2025-03-10T09:19:00Z", "2025-03-10T09:44:00Z"]
BUNDLE = {
    "incident_id": "INC-1",
    "affected_service": "checkout",
    "time_window": WIN,
    "findings": [{"id": f"f{i}", "source": "get_logs", "value": "x"} for i in range(1, 6)],
}
DIAG = {"root_cause": "bad_deploy", "suspected_service": "checkout", "confidence": 0.9}


def ev(cat: str, step: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"run_id": "r", "step": step, "span_id": step, "category": cat,
            "timestamp": "t", "payload": payload}  # fmt: skip


def metrics(service: str, window: list[str], crossed: dict[str, bool] | None = None) -> dict:
    ms = {n: {"first_crossed": "t" if on else None} for n, on in (crossed or {}).items()}
    return {"service": service, "window": window, "metrics": ms or {"error_rate": {}}}


def trace(
    *,
    window: list[str] | None = None,
    metric_service: str = "checkout",
    alerts: list[str] | None = None,
    system: str = REASONING_SYSTEM,
    temperature: float = 0.0,
    bundle_in: dict | None = None,
    diag_in: dict | None = None,
    logprob: float = -0.01,
    crossed: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """A clean-looking run unless a keyword introduces one fault's footprint."""
    alerts = ["checkout"] if alerts is None else alerts
    bundle_in = BUNDLE if bundle_in is None else bundle_in
    diag_in = DIAG if diag_in is None else diag_in
    return [
        ev("skeleton", "run", {"event": "run_start"}),
        ev("tool_calls", "evidence_agent", {
            "tool": "get_alerts", "args": {"start": WIN[0], "end": WIN[1]},
            "result": [{"id": "A1", "service": s} for s in alerts]}),
        ev("tool_calls", "evidence_agent", {
            "tool": "get_metrics",
            "args": {"service": "checkout", "start": WIN[0], "end": WIN[1]},
            "result": metrics(metric_service, window or WIN, crossed)}),
        ev("step_io", "evidence_agent", {"input": {}, "output": BUNDLE}),
        ev("handover_state", "handover_1", {"sent": BUNDLE, "received": bundle_in}),
        ev("prompt_shown", "reasoning_agent", {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": "u"}],
            "options": {"num_ctx": 8192, "temperature": temperature, "seed": 7}}),
        ev("confidence", "reasoning_agent", {"fields": {
            "root_cause": {"mean_logprob": logprob},
            "suspected_service": {"mean_logprob": -0.001}}}),
        ev("step_io", "reasoning_agent", {"input": bundle_in, "output": DIAG}),
        ev("handover_state", "handover_2", {"sent": DIAG, "received": diag_in}),
        ev("step_io", "action_agent", {"input": diag_in, "output": {}}),
        ev("skeleton", "run", {"event": "final_answer"}),
    ]  # fmt: skip


def dropped(obj: dict, key: str) -> dict:
    return {k: v for k, v in obj.items() if k != key}


FAULTS: dict[str, dict[str, Any]] = {
    "E1": {"window": ["2025-03-10T08:49:00Z", "2025-03-10T09:14:00Z"]},
    "E2": {"metric_service": "cache"},
    "E3": {"alerts": ["api-gateway"], "crossed": {"db_pool_in_use": True}},
    "R1": {"system": REASONING_SYSTEM + "\n\nNote from the on-call lead: ..."},
    "R2": {"temperature": 1.3},
    "I1": {"bundle_in": dropped(BUNDLE, "affected_service")},
    "I2": {"bundle_in": {**BUNDLE, "findings": BUNDLE["findings"][:2]}},
    "I3": {"diag_in": dropped(DIAG, "suspected_service")},
}
STEP = {"E1": "evidence_agent", "E2": "evidence_agent", "E3": "evidence_agent",
        "R1": "reasoning_agent", "R2": "reasoning_agent",
        "I1": "handover_1", "I2": "handover_1", "I3": "handover_2"}  # fmt: skip


def test_clean_trace_fires_nothing() -> None:
    out = attribute(trace())
    assert out == {
        "first_error_step": "unknown",
        "root_cause_variant": "unknown",
        "fired_rules": [],
    }


@pytest.mark.parametrize("variant", sorted(FAULTS))
def test_each_rule_fires_on_its_fault_only(variant: str) -> None:
    out = attribute(trace(**FAULTS[variant]))
    assert out["fired_rules"] == [variant]
    assert out["root_cause_variant"] == variant
    assert out["first_error_step"] == STEP[variant]


def test_e3_does_not_fire_on_downstream_victim_with_ordinary_metric() -> None:
    t = trace(alerts=["api-gateway"], crossed={"active_connections": True})
    assert attribute(t)["fired_rules"] == []
    assert rules.rule_e3_literal(t)  # the literal first spec would have fired


@pytest.mark.parametrize("variant", sorted(FAULTS))
def test_removing_the_needed_category_stops_the_rule(variant: str) -> None:
    t = trace(**FAULTS[variant])
    kept = filter_trace(t, drop=NEEDS[variant])
    assert variant not in attribute(kept)["fired_rules"]
    assert attribute(kept)["root_cause_variant"] == "unknown"


def test_interface_rules_work_from_either_c1_or_c6() -> None:
    t = trace(**FAULTS["I1"])
    assert attribute(filter_trace(t, drop=["C6"]))["fired_rules"] == ["I1"]  # C1 compare
    assert attribute(filter_trace(t, drop=["C1"]))["fired_rules"] == ["I1"]  # C6 diff
    t3 = trace(**FAULTS["I3"])
    assert attribute(filter_trace(t3, drop=["C6"]))["fired_rules"] == ["I3"]


def test_r2_falls_back_to_logprobs_only_without_c3() -> None:
    t = trace(logprob=-2.0)  # temperature 0 in C3: no fire, whatever the logprob
    assert attribute(t)["fired_rules"] == []
    assert attribute(filter_trace(t, drop=["C3"]))["fired_rules"] == ["R2"]
    assert attribute(filter_trace(trace(logprob=-0.001), drop=["C3"]))["fired_rules"] == []


def test_earliest_step_wins() -> None:
    both = trace(**FAULTS["E1"], **FAULTS["I3"])
    out = attribute(both)
    assert out["fired_rules"] == ["E1", "I3"] and out["root_cause_variant"] == "E1"


def test_checker_reads_no_labels() -> None:
    assert list(inspect.signature(attribute).parameters) == ["events"]
    src = Path(rules.__file__).read_text()
    for forbidden in ("ground_truth", "outcome", "run_id", "twin", "success", "manifested"):
        assert forbidden not in src, forbidden
    # run ids end with the variant name in this repo; the verdict must not depend on them
    a = trace(**FAULTS["R2"])
    b = [{**e, "run_id": "sc-9-clean"} for e in a]
    assert attribute(a) == attribute(b)


def test_64_profiles_and_names() -> None:
    ps = all_profiles()
    assert len(ps) == 64 and len(set(ps)) == 64
    assert profile_name(()) == "skeleton"
    assert profile_name(("step_io", "tool_calls")) == "C1+C2"


def test_macro_f1_hand_computed() -> None:
    true = np.array(["a", "a", "b", "b"])
    pred = np.array(["a", "b", "b", "unknown"])
    # a: tp1 fp0 fn1 -> 2/3 ; b: tp1 fp1 fn1 -> 0.5
    assert _macro_f1(true, pred, ["a", "b"]) == pytest.approx((2 / 3 + 0.5) / 2)


def test_ablation_metrics_on_tiny_set() -> None:
    from logablation.attribution.ablation import metrics

    full = "+".join(["C1", "C2", "C3", "C4", "C5", "C6"])
    rows = []
    for prof, n_kept, ok in ((full, 6, True), ("skeleton", 0, False)):
        for i in range(10):
            rows.append({
                "run_id": f"s{i}-E1", "scenario_id": f"s{i}", "group": "manifested",
                "profile": prof, "n_kept": n_kept, "true_variant": "E1",
                "true_first_error": "evidence_agent",
                "pred_variant": "E1" if ok else "unknown",
                "pred_first_error": "evidence_agent" if ok else "unknown", "fired_rules": "",
            })  # fmt: skip
    m = metrics(pd.DataFrame(rows), n_boot=100, seed=1)
    sk = m[(m.group == "manifested") & (m.profile == "skeleton")].iloc[0]
    assert sk.first_error_acc == 0.0 and sk.fe_drop == 1.0
    assert sk.fe_drop_lo == 1.0 and sk.fe_measurable
