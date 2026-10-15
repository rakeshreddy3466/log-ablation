"""Tools-layer tests: ticket hygiene, time filtering, metric signatures, determinism, budget."""

from __future__ import annotations

import json
from functools import lru_cache

import pytest

from logablation.scenarios import generate_many, generate_scenario
from logablation.scenarios.models import Scenario
from logablation.tools import ToolContext, build_ticket, estimate_tokens
from logablation.tools.mock_apis import TOOL_NAMES
from logablation.tools.timeutil import parse, shift

BASE_SEED = 1000
TOKEN_BUDGET = 3000

# Signature metric(s) that must cross inside the window at the root-cause service.
SIGNATURE = {
    "db_pool_exhausted": ["db_pool_in_use", "latency_p95_ms"],
    "bad_deploy": ["error_rate"],
    "memory_leak": ["memory_mb"],
    "cross_zone_latency": ["payment_db_rpc_latency_ms"],
    "expired_certificate": ["cert_seconds_to_expiry", "tls_handshake_failures"],
    "cache_eviction_storm": ["evictions_per_s", "hit_ratio"],
}


@lru_cache(maxsize=1)
def _all() -> tuple[Scenario, ...]:
    return tuple(generate_many(BASE_SEED, 300))


@pytest.fixture(scope="module")
def scenarios() -> tuple[Scenario, ...]:
    return _all()


def _one_per_type(scs: tuple[Scenario, ...]) -> list[Scenario]:
    seen: dict[str, Scenario] = {}
    for s in scs:
        seen.setdefault(s.incident_type, s)
    return list(seen.values())


def _calls(ctx: ToolContext, services: list[str], a: str, b: str) -> list:
    out = []
    for svc in services:
        out += [ctx.get_alerts(a, b, svc), ctx.get_metrics(svc, a, b), ctx.get_logs(svc, a, b)]
    out += [ctx.get_deploys(a, b), ctx.get_topology()]
    return out


# ---- ticket ---------------------------------------------------------------------------------


def test_ticket_fields_and_window(scenarios):
    for s in scenarios:
        t = build_ticket(s)
        assert t.incident_id == "INC-" + s.scenario_id.split("-")[1]
        assert t.symptom_service == s.ground_truth.symptom_service
        assert t.time_window == (shift(t.reported_at, -20), shift(t.reported_at, 5))
        # reported_at is the time of the first non-info alert on the symptom service
        sym = [a for a in s.alerts if a.service == t.symptom_service and a.severity != "info"]
        assert t.reported_at == min(sym, key=lambda a: parse(a.time)).time
        assert t.summary.count(".") == 1 and t.summary.endswith(".")


def test_ticket_never_contains_ground_truth(scenarios):
    for s in scenarios:
        gt = s.ground_truth
        blob = build_ticket(s).model_dump_json().lower()
        for secret in (gt.incident_type, gt.root_cause, gt.correct_action, gt.onset_time):
            assert secret.lower() not in blob
        if gt.root_cause_service != gt.symptom_service:
            assert gt.root_cause_service not in blob
        assert gt.decisive_alert_id.lower() not in blob.replace("incident", "")  # 'A3' style id


# ---- no ground truth through tools ------------------------------------------------------------


def test_tools_never_expose_ground_truth(scenarios):
    for s in scenarios[:60]:
        gt = s.ground_truth
        ctx = ToolContext(s)
        a, b = s.time_window
        services = list(s.topology.services)
        blob = json.dumps(_calls(ctx, services, a, b)).lower()
        for secret in (
            gt.incident_type,
            gt.correct_action,
            gt.root_cause,
            "ground_truth",
            "onset",
            s.scenario_id,
            "root_cause",
            "decisive",
        ):
            assert secret.lower() not in blob, secret


# ---- time filtering ---------------------------------------------------------------------------


def test_alert_deploy_log_time_filtering():
    s = generate_scenario(BASE_SEED + 2)  # bad_deploy
    ctx = ToolContext(s)
    a, b = "2025-03-10T09:20:00Z", "2025-03-10T09:40:00Z"
    for r in ctx.get_alerts(a, b):
        assert a <= r["time"] <= b
    assert {r["id"] for r in ctx.get_alerts(a, b)} == {x.id for x in s.alerts if a <= x.time <= b}
    assert {r["id"] for r in ctx.get_deploys(a, b)} == {x.id for x in s.deploys if a <= x.time <= b}
    for svc in s.topology.services:
        res = ctx.get_logs(svc, a, b, limit=1000)
        assert all(a <= ln["time"] <= b for ln in res["lines"])
        assert (
            res["total_matching"]
            == len(res["lines"])
            == sum(1 for ln in s.logs if ln.service == svc and a <= ln.time <= b)
        )
    # window bounds are inclusive and alert service filter works
    alert = s.alerts[0]
    assert ctx.get_alerts(alert.time, alert.time, alert.service)[0]["id"] == alert.id
    assert ctx.get_alerts(shift(alert.time, 1), shift(alert.time, 1), alert.service) == []


def test_logs_level_limit_and_errors():
    s = generate_scenario(BASE_SEED)
    ctx = ToolContext(s)
    a, b = s.time_window
    svc = "api-gateway"
    all_lines = ctx.get_logs(svc, a, b, limit=1000)
    warns = ctx.get_logs(svc, a, b, level="warning", limit=1000)
    assert all(ln["level"] == "WARN" for ln in warns["lines"])
    short = ctx.get_logs(svc, a, b, limit=2)
    assert len(short["lines"]) == 2 and short["truncated"] is True
    assert short["total_matching"] == all_lines["total_matching"]
    assert "error" in ctx.get_logs("nope", a, b)
    assert "error" in ctx.get_logs(svc, a, b, level="LOUD")
    assert "error" in ctx.get_metrics("nope", a, b)
    assert "error" in ctx.get_metrics(svc, b, a)


def test_metrics_only_use_window_points():
    s = generate_scenario(BASE_SEED + 4)  # memory_leak
    ctx = ToolContext(s)
    a, b = "2025-03-10T09:30:00Z", "2025-03-10T09:34:00Z"
    m = ctx.get_metrics(s.ground_truth.root_cause_service, a, b)["metrics"]["memory_mb"]
    ts = s.metrics.timestamps
    vals = s.metrics.series[s.ground_truth.root_cause_service]["memory_mb"]
    inside = [v for t, v in zip(ts, vals, strict=True) if a <= t <= b]
    assert len(inside) == 5
    assert m["max"] == pytest.approx(max(inside), rel=1e-3)
    assert m["min"] == pytest.approx(min(inside), rel=1e-3)
    assert m["mean"] == pytest.approx(sum(inside) / 5, rel=1e-3)
    assert ctx.get_metrics("cache", "2030-01-01T00:00:00Z", "2030-01-01T01:00:00Z")["metrics"] == {}


# ---- incident signature -----------------------------------------------------------------------


def test_root_service_shows_signature_inside_window(scenarios):
    for s in scenarios:
        t = build_ticket(s)
        ctx = ToolContext(s)
        root = s.ground_truth.root_cause_service
        metrics = ctx.get_metrics(root, *t.time_window)["metrics"]
        crossed = [n for n in SIGNATURE[s.incident_type] if metrics[n]["first_crossed"]]
        assert crossed, (s.scenario_id, s.incident_type, metrics)
        onset = parse(s.ground_truth.onset_time)
        for n in crossed:
            # memory_mb in a leak climbs slowly across the whole hour, so it may cross early
            if (s.incident_type, n) != ("memory_leak", "memory_mb"):
                assert parse(metrics[n]["first_crossed"]) >= onset


def test_root_service_healthy_before_onset(scenarios):
    """Window ending before onset: signature metrics never cross, and false alarms are rare.

    memory_mb in a leak is exempt (it climbs slowly all hour) and cert_seconds_to_expiry is a
    countdown. error_rate sits at ~0.2% with noise ~35% of that, so a non-signature metric can
    occasionally show two noisy minutes in a row; we bound that rate instead of pretending 0.
    """
    windows = false_alarms = 0
    for s in scenarios:
        onset = s.ground_truth.onset_time
        a, b = shift(onset, -25), shift(onset, -1)
        for svc in {s.ground_truth.root_cause_service, s.ground_truth.symptom_service}:
            metrics = ToolContext(s).get_metrics(svc, a, b)["metrics"]
            assert metrics
            for name, m in metrics.items():
                if (s.incident_type, name) == ("memory_leak", "memory_mb"):
                    continue
                if svc == s.ground_truth.root_cause_service and name in SIGNATURE[s.incident_type]:
                    assert m["first_crossed"] is None, (s.scenario_id, svc, name, m)
                windows += 1
                false_alarms += m["first_crossed"] is not None
                if name != "cert_seconds_to_expiry" and "baseline_from" not in m:
                    assert 0.5 < (m["change_factor"] or 1.0) < 1.5, (s.scenario_id, svc, name, m)
    assert false_alarms / windows < 0.002, (false_alarms, windows)


# ---- determinism and hook ---------------------------------------------------------------------


def test_outputs_are_deterministic():
    a_s, b_s = generate_scenario(BASE_SEED + 3), generate_scenario(BASE_SEED + 3)
    t1, t2 = build_ticket(a_s), build_ticket(b_s)
    assert t1 == t2
    svcs = list(a_s.topology.services)
    r1 = json.dumps(_calls(ToolContext(a_s), svcs, *t1.time_window))
    r2 = json.dumps(_calls(ToolContext(b_s), svcs, *t2.time_window))
    assert r1 == r2
    assert json.dumps(_calls(ToolContext(a_s), svcs, *t1.time_window)) == r1  # repeat call


def test_transform_hook_is_noop_by_default_and_receives_calls():
    s = generate_scenario(BASE_SEED)
    a, b = build_ticket(s).time_window
    seen: list[tuple[str, dict]] = []

    def spy(name, args, result):
        seen.append((name, args))
        return result

    plain, hooked = ToolContext(s), ToolContext(s, transform=spy)
    assert plain.get_logs("auth", a, b) == hooked.get_logs("auth", a, b)
    assert seen == [
        ("get_logs", {"level": None, "limit": 20, "service": "auth", "start": a, "end": b})
    ]

    def blank(name, args, result):
        return {"replaced": name}

    assert ToolContext(s, transform=blank).get_topology() == {"replaced": "get_topology"}
    assert set(TOOL_NAMES) == {
        "get_alerts",
        "get_metrics",
        "get_deploys",
        "get_logs",
        "get_topology",
    }
    with pytest.raises(KeyError):
        plain.call("get_everything")


# ---- token budget -----------------------------------------------------------------------------


def test_token_budget_all_five_tools_root_and_symptom(scenarios):
    worst = 0
    for s in scenarios:
        t = build_ticket(s)
        services = sorted({s.ground_truth.root_cause_service, t.symptom_service})
        ctx = ToolContext(s)
        for window in (t.time_window, s.time_window):  # ticket window and the whole hour
            tokens = estimate_tokens(_calls(ctx, services, *window))
            worst = max(worst, tokens)
            assert tokens < TOKEN_BUDGET, (s.scenario_id, window, tokens)
    print(f"worst-case tokens over 300 scenarios: {worst}")
