"""Scenario generator tests: determinism, coverage, signal-to-cause consistency, planted fields.

``diagnose`` below is a deliberately simple, independent reader of the *evidence only*
(metrics + deploy log); it never looks at ground truth. If each scenario's signals really point
to its cause, exactly one signature rule fires and it names the right service.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from logablation.scenarios import generate_many, generate_scenario
from logablation.scenarios.models import Scenario
from logablation.scenarios.taxonomy import (
    ACTIONS,
    INCIDENT_TYPES,
    ROOT_CAUSE_SERVICES,
    SENSITIVE_TYPES,
    SERVICES,
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def diagnose(sc: Scenario) -> list[tuple[str, str]]:
    """Return every (incident_type, service) signature that fires, using evidence only."""
    s = sc.metrics.series
    hits: list[tuple[str, str]] = []
    limits = sc.topology.limits

    for svc in SERVICES:
        if min(s[svc]["cert_seconds_to_expiry"]) < 0:
            hits.append(("expired_certificate", svc))
        mem = s[svc]["memory_mb"]
        if mem[-1] / mem[0] > 1.8:
            hits.append(("memory_leak", svc))
    if max(s["cache"]["evictions_per_s"]) > 20 * _mean(s["cache"]["evictions_per_s"][:20]):
        hits.append(("cache_eviction_storm", "cache"))
    if max(s["checkout"]["db_pool_in_use"]) >= 0.95 * limits["checkout.db_pool_max"]:
        hits.append(("db_pool_exhausted", "checkout"))
    rpc = s["checkout"]["payment_db_rpc_latency_ms"]
    if _mean(rpc[-10:]) > 3 * _mean(rpc[:20]):
        hits.append(("cross_zone_latency", "checkout"))
    for dep in sc.deploys:
        if dep.kind != "deploy":
            continue
        err = s[dep.service]["error_rate"]
        step = sc.metrics.timestamps.index(dep.time) + 1
        before = _mean(err[max(0, step - 5) : step])
        after = _mean(err[step : step + 5])
        if after > 5 * before and after - before > 0.03:
            hits.append(("bad_deploy", dep.service))
    return hits


@pytest.fixture(scope="module")
def scenarios() -> list[Scenario]:
    return generate_many(1000, 120)  # 20 per incident type


def test_same_seed_gives_identical_scenarios() -> None:
    for seed in (1000, 1001, 1002, 1003, 1004, 1005, 31337):
        assert (
            generate_scenario(seed).model_dump_json() == generate_scenario(seed).model_dump_json()
        )


def test_different_seeds_differ() -> None:
    assert generate_scenario(1000).model_dump_json() != generate_scenario(1006).model_dump_json()


def test_json_round_trip_is_stable(scenarios: list[Scenario]) -> None:
    sc = scenarios[0]
    assert Scenario.model_validate(json.loads(sc.model_dump_json())) == sc


def test_every_incident_type_is_generated_and_balanced(scenarios: list[Scenario]) -> None:
    counts = Counter(sc.incident_type for sc in scenarios)
    assert set(counts) == set(INCIDENT_TYPES)
    assert len(set(counts.values())) == 1


def test_explicit_incident_type_is_respected() -> None:
    for t in INCIDENT_TYPES:
        sc = generate_scenario(7, t)
        assert sc.incident_type == t == sc.ground_truth.root_cause
        assert sc.ground_truth.correct_action == ACTIONS[t]
    with pytest.raises(ValueError):
        generate_scenario(7, "solar_flare")


def test_ground_truth_and_structure(scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        gt = sc.ground_truth
        assert gt.root_cause_service in ROOT_CAUSE_SERVICES[sc.incident_type]
        assert gt.decisive_alert_id in {a.id for a in sc.alerts}
        assert set(sc.topology.services) == set(SERVICES)
        assert sc.alerts and sc.deploys and sc.logs
        n = len(sc.metrics.timestamps)
        for metrics in sc.metrics.series.values():
            assert all(len(v) == n for v in metrics.values())
        for items in (sc.alerts, sc.deploys, sc.logs):
            times = [i.time for i in items]
            assert times == sorted(times)


def test_signals_point_to_true_root_cause(scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        gt = sc.ground_truth
        assert diagnose(sc) == [(gt.root_cause, gt.root_cause_service)], sc.scenario_id


def test_logs_and_alerts_mention_the_cause(scenarios: list[Scenario]) -> None:
    keywords = {
        "db_pool_exhausted": "pool",
        "bad_deploy": "unhandled exception",
        "memory_leak": "heap usage",
        "cross_zone_latency": "zone",
        "expired_certificate": "expired",
        "cache_eviction_storm": "evicted",
    }
    for sc in scenarios:
        kw = keywords[sc.incident_type]
        assert any(kw in line.message for line in sc.logs), sc.scenario_id
        decisive = next(a for a in sc.alerts if a.id == sc.ground_truth.decisive_alert_id)
        assert decisive.service == sc.ground_truth.root_cause_service
        assert decisive.time >= sc.ground_truth.onset_time


def test_sensitive_fields_recorded_at_exact_positions(scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        assert sc.sensitive_fields
        ids = [f.id for f in sc.sensitive_fields]
        assert len(ids) == len(set(ids))
        for f in sc.sensitive_fields:
            text = getattr(getattr(sc, f.source)[f.index], f.field)
            assert text[f.start : f.end] == f.value
            assert f.type in SENSITIVE_TYPES


def test_every_sensitive_type_planted_in_every_scenario(scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        assert {f.type for f in sc.sensitive_fields} == set(SENSITIVE_TYPES)


def test_no_unrecorded_sensitive_values(scenarios: list[Scenario]) -> None:
    """Every api key / order id / email found by pattern must be in the planted record."""
    pattern = re.compile(r"sk_live_[a-z0-9]{24}|ORD-\d{8}|[\w.]+@[\w.]+\.test")
    for sc in scenarios:
        recorded = {f.value for f in sc.sensitive_fields}
        texts = [a.message for a in sc.alerts] + [d.author for d in sc.deploys]
        texts += [d.description for d in sc.deploys] + [x.message for x in sc.logs]
        for text in texts:
            for m in pattern.findall(text):
                assert m in recorded
