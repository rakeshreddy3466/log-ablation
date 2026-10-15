"""Pure mock APIs over one scenario. Each returns a SHORT, JSON-serialisable summary.

These functions know nothing about faults or logging; ``ToolContext`` wraps them. Faults may
call them again with altered arguments (e.g. another service or window) to build a corrupted
result without touching this code.

Metric summary per metric over the requested window:
  baseline       mean of up to 20 minutes before the window start (if fewer than 5 such points
                 exist, the first 5 minutes of the window are used and ``baseline_from`` says so)
  mean, min, max inside the window
  change_factor  mean / baseline (null when the baseline is ~0)
  first_crossed  first minute of a run of 2 consecutive minutes outside the baseline band
                 (one noisy point is not a crossing), else null.
                 Band: |x - baseline| > max(k * baseline_sd, 25% of |baseline|, 1e-6), k = 5
                 (8 when the baseline has fewer than 10 points).
                 cert_seconds_to_expiry is the exception: it crosses when it goes below 0.
"""

from __future__ import annotations

import statistics
from typing import Any

from ..scenarios.models import Scenario
from .timeutil import parse

BASELINE_MINUTES = 20
MIN_BASELINE_POINTS = 5
LOG_LEVELS = ("ERROR", "WARN", "INFO")
_LEVEL_ALIAS = {"WARNING": "WARN"}

TOOL_NAMES = ("get_alerts", "get_metrics", "get_deploys", "get_logs", "get_topology")


def _sig(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def _bad_window(start: str, end: str) -> dict[str, str] | None:
    try:
        if parse(start) > parse(end):
            return {"error": "start must not be after end"}
    except (ValueError, TypeError):
        return {"error": "start and end must be ISO-8601 timestamps like 2025-03-10T09:00:00Z"}
    return None


def _in_window(ts: str, start: str, end: str) -> bool:
    return parse(start) <= parse(ts) <= parse(end)


def get_alerts(
    scenario: Scenario, service: str | None, start: str, end: str
) -> dict[str, Any] | list[dict[str, Any]]:
    if (err := _bad_window(start, end)) is not None:
        return err
    return [
        a.model_dump()
        for a in scenario.alerts
        if (service is None or a.service == service) and _in_window(a.time, start, end)
    ]


def get_deploys(scenario: Scenario, start: str, end: str) -> dict[str, Any] | list[dict[str, Any]]:
    if (err := _bad_window(start, end)) is not None:
        return err
    return [d.model_dump() for d in scenario.deploys if _in_window(d.time, start, end)]


def get_logs(
    scenario: Scenario,
    service: str,
    start: str,
    end: str,
    level: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Matching lines, at most ``limit``: most severe, then most recent; shown in time order."""
    if (err := _bad_window(start, end)) is not None:
        return err
    if service not in scenario.topology.services:
        return {"error": f"unknown service '{service}'"}
    want = None
    if level is not None:
        want = _LEVEL_ALIAS.get(level.upper(), level.upper())
        if want not in LOG_LEVELS:
            return {"error": f"unknown level '{level}'; use one of {list(LOG_LEVELS)}"}
    rows = [
        (i, ln)
        for i, ln in enumerate(scenario.logs)
        if ln.service == service
        and (want is None or ln.level == want)
        and _in_window(ln.time, start, end)
    ]
    rank = {lvl: n for n, lvl in enumerate(LOG_LEVELS)}
    chosen = sorted(
        rows, key=lambda r: (rank.get(r[1].level, 9), -parse(r[1].time).timestamp(), r[0])
    )
    chosen = sorted(chosen[: max(limit, 0)], key=lambda r: (parse(r[1].time), r[0]))
    return {
        "service": service,
        "total_matching": len(rows),
        "truncated": len(rows) > len(chosen),
        "lines": [{"time": ln.time, "level": ln.level, "message": ln.message} for _, ln in chosen],
    }


def get_topology(scenario: Scenario) -> dict[str, Any]:
    t = scenario.topology
    return {
        "services": list(t.services),
        "dependencies": [f"{a} -> {b}" for a, b in t.dependencies],
        "zones": dict(t.zones),
        "limits": dict(t.limits),
    }


def _summarise_metric(
    name: str, values: list[float], times: list[str], start: str, end: str
) -> dict[str, Any] | None:
    s, e = parse(start), parse(end)
    idx_in = [i for i, t in enumerate(times) if s <= parse(t) <= e]
    if not idx_in:
        return None
    first = idx_in[0]
    inside = [values[i] for i in idx_in]
    pre = values[max(0, first - BASELINE_MINUTES) : first]
    note = None
    if len(pre) < MIN_BASELINE_POINTS:
        pre = inside[:MIN_BASELINE_POINTS]
        note = "window_start"
    base = statistics.fmean(pre)
    sd = statistics.pstdev(pre)
    if name == "cert_seconds_to_expiry":
        out_of_band = [values[i] < 0 for i in idx_in]
    else:
        k = 5 if len(pre) >= 10 else 8  # a short baseline gives a poor sd estimate
        band = max(k * sd, 0.25 * abs(base), 1e-6)
        out_of_band = [abs(values[i] - base) > band for i in idx_in]
    crossed = next(
        (times[idx_in[k]] for k in range(len(idx_in) - 1) if out_of_band[k] and out_of_band[k + 1]),
        None,
    )
    mean = statistics.fmean(inside)
    out: dict[str, Any] = {
        "baseline": _sig(base),
        "mean": _sig(mean),
        "min": _sig(min(inside)),
        "max": _sig(max(inside)),
        "change_factor": _sig(mean / base, 3) if abs(base) > 1e-9 else None,
        "first_crossed": crossed,
    }
    if note:
        out["baseline_from"] = note
    return out


def get_metrics(scenario: Scenario, service: str, start: str, end: str) -> dict[str, Any]:
    if (err := _bad_window(start, end)) is not None:
        return err
    series = scenario.metrics.series.get(service)
    if series is None:
        return {"error": f"unknown service '{service}'"}
    metrics = {}
    for name, values in series.items():
        summary = _summarise_metric(name, values, scenario.metrics.timestamps, start, end)
        if summary is not None:
            metrics[name] = summary
    return {"service": service, "window": [start, end], "metrics": metrics}


TOOLS = {
    "get_alerts": get_alerts,
    "get_metrics": get_metrics,
    "get_deploys": get_deploys,
    "get_logs": get_logs,
    "get_topology": get_topology,
}
