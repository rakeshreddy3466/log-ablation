"""Faulted-phase planner: who gets which variant, decided only from the clean results.

Only scenarios whose clean run succeeded are eligible. Each variant is given to the
`per_variant` least-loaded eligible scenarios (seeded tie-break), so a scenario never gets the
same variant twice and variants are balanced; one scenario hosts several variants and they all
share its clean twin.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..scenarios.taxonomy import INCIDENT_TYPES, VARIANTS


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    scenario_id: str
    variant: str
    twin_run_id: str


@dataclass
class FaultPlan:
    runs: list[PlannedRun]
    eligibility: dict[str, tuple[int, int]]  # incident_type -> (clean successes, clean runs)
    shortfall: dict[str, int] = field(default_factory=dict)  # variant -> runs missing vs target


def fault_run_id(scenario_id: str, variant: str) -> str:
    return f"{scenario_id}-{variant}"


def plan_faulted(
    clean_records: Iterable[dict[str, Any]],
    *,
    per_variant: int,
    variants: tuple[str, ...] = VARIANTS,
    seed: int,
    applicable: Callable[[str, str], bool] = lambda scenario_id, variant: True,
) -> FaultPlan:
    """`clean_records`: dicts with run_id, scenario_id, incident_type, variant, success.

    `applicable(scenario_id, variant)` excludes pairs where the fault would change nothing.
    """
    clean = [r for r in clean_records if r.get("variant") is None]
    elig: dict[str, tuple[int, int]] = {}
    for t in INCIDENT_TYPES:
        rs = [r for r in clean if r["incident_type"] == t]
        elig[t] = (sum(bool(r["success"]) for r in rs), len(rs))
    eligible = sorted((r for r in clean if r["success"]), key=lambda r: r["scenario_id"])
    twin = {r["scenario_id"]: r["run_id"] for r in eligible}
    ids = sorted(twin)
    rng = random.Random(seed)
    load: Counter[str] = Counter()
    chosen: list[PlannedRun] = []
    shortfall: dict[str, int] = {}
    for v in variants:
        order = [sid for sid in ids if applicable(sid, v)]
        rng.shuffle(order)  # seeded tie-break; sort is stable so ties keep the shuffled order
        order.sort(key=lambda sid: load[sid])
        take = order[:per_variant]
        if len(take) < per_variant:
            shortfall[v] = per_variant - len(take)
        for sid in take:
            load[sid] += 1
            chosen.append(PlannedRun(fault_run_id(sid, v), sid, v, twin[sid]))
    chosen.sort(key=lambda p: (p.scenario_id, variants.index(p.variant)))
    return FaultPlan(chosen, elig, shortfall)
