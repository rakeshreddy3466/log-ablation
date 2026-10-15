"""`make run`: phase 1 = one clean run per scenario, phase 2 = faulted runs planned from the clean
results. Skips finished runs, appends one JSON line per run.

Resumable by construction: run ids are deterministic ("<scenario_id>-clean"), the output file is
append-only and flushed after every run, so a crash loses at most the run in progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from ..agents import guard
from ..agents.llm import LLMClient
from ..scenarios.models import Scenario
from ..scenarios.taxonomy import INCIDENT_TYPES
from .graph import run_pipeline
from .hooks import Event, PipelineHooks
from .record import RunRecord

SCEN_DIR = Path("data/scenarios")
OUT = Path("runs/outcomes.jsonl")


def clean_run_id(scenario_id: str) -> str:
    return f"{scenario_id}-clean"


def plan_clean(scen_dir: Path, n: int, first_seed: int) -> list[str]:
    """n scenario ids, round-robin over incident types, lowest seeds >= first_seed first."""
    by_type: dict[str, list[tuple[int, str]]] = {t: [] for t in INCIDENT_TYPES}
    for f in sorted(scen_dir.glob("sc-*.json")):
        sc = Scenario.model_validate_json(f.read_text())
        if sc.seed >= first_seed:
            by_type[sc.incident_type].append((sc.seed, sc.scenario_id))
    for lst in by_type.values():
        lst.sort()
    out = []
    for i in range(n):
        lst = by_type[INCIDENT_TYPES[i % len(INCIDENT_TYPES)]]
        if i // len(INCIDENT_TYPES) >= len(lst):
            raise ValueError(f"not enough scenarios of type {INCIDENT_TYPES[i % 6]} for n={n}")
        out.append(lst[i // len(INCIDENT_TYPES)][1])
    return sorted(out)


def finished_run_ids(path: Path) -> set[str]:
    """Ids already in the output file. A torn last line (crash mid-write) is ignored."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)["run_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _terminate_torn_line(path: Path) -> None:
    """A crash mid-write can leave the file without a final newline; end that line so the next
    record starts on its own line (the torn fragment stays unparseable and is ignored)."""
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a") as fh:
            fh.write("\n")


def _append(out: Path, rec: RunRecord) -> None:
    with out.open("a") as fh:
        fh.write(rec.model_dump_json() + "\n")
        fh.flush()


def _say(rec: RunRecord, log: Callable[[str], None]) -> None:
    flag = f" CONTAMINATED {rec.foreign_models}" if rec.contaminated else ""
    log(
        f"{rec.run_id} {rec.incident_type:<22} {rec.outcome:<15} "
        f"{rec.timings['total_s']:.0f}s failed_step={rec.failed_step}{flag}"
    )


def run_clean_phase(
    scenario_ids: list[str],
    llm: LLMClient,
    out: Path,
    *,
    scen_dir: Path = SCEN_DIR,
    hooks: PipelineHooks | None = None,
    tracer: Callable[[Event], None] | None = None,
    check_foreign: Callable[[], list[str]] = lambda: [],
    log: Callable[[str], None] = print,
) -> list[RunRecord]:
    """Run (or skip, if finished) one clean run per scenario. Returns the records run now."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _terminate_torn_line(out)
    done = finished_run_ids(out)
    new: list[RunRecord] = []
    for sid in scenario_ids:
        rid = clean_run_id(sid)
        if rid in done:
            log(f"skip {rid} (finished)")
            continue
        sc = Scenario.model_validate_json((scen_dir / f"{sid}.json").read_text())
        run_hooks = hooks or PipelineHooks()
        if tracer is not None:
            run_hooks = replace(run_hooks, on_event=tracer)
        rec = run_pipeline(sc, llm, run_id=rid, hooks=run_hooks)
        rec.foreign_models = check_foreign()
        _append(out, rec)
        new.append(rec)
        _say(rec, log)
    return new


def read_records(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run_faulted_phase(
    plan: list[Any],
    llm: LLMClient,
    out: Path,
    *,
    plan_seed: int,
    scen_dir: Path = SCEN_DIR,
    tracer: Callable[[Event], None] | None = None,
    check_foreign: Callable[[], list[str]] = lambda: [],
    log: Callable[[str], None] = print,
) -> list[RunRecord]:
    """Run (or skip, if finished) each planned faulted run, attaching its clean twin's id.

    The twin is eligible by construction (the planner only uses successful clean runs); the
    outcome label is computed against it. Each fault's seed is derived from (plan_seed, run_id).
    """
    from ..faults import build_hooks, derive_seed  # lazy: faults imports pipeline.hooks

    _terminate_torn_line(out)
    done = finished_run_ids(out)
    new: list[RunRecord] = []
    for item in plan:
        if item.run_id in done:
            log(f"skip {item.run_id} (finished)")
            continue
        sc = Scenario.model_validate_json((scen_dir / f"{item.scenario_id}.json").read_text())
        hooks = build_hooks(item.variant, sc, derive_seed(plan_seed, item.run_id))
        if tracer is not None:
            hooks = replace(hooks, on_event=tracer)
        rec = run_pipeline(
            sc,
            llm,
            run_id=item.run_id,
            variant=item.variant,
            twin_run_id=item.twin_run_id,
            twin_success=True,
            hooks=hooks,
        )
        rec.foreign_models = check_foreign()
        _append(out, rec)
        new.append(rec)
        _say(rec, log)
    return new


def write_eligibility(
    eligibility: dict[str, tuple[int, int]],
    path: Path,
    planned_by_type: dict[str, int] | None = None,
) -> None:
    """results/eligibility.csv: per incident type, clean successes (= eligible scenarios), clean
    runs, how many scenarios dropped out (ineligible = the shortfall) and planned faulted runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    planned = planned_by_type or {}
    lines = ["incident_type,eligible,clean_runs,ineligible,rate,planned_faulted_runs"]
    for t, (ok, n) in eligibility.items():
        rate = f"{ok / n:.3f}" if n else ""
        lines.append(f"{t},{ok},{n},{n - ok},{rate},{planned.get(t, 0)}")
    tot_ok = sum(o for o, _ in eligibility.values())
    tot_n = sum(n for _, n in eligibility.values())
    rate = f"{tot_ok / tot_n:.3f}" if tot_n else ""
    lines.append(f"ALL,{tot_ok},{tot_n},{tot_n - tot_ok},{rate},{sum(planned.values())}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    from ..agents import OllamaClient  # local import keeps tests free of the HTTP client
    from ..faults import plan_faulted

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--phase", choices=["clean", "faulted", "all"], default="all")
    ap.add_argument("--limit", type=int, help="run at most this many NEW runs per phase")
    ap.add_argument("--variants", help="faulted phase: comma-separated variants to run")
    ap.add_argument("--scenarios", help="faulted phase: comma-separated scenario ids to run")
    ap.add_argument("--eligibility-out", type=Path, default=Path("results/eligibility.csv"))
    ap.add_argument("--trace-dir", type=Path, default=Path("runs/raw"))
    ap.add_argument("--trace-categories", help="comma list of ids/names to write (default: all)")
    ap.add_argument("--no-trace", action="store_true", help="logging off (latency baseline)")
    ap.add_argument("--traces-parquet", type=Path, default=Path("runs/traces.parquet"))
    a = ap.parse_args(argv)

    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    model = cfg["models"]["agent"]
    mix = cfg["run_mix"]
    guard.preflight(model)  # exits if any other model is loaded
    llm = OllamaClient(model=model, num_ctx=cfg["models"]["num_ctx"])
    foreign = lambda: guard.check_after_run(model)  # noqa: E731
    from ..tracing import Tracer, compact  # lazy: tracing imports pipeline.hooks

    tracer = None
    if not a.no_trace:
        cats = a.trace_categories.split(",") if a.trace_categories else None
        tracer = Tracer(a.trace_dir, cats)
        print(f"tracing to {a.trace_dir}: {sorted(tracer.categories)}")
    try:
        if a.phase in ("clean", "all"):
            ids = plan_clean(SCEN_DIR, mix["clean"], mix["first_seed"])
            if a.limit is not None:
                fin = finished_run_ids(a.out)
                todo = [i for i in ids if clean_run_id(i) not in fin][: a.limit]
                ids = [i for i in ids if i in todo or clean_run_id(i) in fin]
            print(f"clean phase: {len(ids)} scenarios -> {a.out}")
            run_clean_phase(ids, llm, a.out, tracer=tracer, check_foreign=foreign)
        if a.phase in ("faulted", "all"):
            clean = [r for r in read_records(a.out) if r.get("variant") is None]
            plan = plan_faulted(
                clean,
                per_variant=mix["per_variant"],
                variants=tuple(mix["variants"]),
                seed=mix["plan_seed"],
            )
            type_of = {r["scenario_id"]: r["incident_type"] for r in clean}
            planned_by_type: dict[str, int] = {}
            for pr in plan.runs:
                t = type_of[pr.scenario_id]
                planned_by_type[t] = planned_by_type.get(t, 0) + 1
            write_eligibility(plan.eligibility, a.eligibility_out, planned_by_type)
            print(f"planned faulted runs per incident type: {planned_by_type}")
            print(f"eligibility (clean successes / clean runs): {plan.eligibility}")
            print(f"planned {len(plan.runs)} faulted runs; shortfall per variant: {plan.shortfall}")
            runs = plan.runs
            if a.variants:
                runs = [r for r in runs if r.variant in a.variants.split(",")]
            if a.scenarios:
                runs = [r for r in runs if r.scenario_id in a.scenarios.split(",")]
            if a.limit is not None:
                fin = finished_run_ids(a.out)
                runs = [r for r in runs if r.run_id in fin] + [
                    r for r in runs if r.run_id not in fin
                ][: a.limit]
            print(f"faulted phase: {len(runs)} runs -> {a.out}")
            run_faulted_phase(
                runs, llm, a.out, plan_seed=mix["plan_seed"], tracer=tracer, check_foreign=foreign
            )
    finally:
        llm.unload()
    if tracer is not None:
        n_rows = compact(a.trace_dir, a.traces_parquet)
        print(f"compacted {n_rows} events -> {a.traces_parquet}")
    done = read_records(a.out)
    by: dict[str, int] = {}
    for r in done:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    print(f"\n{len(done)} runs on file: {by}")
    bad = [r["run_id"] for r in done if r.get("foreign_models")]
    print(f"foreign model detected: {bad if bad else 'none'}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
