"""`make run`: plan the runs, skip finished ones, run the rest, append one JSON line per run.

Resumable by construction: run ids are deterministic ("<scenario_id>-clean"), the output file is
append-only and flushed after every run, so a crash loses at most the run in progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..agents import guard
from ..agents.llm import LLMClient
from ..scenarios.models import Scenario
from ..scenarios.taxonomy import INCIDENT_TYPES
from .graph import run_pipeline
from .hooks import PipelineHooks
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


def run_clean_phase(
    scenario_ids: list[str],
    llm: LLMClient,
    out: Path,
    *,
    scen_dir: Path = SCEN_DIR,
    hooks: PipelineHooks | None = None,
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
        rec = run_pipeline(sc, llm, run_id=rid, hooks=hooks)
        rec.foreign_models = check_foreign()
        with out.open("a") as fh:
            fh.write(rec.model_dump_json() + "\n")
            fh.flush()
        new.append(rec)
        flag = f" CONTAMINATED {rec.foreign_models}" if rec.contaminated else ""
        log(
            f"{rid} {rec.incident_type:<22} {rec.outcome:<15} "
            f"{rec.timings['total_s']:.0f}s failed_step={rec.failed_step}{flag}"
        )
    return new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, help="run at most this many clean runs")
    a = ap.parse_args(argv)

    cfg: dict[str, Any] = yaml.safe_load(a.config.read_text())
    model = cfg["models"]["agent"]
    mix = cfg["run_mix"]
    ids = plan_clean(SCEN_DIR, mix["clean"], mix.get("first_seed", cfg["scenarios"]["base_seed"]))
    if a.limit is not None:
        todo = [i for i in ids if clean_run_id(i) not in finished_run_ids(a.out)]
        ids = [i for i in ids if i in todo[: a.limit] or clean_run_id(i) in finished_run_ids(a.out)]
    print(f"clean phase: {len(ids)} scenarios -> {a.out}")
    guard.preflight(model)  # exits if any other model is loaded
    from ..agents import OllamaClient  # local import keeps tests free of the HTTP client

    llm = OllamaClient(model=model, num_ctx=cfg["models"]["num_ctx"])
    try:
        run_clean_phase(ids, llm, a.out, check_foreign=lambda: guard.check_after_run(model))
    finally:
        llm.unload()
    done = [json.loads(x) for x in a.out.read_text().splitlines() if x.strip()]
    n = len(done)
    ok = sum(r["outcome"] == "clean_success" for r in done)
    print(f"\n{n} clean runs on file: {ok} clean_success, {n - ok} natural_failure")
    bad = [r["run_id"] for r in done if r.get("foreign_models")]
    print(f"foreign model detected: {bad if bad else 'none'}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
