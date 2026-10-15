"""Real-model smoke test: run the three agents on a few scenarios. NOT part of pytest.

uv run python scripts/smoke_agents.py [scenario_id ...]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from logablation.agents import OllamaClient, action_agent, evidence_agent, guard, reasoning_agent
from logablation.scenarios.models import Scenario
from logablation.scenarios.taxonomy import INCIDENT_TYPES
from logablation.tools import ToolContext, build_ticket

DEFAULT = ["sc-01002", "sc-01003", "sc-01001"]  # db_pool_exhausted, bad_deploy, cache storm
MODEL = "qwen2.5:3b"
SCEN_DIR = Path("data/scenarios")


def pick_clean(n: int, first_seed: int) -> list[str]:
    """n scenarios, round-robin over incident types, taking the lowest seeds >= first_seed."""
    by_type: dict[str, list[str]] = {t: [] for t in INCIDENT_TYPES}
    for f in sorted(SCEN_DIR.glob("sc-*.json")):
        sc = Scenario.model_validate_json(f.read_text())
        if sc.seed >= first_seed:
            by_type[sc.incident_type].append(sc.scenario_id)
    out: list[str] = []
    for i in range(n):
        out.append(by_type[INCIDENT_TYPES[i % len(INCIDENT_TYPES)]][i // len(INCIDENT_TYPES)])
    return sorted(out)


def show(title: str, res) -> None:  # type: ignore[no-untyped-def]
    r = res.llm.final_reply
    t = res.llm.timings
    print(f"--- {title}: ok={res.ok} attempts={len(res.llm.attempts)}")
    if r is not None:
        print(
            f"    prompt_tokens={t.get('prompt_eval_count')} out_tokens={t.get('eval_count')} "
            f"wall={t.get('wall_s', 0):.1f}s logprobs={len(r.logprobs)} options={r.options}"
        )
    if res.ok:
        print(res.output.model_dump_json(indent=2))  # type: ignore[union-attr]
    else:
        print("    FAILED:", res.llm.error, "\n    raw:", res.llm.raw_text[:300])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="scenario ids (default: 3 fixed ones)")
    ap.add_argument("--n", type=int, help="run n scenarios, balanced over incident types")
    ap.add_argument("--first-seed", type=int, default=1100, help="lowest seed used with --n")
    ap.add_argument("--quiet", action="store_true", help="one summary line per scenario")
    a = ap.parse_args()
    ids = a.ids or (pick_clean(a.n, a.first_seed) if a.n else DEFAULT)
    print(f"scenarios ({len(ids)}): {ids}")
    guard.preflight(MODEL)
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    print("ollama ps before:\n" + ps)
    llm = OllamaClient(model=MODEL)
    tally: list[dict] = []  # type: ignore[type-arg]
    for sid in ids:
        sc = Scenario.model_validate_json((SCEN_DIR / f"{sid}.json").read_text())
        gt = sc.ground_truth
        ticket = build_ticket(sc)
        t0 = time.perf_counter()
        row = {
            "id": sid,
            "type": sc.incident_type,
            "rc": False,
            "ac": False,
            "got": None,
            "failed": None,
            "dups": 0,
            "retries": 0,
            "secs": 0.0,
            "foreign": [],
        }
        if not a.quiet:
            print(f"\n===== {sid} ({sc.incident_type}) symptom={ticket.symptom_service}")
            print(
                f"ground truth: root_cause={gt.root_cause} service={gt.root_cause_service} "
                f"action={gt.correct_action}"
            )
        ev = evidence_agent(ticket, ToolContext(sc), llm)
        steps = [ev]
        row["dups"] = ev.meta.get("duplicates_removed", 0)
        if not a.quiet:
            show("evidence_agent", ev)
        if ev.ok:
            dg = reasoning_agent(ev.output, llm)  # type: ignore[arg-type]
            steps.append(dg)
            if not a.quiet:
                show("reasoning_agent", dg)
            if dg.ok:
                row["got"] = str(dg.output.root_cause)  # type: ignore[union-attr]
                ap_ = action_agent(dg.output, llm)  # type: ignore[arg-type]
                steps.append(ap_)
                if not a.quiet:
                    show("action_agent", ap_)
                row["rc"] = row["got"] == gt.root_cause
                row["ac"] = ap_.ok and str(ap_.output.action) == gt.correct_action  # type: ignore[union-attr]
        for name, st in zip(("evidence", "reasoning", "action"), steps, strict=False):
            row["retries"] += len(st.llm.attempts) - 1  # type: ignore[operator]
            if not st.ok and row["failed"] is None:
                row["failed"] = name
        if len(steps) < 3 and row["failed"] is None:
            row["failed"] = "unknown"
        row["secs"] = time.perf_counter() - t0
        row["foreign"] = guard.check_after_run(MODEL)
        tally.append(row)
        print(
            f"{sid} {sc.incident_type:<22} truth_rc={gt.root_cause:<22} got_rc={row['got']!s:<22} "
            f"root_cause={'OK ' if row['rc'] else 'BAD'} action={'OK ' if row['ac'] else 'BAD'} "
            f"dups_removed={row['dups']} retries={row['retries']} failed_step={row['failed']} "
            f"{row['secs']:.0f}s"
            + (f" CONTAMINATED foreign={row['foreign']}" if row["foreign"] else "")
        )
    n = len(tally)
    print(f"\nSUMMARY over {n} scenarios")
    print(
        f"  root_cause accuracy: {sum(r['rc'] for r in tally)}/{n} "
        f"= {sum(r['rc'] for r in tally) / n:.1%}"
    )
    print(
        f"  action accuracy:     {sum(r['ac'] for r in tally)}/{n} "
        f"= {sum(r['ac'] for r in tally) / n:.1%}"
    )
    both = sum(r["rc"] and r["ac"] for r in tally)
    print(f"  both (run success):  {both}/{n} = {both / n:.1%}")
    print(f"  mean time per run:   {sum(r['secs'] for r in tally) / n:.1f}s")
    print(f"  agent failures (schema, after retry): {sum(r['failed'] is not None for r in tally)}")
    print(
        f"  total retries: {sum(r['retries'] for r in tally)}   "
        f"duplicate findings removed: {sum(r['dups'] for r in tally)}"
    )
    bad = [r for r in tally if r["foreign"]]
    verdict = f"YES in {len(bad)} runs: {[r['id'] for r in bad]}" if bad else "none"
    print(f"  foreign model detected: {verdict}")
    print("  per incident type (root_cause ok / n):")
    for t in INCIDENT_TYPES:
        rs = [r for r in tally if r["type"] == t]
        if rs:
            print(f"    {t:<22} {sum(r['rc'] for r in rs)}/{len(rs)}")
    llm.unload()
    print(
        "\nollama ps after unload:\n"
        + subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    )


if __name__ == "__main__":
    main()
