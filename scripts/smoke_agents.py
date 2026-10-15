"""Real-model smoke test: run the three agents on a few scenarios. NOT part of pytest.

uv run python scripts/smoke_agents.py [scenario_id ...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from logablation.agents import OllamaClient, action_agent, evidence_agent, reasoning_agent
from logablation.scenarios.models import Scenario
from logablation.tools import ToolContext, build_ticket

DEFAULT = ["sc-01002", "sc-01003", "sc-01001"]  # db_pool_exhausted, bad_deploy, cache storm
MODEL = "qwen2.5:3b"


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
    ids = sys.argv[1:] or DEFAULT
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    print("ollama ps before:\n" + ps)
    llm = OllamaClient(model=MODEL)
    tally = []
    for sid in ids:
        sc = Scenario.model_validate_json(Path(f"data/scenarios/{sid}.json").read_text())
        gt = sc.ground_truth
        ticket = build_ticket(sc)
        print(f"\n===== {sid} ({sc.incident_type}) symptom={ticket.symptom_service}")
        print(
            f"ground truth: root_cause={gt.root_cause} service={gt.root_cause_service} "
            f"action={gt.correct_action}"
        )
        ev = evidence_agent(ticket, ToolContext(sc), llm)
        print("tool calls:", [(c["tool"], c["args"].get("service")) for c in ev.tool_calls])
        show("evidence_agent", ev)
        if not ev.ok:
            tally.append((sid, False, False))
            continue
        dg = reasoning_agent(ev.output, llm)  # type: ignore[arg-type]
        show("reasoning_agent", dg)
        if not dg.ok:
            tally.append((sid, False, False))
            continue
        ap = action_agent(dg.output, llm)  # type: ignore[arg-type]
        show("action_agent", ap)
        rc = str(dg.output.root_cause) == gt.root_cause  # type: ignore[union-attr]
        ac = ap.ok and str(ap.output.action) == gt.correct_action  # type: ignore[union-attr]
        print(f"root_cause match: {rc}   action match: {ac}")
        tally.append((sid, rc, ac))
    print("\nSUMMARY (scenario, root_cause_ok, action_ok):")
    for row in tally:
        print("  ", row)
    llm.unload()
    print(
        "\nollama ps after unload:\n"
        + subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    )


if __name__ == "__main__":
    main()
