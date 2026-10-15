"""LLM judge: all-at-once attribution over the filtered, budgeted trace (qwen2.5:7b).

The judge sees the taxonomy and one rendered trace. It never sees the variant, the outcome, the
ground truth or the run id. Output is validated {first_error_step, root_cause_variant, reason,
confidence} with one retry. The runner is resumable (jsonl is appended after every call and
compacted to parquet), randomises the (run, condition) order with a seed, and checks Ollama's
resident models after every call.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..agents import guard
from ..agents.llm import NUM_CTX, LLMClient, OllamaClient
from ..agents.structured import generate_structured
from ..scenarios.taxonomy import FIRST_ERROR_STEP, VARIANT_NAMES, VARIANTS
from ..tracing.categories import ABLATABLE, CATEGORY_IDS
from ..tracing.filter import filter_trace
from .ablation import load_side, profile_name
from .render import render_trace

STEPS = ("evidence_agent", "handover_1", "reasoning_agent", "handover_2", "action_agent")
JUDGE_MODEL = "qwen2.5:7b"
JUDGE_SEED = 7
OUT_JSONL = Path("attribution/predictions_judge.jsonl")
OUT_PARQUET = Path("attribution/predictions_judge.parquet")

StepName = Literal["evidence_agent", "handover_1", "reasoning_agent", "handover_2", "action_agent"]
VariantName = Literal["E1", "E2", "E3", "R1", "R2", "I1", "I2", "I3"]


class JudgeVerdict(BaseModel):
    first_error_step: StepName
    root_cause_variant: VariantName
    reason: str = Field(max_length=600)
    confidence: float = Field(ge=0.0, le=1.0)


STEP_TEXT = {
    "evidence_agent": "collects facts with the monitoring tools and writes the evidence bundle",
    "handover_1": "passes the evidence bundle from the evidence agent to the reasoning agent",
    "reasoning_agent": "reads the bundle and decides the root cause",
    "handover_2": "passes the diagnosis from the reasoning agent to the action agent",
    "action_agent": "chooses the remediation action and its target service",
}
VARIANT_TEXT = {
    "E1": "a metrics tool returned a time window different from the one that was requested",
    "E2": "tools returned data about a neighbouring service instead of the one that was asked for",
    "E3": "the alert list is missing the alert for a service whose metrics show the anomaly",
    "R1": "the reasoning prompt had a biasing note about past incidents added to it",
    "R2": "the reasoning call ran with a high sampling temperature (unstable sampling)",
    "I1": "handover 1 removed a field (such as affected_service) from the evidence bundle",
    "I2": "handover 1 passed on only the first findings of the evidence bundle",
    "I3": "handover 2 removed suspected_service from the diagnosis",
}


def system_prompt() -> str:
    steps = "\n".join(f"- {s}: {STEP_TEXT[s]}" for s in STEPS)
    variants = "\n".join(
        f"- {v} ({VARIANT_NAMES[v]}, first error at {FIRST_ERROR_STEP[v]}): {VARIANT_TEXT[v]}"
        for v in VARIANTS
    )
    return (
        "You are a reliability engineer reading the log of one run of a three-agent "
        "incident-triage pipeline for an online shop. Steps, in order:\n"
        f"{steps}\n\n"
        "Exactly one fault was injected into this run. Possible faults:\n"
        f"{variants}\n\n"
        "The log is split into sections C1..C6 plus the skeleton; a section may say "
        "'(not logged)'. "
        "Long strings are shortened with ' ... ' (head and tail kept). The run may still have "
        "ended with a correct answer: find where the fault was introduced anyway. "
        "Output only JSON with first_error_step, root_cause_variant, reason (one or two "
        "sentences, citing the log evidence) and confidence (0 to 1)."
    )


def user_prompt(events: list[dict[str, Any]]) -> str:
    return "Log of the run:\n\n" + render_trace(events) + "\n\nWhich fault was injected, and where?"


def judge_events(llm: LLMClient, events: list[dict[str, Any]]) -> dict[str, Any]:
    """One judge call over already-filtered events; a failed call returns valid=False."""
    t0 = time.perf_counter()
    res = generate_structured(
        llm, JudgeVerdict, system_prompt(), user_prompt(events), temperature=0.0, seed=JUDGE_SEED
    )
    wall = time.perf_counter() - t0
    v: JudgeVerdict | None = res.parsed
    tm = res.timings
    return {
        "valid": res.ok,
        "pred_first_error": v.first_error_step if v else "invalid",
        "pred_variant": v.root_cause_variant if v else "invalid",
        "reason": v.reason if v else "",
        "confidence": v.confidence if v else None,
        "attempts": len(res.attempts),
        "error": res.error,
        "wall_s": round(wall, 3),
        "prompt_tokens": tm.get("prompt_eval_count"),
        "output_tokens": tm.get("eval_count"),
        "context_full": any(a.context_full for a in res.attempts),
    }


def conditions() -> list[tuple[str, tuple[str, ...]]]:
    """full, minus each of C1..C6, skeleton only."""
    full = tuple(ABLATABLE)
    out = [("full", full)]
    out += [(f"-{i}", tuple(c for c in full if c != n)) for i, n in CATEGORY_IDS.items()]
    out.append(("skeleton", ()))
    return out


def select_runs(groups: tuple[str, ...]) -> list[dict[str, Any]]:
    return [r for r in load_side("design") if r["meta"]["outcome"] in groups]


def make_client(keep_alive: str = "30m") -> OllamaClient:
    return OllamaClient(model=JUDGE_MODEL, num_ctx=NUM_CTX, keep_alive=keep_alive, timeout_s=600)


def switch_to_judge(agent_model: str = "qwen2.5:3b") -> OllamaClient:
    """Unload the agent model, then refuse to go on unless only the judge model may load."""
    client = make_client()
    try:
        OllamaClient(model=agent_model).unload(agent_model)
    except Exception as e:  # noqa: BLE001 - nothing loaded / not installed: preflight decides
        print(f"unload {agent_model}: {e}")
    guard.preflight(JUDGE_MODEL)
    return client


def timing_pilot(n: int = 5) -> None:
    client = switch_to_judge()
    runs = sorted(select_runs(("manifested",)), key=lambda r: r["meta"]["run_id"])
    rng = random.Random(JUDGE_SEED)
    sample = rng.sample(runs, n)
    secs, rows = [], []
    for r in sample:
        out = judge_events(client, filter_trace(r["events"], []))
        foreign = guard.check_after_run(JUDGE_MODEL)
        secs.append(out["wall_s"])
        rows.append((r["meta"]["run_id"], out, foreign))
        print(
            f"{out['wall_s']:.1f}s prompt_tokens={out['prompt_tokens']} "
            f"out_tokens={out['output_tokens']} attempts={out['attempts']} valid={out['valid']} "
            f"ctx_full={out['context_full']} foreign={foreign}"
        )
    client.unload()
    mean = statistics.fmean(secs)
    n_runs = len(runs)
    n_cond = len(conditions())
    proj = mean * n_runs * n_cond / 3600
    print(
        f"mean {mean:.1f}s/call (median {statistics.median(secs):.1f}, max {max(secs):.1f}); "
        f"projection {n_runs} runs x {n_cond} conditions = {n_runs * n_cond} calls "
        f"~ {proj:.1f} h"
    )


def _done_keys() -> set[tuple[str, str]]:
    if not OUT_JSONL.exists():
        return set()
    done = set()
    for line in OUT_JSONL.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add((r["run_id"], r["profile"]))
    return done


def compact_parquet() -> int:
    rows = [json.loads(x) for x in OUT_JSONL.read_text().splitlines() if x.strip()]
    pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
    return len(rows)


def run_all(groups: tuple[str, ...], limit: int | None = None) -> None:
    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    seed = cfg["bootstrap"]["seed"]
    runs = select_runs(groups)
    jobs = [(r, name, kept) for r in runs for name, kept in conditions()]
    random.Random(seed).shuffle(jobs)  # conditions interleaved, never in blocks
    done = _done_keys()
    todo = [j for j in jobs if (j[0]["meta"]["run_id"], profile_name(j[2])) not in done]
    print(f"{len(jobs)} calls planned, {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    client = switch_to_judge()
    OUT_JSONL.parent.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    with OUT_JSONL.open("a") as fh:
        for i, (r, cond, kept) in enumerate(todo[:limit] if limit else todo, 1):
            m = r["meta"]
            drop = [c for c in ABLATABLE if c not in kept]
            out = judge_events(client, filter_trace(r["events"], drop))
            foreign = guard.check_after_run(JUDGE_MODEL)
            rec = {
                "run_id": m["run_id"], "scenario_id": m["scenario_id"], "group": m["outcome"],
                "condition": cond, "profile": profile_name(kept), "n_kept": len(kept),
                "true_variant": m["variant"], "true_first_error": FIRST_ERROR_STEP[m["variant"]],
                "foreign_models": foreign, "judge_model": JUDGE_MODEL,
                **out,
            }  # fmt: skip
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if foreign:
                raise SystemExit(f"foreign model resident after call {i}: {foreign}")
            if i % 25 == 0:
                el = time.perf_counter() - t_start
                left = el / i * (len(todo) - i) / 3600
                print(f"{i}/{len(todo)} done, {el / i:.1f}s/call, ~{left:.1f} h left")
                compact_parquet()
    n = compact_parquet()
    client.unload()
    print(f"finished: {n} predictions in {OUT_PARQUET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-pilot", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--groups", default="manifested", help="comma list: manifested,masked")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.timing_pilot:
        timing_pilot()
    elif a.run:
        run_all(tuple(a.groups.split(",")), a.limit)
    else:
        ap.error("pass --timing-pilot or --run")


if __name__ == "__main__":
    main()
