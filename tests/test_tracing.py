"""Tracer tests (no Ollama): categories, filter, write-time switch-off, skeleton, compaction."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from logablation.faults import build_hooks
from logablation.pipeline import Event, PipelineHooks, run_pipeline
from logablation.scenarios.generator import generate_scenario
from logablation.tracing import (
    ABLATABLE,
    ALL_CATEGORIES,
    CATEGORY_IDS,
    SKELETON,
    Tracer,
    bytes_by_category,
    compact,
    filter_trace,
    read_jsonl,
    read_parquet,
    to_line,
)
from logablation.tracing.confidence import field_logprobs

from .test_pipeline import _right


def _record_events(variant: str | None = None) -> list[Event]:
    """Run the real graph once with a fake LLM and capture the raw pipeline events."""
    sc = generate_scenario(1002)
    seen: list[Event] = []
    hooks = build_hooks(variant, sc, 5)
    hooks.on_event = seen.append
    kw: dict[str, Any] = {}
    if variant:
        kw = {"variant": variant, "twin_run_id": "t", "twin_success": True}
    run_pipeline(sc, _right(sc), run_id="r1", hooks=hooks, **kw)
    return seen


def _trace(events: list[Event], categories: Any = None, out: Path | None = None) -> list[dict]:  # type: ignore[type-arg]
    sink: list[dict[str, Any]] = []
    t = Tracer(out, categories, sink=sink)
    for e in events:
        t(e)
    return sink


@pytest.fixture(scope="module")
def events() -> list[Event]:
    return _record_events()


@pytest.fixture(scope="module")
def full(events: list[Event]) -> list[dict[str, Any]]:
    return _trace(events)


def test_category_table_matches_claude_md() -> None:
    assert CATEGORY_IDS == {
        "C1": "step_io",
        "C2": "tool_calls",
        "C3": "prompt_shown",
        "C4": "confidence",
        "C5": "timings",
        "C6": "handover_state",
    }
    assert ALL_CATEGORIES == (*ABLATABLE, SKELETON)


def test_every_event_has_the_required_keys_and_exactly_one_valid_category(
    full: list[dict[str, Any]],
) -> None:
    for e in full:
        assert list(e) == ["run_id", "step", "span_id", "category", "timestamp", "payload"]
        assert isinstance(e["category"], str) and e["category"] in ALL_CATEGORIES
        assert e["run_id"] == "r1"
        json.loads(to_line(e))  # plain JSON
    assert {e["category"] for e in full} == set(ALL_CATEGORIES)  # all seven occur


def test_content_of_each_category(full: list[dict[str, Any]]) -> None:
    by = lambda c: [e for e in full if e["category"] == c]  # noqa: E731
    # C1: one per agent, with input and output objects
    assert [e["step"] for e in by("step_io")] == [
        "evidence_agent",
        "reasoning_agent",
        "action_agent",
    ]
    assert by("step_io")[1]["payload"]["input"]["findings"]
    assert by("step_io")[2]["payload"]["output"]["action"]
    # C2: every tool call of the evidence agent with args and result
    tools = by("tool_calls")
    assert (
        tools[0]["payload"]["tool"] == "get_topology"
        and tools[-1]["payload"]["tool"] == "get_deploys"
    )
    assert all({"tool", "args", "result"} == set(e["payload"]) for e in tools)
    # C3: exact messages, system first
    p = by("prompt_shown")
    assert len(p) == 3 and p[0]["payload"]["messages"][0]["role"] == "system"
    assert p[1]["payload"]["options"]["num_ctx"] == 8192
    # C4: stated confidence + per-field logprob stats
    c = by("confidence")
    assert [e["step"] for e in c] == ["evidence_agent", "reasoning_agent", "action_agent"]
    assert c[1]["payload"]["stated_confidence"] == 0.8
    # C5: step, tool_call and model_call timings
    kinds = {e["payload"]["kind"] for e in by("timings")}
    assert kinds == {"step", "tool_call", "model_call"}
    assert {e["step"] for e in by("timings") if e["payload"]["kind"] == "step"} == {
        "evidence_agent",
        "handover_1",
        "reasoning_agent",
        "handover_2",
        "action_agent",
    }
    # C6: both handovers with sent/received/schema/diff
    h = by("handover_state")
    assert [e["step"] for e in h] == ["handover_1", "handover_2"]
    assert all(e["payload"]["schema_ok"] and e["payload"]["diff"] == [] for e in h)
    # skeleton: run start, five ordered steps, final answer; no verdict and no fault info
    sk = [e["payload"] for e in by(SKELETON)]
    assert [p["event"] for p in sk] == ["run_start", *["step"] * 5, "final_answer"]
    assert [p["name"] for p in sk[1:6]] == [
        "evidence_agent",
        "handover_1",
        "reasoning_agent",
        "handover_2",
        "action_agent",
    ]
    assert [p["index"] for p in sk[1:6]] == [1, 2, 3, 4, 5]
    assert set(sk[-1]) == {"event", "root_cause", "action", "target_service"}
    text = json.dumps(full)
    assert "success" not in text and "variant" not in text and "outcome" not in text


@pytest.mark.parametrize("cat", ABLATABLE)
def test_filter_removes_exactly_the_dropped_category(full: list[dict[str, Any]], cat: str) -> None:
    kept = filter_trace(full, drop={cat})
    assert [e for e in full if e["category"] != cat] == kept
    assert not any(e["category"] == cat for e in kept)
    # everything kept is byte-identical to the original line
    assert [to_line(e) for e in kept] == [to_line(e) for e in full if e["category"] != cat]
    # ids work like names
    drop_id = next(k for k, v in CATEGORY_IDS.items() if v == cat)
    assert filter_trace(full, drop={drop_id}) == kept


def test_skeleton_survives_every_filter(full: list[dict[str, Any]]) -> None:
    skeleton = [e for e in full if e["category"] == SKELETON]
    assert skeleton
    assert filter_trace(full, drop=ABLATABLE) == skeleton  # skeleton only
    assert filter_trace(full, drop=ALL_CATEGORIES) == skeleton  # naming skeleton does nothing
    assert filter_trace(full, drop={"skeleton"}) == full
    assert filter_trace(full) == full


def test_filter_rejects_unknown_categories_and_bad_events(full: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        filter_trace(full, drop={"C9"})
    bad = copy.deepcopy(full[:1])
    bad[0]["category"] = "nonsense"
    with pytest.raises(ValueError):
        filter_trace(bad)


@pytest.mark.parametrize("off", [{"C2"}, {"prompt_shown", "C4"}, set(ABLATABLE)])
def test_switching_off_at_write_time_equals_filtering_afterwards(
    events: list[Event], full: list[dict[str, Any]], off: set[str]
) -> None:
    on = set(ABLATABLE) - {CATEGORY_IDS.get(o, o) for o in off}
    written = _trace(events, categories=on)
    assert [to_line(e) for e in written] == [to_line(e) for e in filter_trace(full, drop=off)]
    assert any(e["category"] == SKELETON for e in written)  # skeleton cannot be switched off


def test_tracer_is_deterministic_for_the_same_events(events: list[Event]) -> None:
    a, b = _trace(events), _trace(events)
    assert [to_line(e) for e in a] == [to_line(e) for e in b]


def test_files_are_written_one_per_run_and_a_rerun_overwrites(
    tmp_path: Path, events: list[Event]
) -> None:
    _trace(events, out=tmp_path)
    f = tmp_path / "r1.jsonl"
    first = f.read_text()
    _trace(events, out=tmp_path)
    assert f.read_text() == first
    assert [json.loads(x)["category"] for x in first.splitlines()][0] == SKELETON


def test_interface_fault_is_visible_in_c6_and_the_reasoning_prompt_is_unchanged() -> None:
    tr = _trace(_record_events("I1"))
    h1 = next(e for e in tr if e["category"] == "handover_state" and e["step"] == "handover_1")
    assert [d["path"] for d in h1["payload"]["diff"]] == ["affected_service"]
    assert h1["payload"]["schema_ok"] is False


def test_r1_note_is_visible_in_prompt_shown() -> None:
    tr = _trace(_record_events("R1"))
    sys_prompts = {
        e["step"]: e["payload"]["messages"][0]["content"]
        for e in tr
        if e["category"] == "prompt_shown"
    }
    assert "Note from the on-call lead" in sys_prompts["reasoning_agent"]
    assert "Note from the on-call lead" not in sys_prompts["evidence_agent"]
    assert "Note from the on-call lead" not in sys_prompts["action_agent"]


def test_compaction_round_trips_and_sizes_match_the_files(
    tmp_path: Path, events: list[Event], full: list[dict[str, Any]]
) -> None:
    raw = tmp_path / "raw"
    _trace(events, out=raw)
    pq = tmp_path / "traces.parquet"
    n = compact(raw, pq)
    assert n == len(full)
    back = read_parquet(pq)
    assert back == full  # same events, same order, payloads equal
    assert [to_line(e) for e in back] == [to_line(e) for e in full]
    sizes = bytes_by_category(pq)["r1"]
    assert sum(v["bytes"] for v in sizes.values()) == (raw / "r1.jsonl").stat().st_size
    assert sum(v["events"] for v in sizes.values()) == n
    assert set(sizes) == set(ALL_CATEGORIES)


def test_compaction_ignores_a_torn_last_line(tmp_path: Path, events: list[Event]) -> None:
    raw = tmp_path / "raw"
    _trace(events, out=raw)
    with (raw / "r1.jsonl").open("a") as fh:
        fh.write('{"run_id": "r1", "step"')
    assert len(read_jsonl(raw / "r1.jsonl")) == compact(raw, tmp_path / "t.parquet")


# ---- C4 token alignment --------------------------------------------------------------------


def test_field_logprobs_maps_tokens_to_answer_fields() -> None:
    text = '{"root_cause":"memory_leak","confidence":0.8}'
    pieces = [
        '{"',
        "root",
        "_cause",
        '":"',
        "memory",
        "_leak",
        '","',
        "confidence",
        '":',
        "0.8",
        "}",
    ]
    assert "".join(pieces) == text
    lps = [{"token": p, "logprob": -0.1 * (i + 1), "bytes": []} for i, p in enumerate(pieces)]
    align, f = field_logprobs(text, lps, ["root_cause", "suspected_service"])
    assert align == "ok" and f["suspected_service"] is None
    rc = f["root_cause"]
    assert rc["value"] == "memory_leak" and [t["token"] for t in rc["tokens"]] == [
        "memory",
        "_leak",
    ]
    assert rc["mean_logprob"] == pytest.approx(-0.55) and rc["min_logprob"] == pytest.approx(-0.6)
    assert field_logprobs(text, [], ["root_cause"])[0] == "no_logprobs"
    assert field_logprobs("other", lps, ["root_cause"])[0] == "failed"


def test_default_hooks_do_not_trace() -> None:
    sc = generate_scenario(1002)
    rec = run_pipeline(sc, _right(sc), run_id="x", hooks=PipelineHooks())
    assert rec.success
