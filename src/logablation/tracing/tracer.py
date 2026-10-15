"""The tracer: plugs into the pipeline's on_event hook and writes one JSON line per log event.

It is a pure function of the events it receives (times come from the events, never from the
clock), so a category switched off at write time gives exactly the trace that filtering it
afterwards gives. The skeleton cannot be switched off.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from ..pipeline.hooks import Event
from .categories import ABLATABLE, ALL_CATEGORIES, SKELETON, normalise_categories, to_line
from .confidence import field_logprobs

ANSWER_FIELDS = {
    "evidence_agent": ["affected_service"],
    "reasoning_agent": ["root_cause", "suspected_service"],
    "action_agent": ["action", "target_service"],
}
STATED_CONFIDENCE = {
    "evidence_agent": "evidence_confidence",
    "reasoning_agent": "confidence",
    "action_agent": "confidence",
}
AGENT_STEPS = tuple(ANSWER_FIELDS)
HANDOVER_STEPS = ("handover_1", "handover_2")
NS = 1e-9


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _RunState:
    def __init__(self) -> None:
        self.starts: dict[str, float] = {}
        self.inputs: dict[str, Any] = {}
        self.order = 0
        self.root_cause: str | None = None
        self.action: str | None = None
        self.target: str | None = None


class Tracer:
    """Callable event hook. `categories` = names/ids to write (default all); skeleton always."""

    def __init__(
        self,
        out_dir: Path | None = None,
        categories: Iterable[str] | None = None,
        sink: list[dict[str, Any]] | None = None,
    ) -> None:
        wanted = set(ABLATABLE) if categories is None else normalise_categories(categories)
        self.categories = (wanted & set(ABLATABLE)) | {SKELETON}
        self.out_dir = out_dir
        self.sink = sink
        self._runs: dict[str, _RunState] = {}
        self._files: dict[str, IO[str]] = {}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

    # ---- event hook ----------------------------------------------------------------------
    def __call__(self, ev: Event) -> None:
        st = self._runs.setdefault(ev.run_id, _RunState())
        built: list[dict[str, Any]] = []
        if ev.step == "run":
            if ev.phase == "start":
                built.append(
                    self._ev(ev, "run", "run", SKELETON, ev.timestamp, {"event": "run_start"})
                )
            else:
                final = {
                    "event": "final_answer",
                    "root_cause": st.root_cause,
                    "action": st.action,
                    "target_service": st.target,
                }
                built.append(self._ev(ev, "run", "run", SKELETON, ev.timestamp, final))
        elif ev.step in AGENT_STEPS or ev.step in HANDOVER_STEPS:
            if ev.phase == "start":
                st.starts[ev.step] = ev.timestamp
                st.inputs[ev.step] = ev.data.get("input", ev.data.get("sent"))
            else:
                built += self._step_end(ev, st)
        # outcome_check is deliberately not traced: the verdict is a label, not log content
        for e in built:
            if e["category"] in self.categories:
                self._write(ev.run_id, e)
        if ev.step == "run" and ev.phase == "end":
            self._close(ev.run_id)
            self._runs.pop(ev.run_id, None)

    # ---- builders ------------------------------------------------------------------------
    @staticmethod
    def _ev(
        ev: Event, step: str, span: str, category: str, ts: float, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "run_id": ev.run_id,
            "step": step,
            "span_id": span,
            "category": category,
            "timestamp": iso(ts),
            "payload": payload,
        }

    def _step_end(self, ev: Event, st: _RunState) -> list[dict[str, Any]]:
        step, d = ev.step, ev.data
        t0, t1 = st.starts.get(step, ev.timestamp), ev.timestamp
        out: list[dict[str, Any]] = []

        def add(cat: str, span: str, ts: float, payload: dict[str, Any]) -> None:
            out.append(self._ev(ev, step, span, cat, ts, payload))

        timing = {"kind": "step", "start": iso(t0), "end": iso(t1), "duration_s": d.get("wall_s")}
        if step in AGENT_STEPS:
            res = d["agent_result"]
            output = d.get("output")
            for i, call in enumerate(res.tool_calls):
                ts = d["tool_timings"][i] if i < len(d.get("tool_timings", [])) else (t0, t0)
                add(
                    "tool_calls",
                    f"{step}.tool.{i + 1}",
                    ts[0],
                    {
                        "tool": call["tool"],
                        "args": call["args"],
                        "result": call["result"],
                    },
                )
                add(
                    "timings",
                    f"{step}.tool.{i + 1}",
                    ts[0],
                    {
                        "kind": "tool_call",
                        "tool": call["tool"],
                        "start": iso(ts[0]),
                        "end": iso(ts[1]),
                        "duration_s": round(ts[1] - ts[0], 6),
                    },
                )
            mts = d.get("model_timings", [])
            for i, att in enumerate(res.llm.attempts):
                span = f"{step}.model.{i + 1}"
                r = att.reply
                add(
                    "prompt_shown",
                    span,
                    mts[i]["start"] if i < len(mts) else t0,
                    {
                        "attempt": i + 1,
                        "model": r.model if r else None,
                        "messages": r.messages if r else None,
                        "options": r.options if r else None,
                        "error": None if r else att.error,
                    },
                )
                if i < len(mts):
                    m = mts[i]
                    p: dict[str, Any] = {
                        "kind": "model_call",
                        "attempt": i + 1,
                        "start": iso(m["start"]),
                        "end": iso(m["end"]),
                        "duration_s": round(m["end"] - m["start"], 6),
                    }
                    if r:
                        tm = r.timings
                        p |= {
                            "prompt_tokens": tm.get("prompt_eval_count"),
                            "output_tokens": tm.get("eval_count"),
                            "load_s": round(tm.get("load_duration", 0) * NS, 6),
                            "prompt_eval_s": round(tm.get("prompt_eval_duration", 0) * NS, 6),
                            "generation_s": round(tm.get("eval_duration", 0) * NS, 6),
                        }
                    add("timings", span, m["start"], p)
            final = res.llm.final_reply
            align, fields = (
                field_logprobs(final.text, final.logprobs, ANSWER_FIELDS[step])
                if final
                else ("no_logprobs", {f: None for f in ANSWER_FIELDS[step]})
            )
            add(
                "confidence",
                step,
                t1,
                {
                    "stated_confidence": (output or {}).get(STATED_CONFIDENCE[step]),
                    "alignment": align,
                    "fields": fields,
                },
            )
            add(
                "step_io",
                step,
                t1,
                {
                    "input": st.inputs.get(step),
                    "output": output,
                    "attempts": [
                        {"attempt": i + 1, "valid": a.error is None, "error": a.error}
                        for i, a in enumerate(res.llm.attempts)
                    ],
                },
            )
            if step == "reasoning_agent" and output:
                st.root_cause = output.get("root_cause")
            if step == "action_agent" and output:
                st.action, st.target = output.get("action"), output.get("target_service")
        else:
            rec = d["record"]
            add(
                "handover_state",
                step,
                t1,
                {
                    "sent": rec.sent,
                    "received": rec.received,
                    "expected_schema": rec.expected_schema,
                    "schema_ok": rec.schema_ok,
                    "schema_error": rec.schema_error,
                    "diff": rec.diff,
                },
            )
        add("timings", step, t0, timing)
        st.order += 1
        add(SKELETON, step, t1, {"event": "step", "name": step, "index": st.order})
        return out

    # ---- output --------------------------------------------------------------------------
    def _write(self, run_id: str, event: dict[str, Any]) -> None:
        assert event["category"] in ALL_CATEGORIES
        if self.sink is not None:
            self.sink.append(event)
        if self.out_dir is not None:
            fh = self._files.get(run_id)
            if fh is None:  # a rerun of a run overwrites its partial file
                fh = self._files[run_id] = (self.out_dir / f"{run_id}.jsonl").open("w")
            fh.write(to_line(event) + "\n")
            fh.flush()

    def _close(self, run_id: str) -> None:
        fh = self._files.pop(run_id, None)
        if fh is not None:
            fh.close()
