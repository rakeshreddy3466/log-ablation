"""Cost and privacy numbers from the recorded traces (no model calls).

One pass over runs/traces.parquet builds a per-run, per-category table: events, raw bytes, gzip
bytes, entities found by Presidio, and planted values actually present (the ground truth, so the
effect of masking does not depend on the detector). Also the lighter variants of C2 and C3, the
write-overhead replay and Presidio's recall / precision against the planted values.

All 375 runs are used for the cost *report*. Profile selection (analysis/profile.py) reads only
the design-side rows of the per-run table.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..agents.prompts import ACTION_SYSTEM, EVIDENCE_SYSTEM, REASONING_SYSTEM
from ..tracing.categories import ABLATABLE, SKELETON, to_line
from ..tracing.compact import read_parquet
from ..tracing.tracer import Tracer
from .privacy import PLANTED_TYPE, Detector, iter_strings, map_strings

# lighter variants: category -> variant names
LIGHT = {"tool_calls": ("masked", "masked_x"), "prompt_shown": ("masked", "masked_x", "light")}
KEYS = [SKELETON, *ABLATABLE] + [f"{c}:{v}" for c, vs in LIGHT.items() for v in vs]
TEMPLATES = {
    hashlib.sha256(EVIDENCE_SYSTEM.encode()).hexdigest(): "evidence_v1",
    hashlib.sha256(REASONING_SYSTEM.encode()).hexdigest(): "reasoning_v1",
    hashlib.sha256(ACTION_SYSTEM.encode()).hexdigest(): "action_v1",
}


def light_prompt(ev: dict[str, Any]) -> dict[str, Any]:
    """C3-light: per model call only {template_id, sha256 of the system prompt, options}."""
    p = ev["payload"]
    system = next(
        (m["content"] for m in (p.get("messages") or []) if m.get("role") == "system"), None
    )
    digest = hashlib.sha256(system.encode()).hexdigest() if system is not None else None
    return {
        **ev,
        "payload": {
            "template_id": TEMPLATES.get(digest, "unknown"),
            "system_sha256": digest,
            "options": p.get("options"),
        },
    }


def build_alternatives(
    events: list[dict[str, Any]], stock: Detector, fixed: Detector
) -> dict[str, dict[int, dict[str, Any]]]:
    """variant key -> {event index -> replacement event}, for the events of that category."""
    alt: dict[str, dict[int, dict[str, Any]]] = {k: {} for k in KEYS if ":" in k}
    for i, ev in enumerate(events):
        cat = ev["category"]
        if cat not in LIGHT:
            continue
        for name in LIGHT[cat]:
            key = f"{cat}:{name}"
            if name == "light":
                alt[key][i] = light_prompt(ev)
            else:
                det = fixed if name == "masked_x" else stock
                alt[key][i] = {**ev, "payload": map_strings(ev["payload"], det.mask)}
    return alt


def load_runs() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """run_id -> events from the parquet (file order), run_id -> outcome metadata."""
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in read_parquet(Path("runs/traces.parquet")):
        by_run[e["run_id"]].append(e)
    meta = {}
    for line in Path("runs/outcomes.jsonl").read_text().splitlines():
        r = json.loads(line)
        meta[r["run_id"]] = r
    return by_run, meta


def planted_by_scenario() -> dict[str, dict[str, str]]:
    out = {}
    for f in Path("data/scenarios").glob("*.json"):
        s = json.loads(f.read_text())
        out[s["scenario_id"]] = {x["value"]: PLANTED_TYPE[x["type"]] for x in s["sensitive_fields"]}
    return out


def planted_count(payload: Any, planted: dict[str, str]) -> int:
    return sum(text.count(v) for text in iter_strings(payload) for v in planted)


def cost_table(
    by_run: dict[str, list[dict[str, Any]]],
    meta: dict[str, dict[str, Any]],
    planted: dict[str, dict[str, str]],
    stock: Detector,
    fixed: Detector,
    side_of: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    det = {t: {"planted": 0, "found": 0, "detected": 0, "false": 0} for t in PLANTED_TYPE.values()}
    for run_id, events in by_run.items():
        sc = meta[run_id]["scenario_id"]
        pl = planted[sc]
        alt = build_alternatives(events, stock, fixed)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            groups[e["category"]].append(e)
        for i, evs in alt.items():
            groups[i] = list(evs.values())
        for key in KEYS:
            evs = groups.get(key, [])
            lines = [to_line(e) for e in evs]
            raw = sum(len(x.encode()) + 1 for x in lines)
            gz = len(gzip.compress(("\n".join(lines) + "\n").encode())) if lines else 0
            n_ent = n_pl = ev_ent = ev_pl = 0
            for e in evs:
                ents = sum(len(stock.spans(s)) for s in iter_strings(e["payload"]))
                pls = planted_count(e["payload"], pl)
                n_ent += ents
                n_pl += pls
                ev_ent += ents > 0
                ev_pl += pls > 0
                if ":" not in key:  # detector accuracy on the original, unmodified events
                    for s in iter_strings(e["payload"]):
                        found = {(t, s[a:b]) for t, a, b in stock.spans(s)}
                        for v, typ in pl.items():
                            n = s.count(v)
                            if n:
                                det[typ]["planted"] += n
                                det[typ]["found"] += n if (typ, v) in found else 0
                        for t, v in found:
                            det[t]["detected"] += 1
                            det[t]["false"] += int(pl.get(v) != t)
            rows.append({
                "run_id": run_id, "scenario_id": sc, "side": side_of.get(sc, "?"),
                "key": key, "events": len(evs), "bytes": raw, "gzip": gz,
                "entities": n_ent, "planted": n_pl,
                "events_with_entity": ev_ent, "events_with_planted": ev_pl,
            })  # fmt: skip
    return pd.DataFrame(rows), det


def write_overhead(by_run: dict[str, list[dict[str, Any]]], rounds: int = 1) -> pd.DataFrame:
    """Time the tracer's write path (serialise + file write + flush) per run and category."""
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tr = Tracer(out_dir=Path(td))
        for warm in (True, False):  # first pass warms the file system and the allocator
            for run_id, events in by_run.items():
                for cat in (SKELETON, *ABLATABLE, "ALL"):
                    evs = [e for e in events if cat == "ALL" or e["category"] == cat]
                    t0 = time.perf_counter_ns()
                    for e in evs:
                        tr._write(run_id, e)
                    tr._close(run_id)
                    dt = (time.perf_counter_ns() - t0) / 1e6
                    if not warm:
                        rows.append(
                            {"run_id": run_id, "key": cat, "write_ms": dt, "events": len(evs)}
                        )
    return pd.DataFrame(rows)


def p95(x: Any) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), 95))


def summarise(
    table: pd.DataFrame, overhead: pd.DataFrame, run_s_median: float, det: dict[str, Any]
) -> pd.DataFrame:
    total_bytes = table[table.key.isin([SKELETON, *ABLATABLE])].bytes.sum()
    out = []
    for key in KEYS:
        t = table[table.key == key]
        row: dict[str, Any] = {
            "key": key,
            "events_per_run_median": float(t.events.median()),
            "bytes_median": float(t.bytes.median()),
            "bytes_p95": p95(t.bytes),
            "gzip_median": float(t.gzip.median()),
            "gzip_p95": p95(t.gzip),
            "share_of_total_bytes": float(t.bytes.sum() / total_bytes) if ":" not in key else None,
            "events_with_entity_share": float(t.events_with_entity.sum() / max(t.events.sum(), 1)),
            "entities_per_run_mean": float(t.entities.mean()),
            "planted_present_per_run_mean": float(t.planted.mean()),
            "events_with_planted_share": float(
                t.events_with_planted.sum() / max(t.events.sum(), 1)
            ),
        }
        o = overhead[overhead.key == key]
        if len(o):
            row |= {
                "write_ms_median": float(o.write_ms.median()),
                "write_ms_p95": p95(o.write_ms),
                "write_pct_of_median_run": float(o.write_ms.median() / (run_s_median * 1000) * 100),
            }
        out.append(row)
    o = overhead[overhead.key == "ALL"]
    out.append(
        {
            "key": "ALL",
            "write_ms_median": float(o.write_ms.median()),
            "write_ms_p95": p95(o.write_ms),
            "write_pct_of_median_run": float(o.write_ms.median() / (run_s_median * 1000) * 100),
        }
    )
    return pd.DataFrame(out)


def detector_scores(det: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for typ, d in det.items():
        pos = d["planted"]
        out[typ] = {
            **d,
            "recall": d["found"] / pos if pos else None,
            "precision": (d["detected"] - d["false"]) / d["detected"] if d["detected"] else None,
        }
    tot = {k: sum(d[k] for d in det.values()) for k in ("planted", "found", "detected", "false")}
    out["ALL"] = {
        **tot,
        "recall": tot["found"] / tot["planted"],
        "precision": (tot["detected"] - tot["false"]) / tot["detected"]
        if tot["detected"]
        else None,
    }
    return out


def main() -> None:
    split = json.loads(Path("results/split.json").read_text())
    side_of = {s: side for side in ("design", "holdout") for s in split[side]}
    by_run, meta = load_runs()
    planted = planted_by_scenario()
    stock, fixed = Detector(False), Detector(True)
    table, det = cost_table(by_run, meta, planted, stock, fixed, side_of)
    stock.save()
    fixed.save()
    overhead = write_overhead(by_run)
    run_s = statistics.median(m["timings"]["total_s"] for m in meta.values())
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/costs_per_run.csv", index=False)
    summ = summarise(table, overhead, run_s, det)
    summ.to_csv("results/costs.csv", index=False)
    overhead.groupby("key").write_ms.describe().to_csv("results/write_overhead.csv")
    scores = detector_scores(det)
    scores["_note"] = (
        "Presidio (stock e-mail recognizer + custom API_KEY and ORDER_ID patterns) against the "
        "planted values, counted over every string leaf of every event of all 375 runs."
    )
    Path("results/privacy_detector.json").write_text(json.dumps(scores, indent=1) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"median run time {run_s:.1f}s over {len(meta)} runs")
    print(summ.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in scores.items() if k != "_note"}, indent=1))


if __name__ == "__main__":
    main()
