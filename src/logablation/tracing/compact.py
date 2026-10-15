"""Compaction: runs/raw/*.jsonl -> runs/traces.parquet, one row per event with its byte size."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = ["run_id", "step", "span_id", "category", "timestamp", "payload", "bytes"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Events of one raw file; a torn last line (crash mid-run) is ignored."""
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def compact(raw_dir: Path, out_path: Path) -> int:
    """Write every event of every raw file; returns the number of rows."""
    rows: dict[str, list[Any]] = {c: [] for c in COLUMNS}
    for f in sorted(raw_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            for c in COLUMNS[:5]:
                rows[c].append(ev[c])
            rows["payload"].append(
                json.dumps(ev["payload"], separators=(",", ":"), ensure_ascii=False)
            )
            rows["bytes"].append(len(line.encode()) + 1)  # + newline: bytes on disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), out_path)
    return len(rows["run_id"])


def read_parquet(path: Path) -> list[dict[str, Any]]:
    """Events back from the parquet file (payload parsed), in file order."""
    t = pq.read_table(path).to_pylist()
    out = []
    for r in t:
        r = dict(r)
        r.pop("bytes")
        r["payload"] = json.loads(r["payload"])
        out.append(r)
    return out


def bytes_by_category(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    """run_id -> category -> {"events": n, "bytes": total}, straight from the parquet."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for r in pq.read_table(path, columns=["run_id", "category", "bytes"]).to_pylist():
        c = out.setdefault(r["run_id"], {}).setdefault(r["category"], {"events": 0, "bytes": 0})
        c["events"] += 1
        c["bytes"] += r["bytes"]
    return out
