"""Render a (filtered) trace as text for the LLM judge, with a fixed budget per category.

Every category gets its own character budget (tokens x 3) and is rendered from its own events
only, so removing one category never gives the others more room and never changes how they look.
A removed category renders as an explicit "not logged" line. Nothing is hand-picked per category:
events are compact JSON, long strings keep their head and tail (text appended to the end of a
prompt survives), and if a category is still over budget the cap is tightened, then whole events
from the middle are left out. The run id is never shown (it ends in the fault's name here).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..tracing.categories import CATEGORY_IDS, SKELETON

CHARS_PER_TOKEN = 3  # JSON runs about 3 characters per token
# the original budgets (300/900/1500/1200/300/400/700) scaled by the same 0.75, proportions kept
BUDGET_TOKENS: dict[str, int] = {
    SKELETON: 225,
    "step_io": 675,
    "tool_calls": 1125,
    "prompt_shown": 900,
    "confidence": 225,
    "timings": 300,
    "handover_state": 525,
}
ORDER: tuple[str, ...] = (SKELETON, *CATEGORY_IDS.values())
START_CAP = 600  # longest string kept at first
MIN_CAP = 24
ELLIPSIS = " ... "


def shorten(s: str, cap: int) -> str:
    """Keep the head and the tail of a long string; the middle becomes an ellipsis."""
    if len(s) <= cap:
        return s
    keep = max(cap - len(ELLIPSIS), 2)
    head = (keep + 1) // 2
    return s[:head] + ELLIPSIS + s[len(s) - (keep - head) :]


def shrink(obj: Any, cap: int) -> Any:
    if isinstance(obj, str):
        return shorten(obj, cap)
    if isinstance(obj, dict):
        return {k: shrink(v, cap) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shrink(v, cap) for v in obj]
    return obj


def _line(ev: dict[str, Any], cap: int) -> str:
    return json.dumps(
        {"step": ev["step"], "span": ev["span_id"], "data": shrink(ev["payload"], cap)},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def render_category(events: list[dict[str, Any]], budget_chars: int) -> str:
    """The events of one category as JSON lines that fit in `budget_chars`."""
    if not events:
        return ""
    cap = START_CAP
    while True:
        lines = [_line(e, cap) for e in events]
        if sum(len(x) + 1 for x in lines) <= budget_chars or cap <= MIN_CAP:
            break
        cap = max(cap // 2, MIN_CAP)
    if sum(len(x) + 1 for x in lines) <= budget_chars:
        return "\n".join(lines)
    # still too long at the tightest cap: keep events from both ends, leave out the middle
    marker = "...{} events left out..."
    lo, hi = 0, len(lines) // 2
    best = [marker.format(len(lines))]
    while lo <= hi:
        k = (lo + hi) // 2
        cand = [*lines[:k], marker.format(len(lines) - 2 * k), *lines[len(lines) - k :]]
        if sum(len(x) + 1 for x in cand) <= budget_chars:
            best, lo = cand, k + 1
        else:
            hi = k - 1
    return "\n".join(best)


def render_trace(events: Iterable[dict[str, Any]], budgets: dict[str, int] | None = None) -> str:
    """Sections in a fixed order. `events` may already be filtered; absent categories say so."""
    tokens = budgets or BUDGET_TOKENS
    by_cat: dict[str, list[dict[str, Any]]] = {c: [] for c in ORDER}
    for e in events:
        by_cat[e["category"]].append(e)
    parts = []
    for cat in ORDER:
        ident = (
            "skeleton" if cat == SKELETON else next(k for k, v in CATEGORY_IDS.items() if v == cat)
        )
        title = f"## {ident} {cat}"
        body = render_category(by_cat[cat], tokens[cat] * CHARS_PER_TOKEN)
        parts.append(f"{title}\n{body}" if body else f"{title}\n(not logged)")
    return "\n\n".join(parts)
