"""The log categories (CLAUDE.md "Logging") and the one serialisation every module shares."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

SKELETON = "skeleton"
CATEGORY_IDS: dict[str, str] = {
    "C1": "step_io",
    "C2": "tool_calls",
    "C3": "prompt_shown",
    "C4": "confidence",
    "C5": "timings",
    "C6": "handover_state",
}
ABLATABLE: tuple[str, ...] = tuple(CATEGORY_IDS.values())
ALL_CATEGORIES: tuple[str, ...] = (*ABLATABLE, SKELETON)


def normalise_categories(names: Iterable[str]) -> set[str]:
    """Accept ids (C1) or names (step_io); return names. Unknown values are an error."""
    out: set[str] = set()
    for n in names:
        n = n.strip()
        name = CATEGORY_IDS.get(n.upper(), n)
        if name not in ALL_CATEGORIES:
            raise ValueError(f"unknown log category {n!r}; use one of {list(ALL_CATEGORIES)}")
        out.add(name)
    return out


def to_line(event: dict[str, Any]) -> str:
    """Canonical one-line JSON of an event (compact, key order as built, no trailing newline)."""
    return json.dumps(event, separators=(",", ":"), ensure_ascii=False)
