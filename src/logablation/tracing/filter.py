"""Post-hoc ablation: a trace without the given categories. The skeleton is never removed."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .categories import ALL_CATEGORIES, SKELETON, normalise_categories


def filter_trace(
    events: Iterable[dict[str, Any]], drop: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Events whose category is not in `drop` (ids like "C2" or names), order preserved.

    Kept events are returned untouched (same objects), so they are byte-identical to the input.
    "skeleton" in `drop` is ignored: the skeleton always survives.
    """
    dropped = normalise_categories(drop) - {SKELETON}
    out = []
    for ev in events:
        cat = ev["category"]
        if cat not in ALL_CATEGORIES:
            raise ValueError(f"event with invalid category {cat!r}")
        if cat not in dropped:
            out.append(ev)
    return out
