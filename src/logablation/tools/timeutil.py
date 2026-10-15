"""Tiny helpers for the scenario's ISO-8601 'Z' timestamps (one-minute resolution)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

_FMT = "%Y-%m-%dT%H:%M:%SZ"


def parse(ts: str) -> datetime:
    """Parse '2025-03-10T09:40:00Z' (also accepts '+00:00') into an aware UTC datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(_FMT)


def shift(ts: str, minutes: int) -> str:
    return fmt(parse(ts) + timedelta(minutes=minutes))
