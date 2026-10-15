"""ToolContext: the single door every tool call goes through.

The ``transform`` hook sees (tool_name, args, result) and returns the result the agent will
get. It is the identity for now; milestone 5 plugs the E1-E3 faults in here, so tool code and
agent code never change. ``args`` is the full, normalised argument dict (defaults filled in),
which is also what the tracer will record as the tool call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..scenarios.models import Scenario
from .mock_apis import TOOLS

Transform = Callable[[str, dict[str, Any], Any], Any]
CHARS_PER_TOKEN = 4


def identity_transform(tool_name: str, args: dict[str, Any], result: Any) -> Any:
    return result


def estimate_tokens(obj: Any) -> int:
    """Rough token count: compact JSON length / 4 (the project's agreed estimate)."""
    return -(-len(json.dumps(obj, separators=(",", ":"))) // CHARS_PER_TOKEN)


@dataclass
class ToolContext:
    scenario: Scenario
    transform: Transform = identity_transform

    def call(self, tool_name: str, **args: Any) -> Any:
        if tool_name not in TOOLS:
            raise KeyError(f"unknown tool {tool_name!r}")
        result = TOOLS[tool_name](self.scenario, **args)
        return self.transform(tool_name, _normalised(tool_name, args), result)

    def get_alerts(self, start: str, end: str, service: str | None = None) -> Any:
        return self.call("get_alerts", service=service, start=start, end=end)

    def get_metrics(self, service: str, start: str, end: str) -> Any:
        return self.call("get_metrics", service=service, start=start, end=end)

    def get_deploys(self, start: str, end: str) -> Any:
        return self.call("get_deploys", start=start, end=end)

    def get_logs(
        self, service: str, start: str, end: str, level: str | None = None, limit: int = 20
    ) -> Any:
        return self.call(
            "get_logs", service=service, start=start, end=end, level=level, limit=limit
        )

    def get_topology(self) -> Any:
        return self.call("get_topology")


_DEFAULTS: dict[str, dict[str, Any]] = {
    "get_alerts": {"service": None},
    "get_logs": {"level": None, "limit": 20},
}


def _normalised(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {**_DEFAULTS.get(tool_name, {}), **args}
