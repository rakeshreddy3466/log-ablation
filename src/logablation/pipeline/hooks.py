"""Every place a fault or the tracer can attach. All default to no-ops.

Milestone 5 (faults) fills the tool transform, the reasoning prompt/options hooks and the two
handover hooks; milestone 6 (tracer) fills on_event. Agent code is never edited.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agents.llm import LLMClient, LLMReply
from ..tools.context import Transform, identity_transform

HandoverHook = Callable[[dict[str, Any]], dict[str, Any]]  # object sent -> object received
PromptHook = Callable[[str, str], tuple[str, str]]  # (system, user) -> (system, user)
# (temperature, seed) as the agent asked for them (None = client default) -> what is used
OptionsHook = Callable[[float | None, int | None], tuple[float | None, int | None]]


@dataclass
class Event:
    """One step-boundary event. `data` carries the live objects; the tracer serialises them."""

    run_id: str
    step: str
    phase: str  # "start" | "end"
    timestamp: float  # time.time()
    data: dict[str, Any] = field(default_factory=dict)


EventHook = Callable[[Event], None]


def _identity_handover(obj: dict[str, Any]) -> dict[str, Any]:
    return obj


def _identity_prompt(system: str, user: str) -> tuple[str, str]:
    return system, user


def _identity_options(
    temperature: float | None, seed: int | None
) -> tuple[float | None, int | None]:
    return temperature, seed


def _no_event(event: Event) -> None:
    return None


@dataclass
class PipelineHooks:
    tool_transform: Transform = identity_transform  # E1-E3
    reasoning_prompt: PromptHook = _identity_prompt  # R1
    reasoning_options: OptionsHook = _identity_options  # R2
    handover_1: HandoverHook = _identity_handover  # I1, I2
    handover_2: HandoverHook = _identity_handover  # I3
    on_event: EventHook = _no_event  # tracer


class HookedLLM:
    """Wraps an LLMClient so the reasoning call passes through the prompt and options hooks.

    The agent calls ``chat`` as usual; this is the only thing between it and the model, which is
    how R1 and R2 work without touching agent code.
    """

    def __init__(self, inner: LLMClient, prompt: PromptHook, options: OptionsHook) -> None:
        self.inner, self.prompt, self.options = inner, prompt, options
        self.model = getattr(inner, "model", "unknown")

    def chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMReply:
        system, user = self.prompt(system, user)
        temperature, seed = self.options(temperature, seed)
        return self.inner.chat(system, user, schema, temperature=temperature, seed=seed)


def now() -> float:
    return time.time()
