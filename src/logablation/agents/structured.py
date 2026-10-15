"""Schema-validated model call: one retry, then a recorded failure (never an exception)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .llm import NUM_CTX, LLMClient, LLMError, LLMReply

Check = Callable[[Any], str | None]  # returns an error message, or None when fine
MAX_ATTEMPTS = 2


@dataclass
class Attempt:
    reply: LLMReply | None  # None when the transport itself failed
    error: str | None  # None = valid
    context_full: bool = False  # prompt + output filled num_ctx: input may have been cut


@dataclass
class StructuredResult:
    parsed: Any | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.parsed is not None

    @property
    def error(self) -> str | None:
        return None if self.ok else (self.attempts[-1].error if self.attempts else "no attempt")

    @property
    def final_reply(self) -> LLMReply | None:
        return self.attempts[-1].reply if self.attempts else None

    # Convenience views of the last attempt, for the tracer.
    @property
    def raw_text(self) -> str:
        r = self.final_reply
        return r.text if r else ""

    @property
    def prompt(self) -> str:
        r = self.final_reply
        return r.prompt if r else ""

    @property
    def logprobs(self) -> list[dict[str, Any]]:
        r = self.final_reply
        return r.logprobs if r else []

    @property
    def timings(self) -> dict[str, Any]:
        r = self.final_reply
        return r.timings if r else {}


def _short(msg: str, limit: int = 400) -> str:
    return msg if len(msg) <= limit else msg[:limit] + "..."


def generate_structured(
    llm: LLMClient,
    schema_model: type[BaseModel],
    system: str,
    user: str,
    *,
    check: Check | None = None,
    temperature: float | None = None,
    seed: int | None = None,
) -> StructuredResult:
    """Ask for ``schema_model`` JSON. On invalid output retry once, telling the model why.

    The retry uses a different seed (same seed + same prompt at temperature 0 would just repeat
    the mistake). Both attempts are kept so the tracer can log them.
    """
    schema = schema_model.model_json_schema()
    result = StructuredResult(parsed=None)
    prompt_user = user
    for attempt_no in range(MAX_ATTEMPTS):
        try:
            reply = llm.chat(
                system,
                prompt_user,
                schema,
                temperature=temperature,
                seed=seed if attempt_no == 0 else (seed or 0) + 1000 * attempt_no,
            )
        except LLMError as e:
            result.attempts.append(Attempt(reply=None, error=f"llm_error: {e}"))
            continue
        used = reply.timings.get("prompt_eval_count", 0) + reply.timings.get("eval_count", 0)
        full = used >= reply.options.get("num_ctx", NUM_CTX)
        error: str | None = None
        obj: Any = None
        try:
            obj = schema_model.model_validate_json(reply.text)
            if check is not None:
                error = check(obj)
        except ValidationError as e:
            error = f"schema_validation: {_short(str(e))}"
        if error is None:
            result.attempts.append(Attempt(reply=reply, error=None, context_full=full))
            result.parsed = obj
            return result
        result.attempts.append(Attempt(reply=reply, error=error, context_full=full))
        prompt_user = (
            f"{user}\n\nYour previous answer was rejected: {error}\n"
            f"Previous answer: {_short(reply.text, 600)}\n"
            "Answer again with one JSON object that matches the required schema exactly."
        )
    return result


def dump(obj: BaseModel | dict[str, Any]) -> str:
    """Compact, stable JSON for prompts."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    return json.dumps(data, separators=(",", ":"), sort_keys=False)
