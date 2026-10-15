"""One small wrapper around Ollama used by every model call (agents now, judge later).

Safety rules baked in: ``num_ctx`` is always sent (8192 unless the caller asks for less; asking
for more raises), temperature defaults to 0 with a fixed seed, ``format`` is the Pydantic JSON
schema and ``logprobs`` is always on. Everything the tracer will need comes back in LLMReply.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

NUM_CTX = 8192
DEFAULT_SEED = 7
DEFAULT_URL = "http://localhost:11434"


class LLMError(RuntimeError):
    """Transport / server failure (Ollama down, model missing, bad HTTP status)."""


@dataclass
class LLMReply:
    text: str
    model: str
    messages: list[dict[str, str]]  # exactly what was sent as chat messages
    prompt: str  # the same messages rendered as one string
    logprobs: list[dict[str, Any]]  # [{token, logprob, bytes}, ...] for the generated tokens
    timings: dict[str, Any]  # Ollama durations (ns) and token counts, plus wall_s
    options: dict[str, Any]  # num_ctx / temperature / seed actually sent


class LLMClient(Protocol):
    """The seam used by agents. Fault hooks (R1, R2) and test fakes wrap or implement this."""

    model: str

    def chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMReply: ...


def render_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


@dataclass
class OllamaClient:
    model: str = "qwen2.5:3b"
    url: str = DEFAULT_URL
    temperature: float = 0.0
    seed: int = DEFAULT_SEED
    num_ctx: int = NUM_CTX
    keep_alive: str = "10m"
    timeout_s: float = 300.0
    top_logprobs: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.num_ctx <= NUM_CTX:
            raise ValueError(f"num_ctx must be in 1..{NUM_CTX}, got {self.num_ctx}")

    def build_payload(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float | None,
        seed: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "logprobs": True,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature if temperature is None else temperature,
                "seed": self.seed if seed is None else seed,
            },
        }
        if self.top_logprobs:
            payload["top_logprobs"] = self.top_logprobs
        return payload

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
            # URLError, timeouts and RemoteDisconnected are all OSError/HTTPException
            raise LLMError(f"ollama {path} failed: {e}") from e

    def chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMReply:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload = self.build_payload(messages, schema, temperature, seed)
        t0 = time.perf_counter()
        body = self._post("/api/chat", payload)
        wall = time.perf_counter() - t0
        if "error" in body:
            raise LLMError(str(body["error"]))
        timing_keys = (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
        timings = {k: body[k] for k in timing_keys if k in body} | {"wall_s": wall}
        return LLMReply(
            text=body.get("message", {}).get("content", ""),
            model=self.model,
            messages=messages,
            prompt=render_prompt(messages),
            logprobs=body.get("logprobs") or [],
            timings=timings,
            options=payload["options"],
        )

    def unload(self, model: str | None = None) -> None:
        """Free the model (keep_alive=0). Call when switching agent model -> judge model."""
        self._post(
            "/api/generate",
            {"model": model or self.model, "keep_alive": 0, "options": {"num_ctx": self.num_ctx}},
        )
