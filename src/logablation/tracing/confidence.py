"""C4 helper: map token logprobs onto the answer fields of the model's JSON text."""

from __future__ import annotations

import re
from typing import Any


def field_logprobs(
    text: str, logprobs: list[dict[str, Any]], fields: list[str]
) -> tuple[str, dict[str, Any]]:
    """Per answer field: its tokens' logprobs with mean and min.

    Returns (alignment, fields). alignment is "ok", "no_logprobs", or "failed" when the tokens do
    not concatenate to the text (then no field is guessed).
    """
    if not logprobs:
        return "no_logprobs", {f: None for f in fields}
    tokens = [str(t.get("token", "")) for t in logprobs]
    if "".join(tokens) != text:
        return "failed", {f: None for f in fields}
    bounds: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        bounds.append((pos, pos + len(tok)))
        pos += len(tok)
    out: dict[str, Any] = {}
    for f in fields:
        m = re.search('"' + re.escape(f) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if not m:
            out[f] = None
            continue
        s, e = m.span(1)
        picked = [i for i, (a, b) in enumerate(bounds) if a < e and b > s and b > a]
        lps = [float(logprobs[i]["logprob"]) for i in picked]
        out[f] = {
            "value": m.group(1),
            "n_tokens": len(lps),
            "mean_logprob": round(sum(lps) / len(lps), 6) if lps else None,
            "min_logprob": round(min(lps), 6) if lps else None,
            "tokens": [
                {"token": tokens[i], "logprob": round(float(logprobs[i]["logprob"]), 6)}
                for i in picked
            ],
        }
    return "ok", out
