"""Ollama residency guard: only the model we are about to use may be loaded.

A stray model once loaded with a 131072 context hung the Mac, so every run script calls
`preflight` before starting and `check_after_run` after each run. Both read Ollama's /api/ps.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from typing import Any

from .llm import NUM_CTX

DEFAULT_URL = "http://localhost:11434"


def _norm(name: str) -> str:
    return name.removesuffix(":latest")


def loaded_models(url: str = DEFAULT_URL) -> list[dict[str, Any]]:
    """Models currently resident in Ollama (raw /api/ps entries)."""
    with urllib.request.urlopen(url + "/api/ps", timeout=10) as r:
        models: list[dict[str, Any]] = json.loads(r.read()).get("models", [])
    return models


def foreign_models(allowed: str, url: str = DEFAULT_URL) -> list[str]:
    """Names of loaded models other than `allowed`, each with size and context when reported."""
    out = []
    for m in loaded_models(url):
        name = m.get("name") or m.get("model") or "?"
        if _norm(name) != _norm(allowed):
            gb = m.get("size", 0) / 1e9
            ctx = m.get("context_length")
            out.append(f"{name} ({gb:.1f} GB" + (f", ctx {ctx})" if ctx else ")"))
    return out


def oversized_context(allowed: str, url: str = DEFAULT_URL) -> list[str]:
    """The allowed model, if resident with a context above NUM_CTX (never happens by design)."""
    out = []
    for m in loaded_models(url):
        name = m.get("name") or m.get("model") or "?"
        ctx = m.get("context_length")
        if _norm(name) == _norm(allowed) and ctx and ctx > NUM_CTX:
            out.append(f"{name} ctx {ctx} > {NUM_CTX}")
    return out


def preflight(model: str, url: str = DEFAULT_URL) -> None:
    """Exit with an error if any model other than `model` is loaded (or ollama is unreachable)."""
    try:
        foreign = foreign_models(model, url)
        big = oversized_context(model, url)
    except OSError as e:
        sys.exit(f"PREFLIGHT FAILED: cannot reach Ollama /api/ps at {url}: {e}")
    if foreign or big:
        for f in foreign:
            print(f"PREFLIGHT FAILED: foreign model loaded: {f}", file=sys.stderr)
        for b in big:
            print(f"PREFLIGHT FAILED: oversized context: {b}", file=sys.stderr)
        sys.exit(f"Refusing to run: only {model} may be loaded. Unload the above and retry.")
    print(f"preflight ok: no model other than {model} is loaded")


def check_after_run(model: str, url: str = DEFAULT_URL) -> list[str]:
    """Foreign models seen after a run (empty list = clean). Never raises; failures are recorded."""
    try:
        return foreign_models(model, url) + oversized_context(model, url)
    except OSError as e:
        return [f"ps-unreachable: {e}"]
