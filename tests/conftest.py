"""Tests marked `ollama` need a running Ollama server and are skipped when it is unreachable
(always the case on CI)."""

import urllib.request

import pytest


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=1)
        return True
    except OSError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ollama_up():
        return
    skip = pytest.mark.skip(reason="needs a running Ollama server")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)
