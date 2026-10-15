"""Residency guard: foreign models are detected, the allowed one is not."""

from __future__ import annotations

import pytest

from logablation.agents import guard


def _fake(monkeypatch: pytest.MonkeyPatch, models: list[dict]) -> None:  # type: ignore[type-arg]
    monkeypatch.setattr(guard, "loaded_models", lambda url=guard.DEFAULT_URL: models)


def test_allowed_only_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, [{"name": "qwen2.5:3b", "size": 2_500_000_000, "context_length": 8192}])
    assert guard.foreign_models("qwen2.5:3b") == []
    guard.preflight("qwen2.5:3b")


def test_latest_suffix_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, [{"name": "phi4-mini:latest", "size": 1}])
    assert guard.foreign_models("phi4-mini") == []


def test_foreign_model_blocks_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, [{"name": "qwen2.5:7b", "size": 5_000_000_000, "context_length": 131072}])
    assert len(guard.foreign_models("qwen2.5:3b")) == 1
    with pytest.raises(SystemExit):
        guard.preflight("qwen2.5:3b")


def test_oversized_context_blocks_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake(monkeypatch, [{"name": "qwen2.5:3b", "size": 1, "context_length": 131072}])
    with pytest.raises(SystemExit):
        guard.preflight("qwen2.5:3b")
    assert guard.check_after_run("qwen2.5:3b")
