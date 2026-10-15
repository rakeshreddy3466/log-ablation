"""Sensitive-content detection and masking with Microsoft Presidio.

Presidio's own e-mail and IP recognizers plus two custom pattern recognizers (planted API keys
with the synth_key_ prefix, and ORD- order ids). The NLP engine is spaCy en_core_web_sm; only the
four planted entity types are requested. Detection runs per string leaf and is cached by string
(the same log line recurs in every run of a scenario), so the cache is shared by all analyses.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

ENTITIES = ["EMAIL_ADDRESS", "IP_ADDRESS", "API_KEY", "ORDER_ID"]
TAG = {"EMAIL_ADDRESS": "<EMAIL>", "IP_ADDRESS": "<IP>", "API_KEY": "<API_KEY>",
       "ORDER_ID": "<ORDER_ID>"}  # fmt: skip
PLANTED_TYPE = {"email": "EMAIL_ADDRESS", "ip": "IP_ADDRESS", "api_key": "API_KEY",
                "order_id": "ORDER_ID"}  # fmt: skip
CACHE = Path("runs/cache/presidio_leaves.pkl")
CACHE_FIX = Path("runs/cache/presidio_leaves_fix.pkl")
# Presidio's e-mail recognizer validates the top-level domain against the public-suffix list and
# so never matches the reserved ".test" domain the generator plants. The "fixed" detector adds
# one pattern for exactly that domain; the stock detector is what the recall numbers report.
TEST_EMAIL = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.test\b"

Span = tuple[str, int, int]  # (entity type, start, end)


class Detector:
    def __init__(self, email_fix: bool = False) -> None:
        self.email_fix = email_fix
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_analyzer.predefined_recognizers import EmailRecognizer, IpRecognizer

        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        ).create_engine()
        reg = RecognizerRegistry(supported_languages=["en"])
        reg.add_recognizer(EmailRecognizer(supported_language="en"))
        reg.add_recognizer(IpRecognizer(supported_language="en"))
        reg.add_recognizer(
            PatternRecognizer(
                supported_entity="API_KEY",
                patterns=[Pattern("synth_key", r"\bsynth_key_[A-Za-z0-9]{6,}\b", 0.9)],
            )
        )
        reg.add_recognizer(
            PatternRecognizer(
                supported_entity="ORDER_ID",
                patterns=[Pattern("order_id", r"\bORD-\d{4,}\b", 0.9)],
            )
        )
        if email_fix:
            reg.add_recognizer(
                PatternRecognizer(
                    supported_entity="EMAIL_ADDRESS",
                    patterns=[Pattern("test_tld_email", TEST_EMAIL, 0.9)],
                )
            )
        self.engine = AnalyzerEngine(nlp_engine=nlp, registry=reg, supported_languages=["en"])
        self.cache: dict[str, tuple[Span, ...]] = {}
        self.path = CACHE_FIX if email_fix else CACHE
        if self.path.exists():
            self.cache = pickle.loads(self.path.read_bytes())

    def spans(self, text: str) -> tuple[Span, ...]:
        hit = self.cache.get(text)
        if hit is None:
            res = self.engine.analyze(text=text, language="en", entities=ENTITIES)
            res = sorted(res, key=lambda r: (r.start, -(r.end - r.start)))
            kept: list[Span] = []
            end = -1
            for r in res:  # drop overlapping detections, keep the earliest / longest
                if r.start >= end:
                    kept.append((r.entity_type, r.start, r.end))
                    end = r.end
            hit = tuple(kept)
            self.cache[text] = hit
        return hit

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.cache))

    def mask(self, text: str) -> str:
        out, pos = [], 0
        for typ, s, e in self.spans(text):
            out += [text[pos:s], TAG[typ]]
            pos = e
        out.append(text[pos:])
        return "".join(out)


def map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    """Copy of a JSON-like object with every string leaf passed through `fn`."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_strings(v, fn) for v in obj]
    return obj


def iter_strings(obj: Any) -> Any:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def count_entities(det: Detector, payload: Any) -> int:
    return sum(len(det.spans(s)) for s in iter_strings(payload))


_LEAK = re.compile(r"synth_key_|ORD-\d|@[a-z0-9.-]+\.test|\b\d{1,3}(\.\d{1,3}){3}\b")


def has_residual_sensitive(payload: Any) -> bool:
    """Regex backstop used only in tests: does any planted-looking value survive masking?"""
    return any(_LEAK.search(s) for s in iter_strings(payload))
