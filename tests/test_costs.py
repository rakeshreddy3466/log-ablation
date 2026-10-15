import hashlib
from typing import Any

from logablation.agents.prompts import REASONING_SYSTEM
from logablation.analysis.costs import TEMPLATES, light_prompt
from logablation.analysis.privacy import (
    Detector,
    count_entities,
    has_residual_sensitive,
    map_strings,
)
from logablation.analysis.profile import FULL_NAME, apply_profile, candidates, name_of
from logablation.attribution.rules import attribute

SENS = {
    "msg": "login ok for eli.weber86@example.test from 113.102.28.126",
    "n": 5,
    "k": ["api key synth_key_e2zqsb4g5a5q84qirnm8l006 ok", "order ORD-88278651 created"],
}


def test_stock_presidio_misses_test_domain_emails_but_fixed_detector_finds_them() -> None:
    stock, fixed = Detector(False), Detector(True)
    s = "login ok for eli.weber86@example.test from 113.102.28.126"
    assert [t for t, *_ in stock.spans(s)] == ["IP_ADDRESS"]
    assert sorted(t for t, *_ in fixed.spans(s)) == ["EMAIL_ADDRESS", "IP_ADDRESS"]


def test_custom_recognizers_find_keys_and_order_ids() -> None:
    d = Detector(False)
    types = {t for s in SENS["k"] for t, *_ in d.spans(s)}
    assert types == {"API_KEY", "ORDER_ID"}


def test_masking_replaces_values_with_type_tags_and_keeps_structure() -> None:
    fixed = Detector(True)
    out = map_strings(SENS, fixed.mask)
    assert out["n"] == 5 and set(out) == set(SENS)
    assert not has_residual_sensitive(out)
    assert "<EMAIL>" in out["msg"] and "<IP>" in out["msg"]
    assert out["k"][1] == "order <ORDER_ID> created"
    assert count_entities(fixed, out) == 0
    # stock masking leaves the e-mail, which is the whole reason for the extended detector
    assert has_residual_sensitive(map_strings(SENS, Detector(False).mask))


def _prompt(system: str) -> dict[str, Any]:
    return {"run_id": "r", "step": "reasoning_agent", "span_id": "reasoning_agent.model.1",
            "category": "prompt_shown", "timestamp": "t",
            "payload": {"attempt": 1, "model": "m", "options": {"temperature": 0.0, "seed": 7},
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": "u"}]}}  # fmt: skip


def test_light_prompt_keeps_template_hash_and_options_only() -> None:
    p = light_prompt(_prompt(REASONING_SYSTEM))["payload"]
    assert set(p) == {"template_id", "system_sha256", "options"}
    assert p["template_id"] == "reasoning_v1"
    assert p["system_sha256"] == hashlib.sha256(REASONING_SYSTEM.encode()).hexdigest()
    assert TEMPLATES[p["system_sha256"]] == "reasoning_v1"
    assert light_prompt(_prompt("changed"))["payload"]["template_id"] == "unknown"


def test_rules_r1_and_r2_work_on_the_light_form() -> None:
    clean = [light_prompt(_prompt(REASONING_SYSTEM))]
    assert attribute(clean)["fired_rules"] == []
    biased = [light_prompt(_prompt(REASONING_SYSTEM + "\n\nNote: 80% were X"))]
    assert attribute(biased)["fired_rules"] == ["R1"]
    hot = light_prompt(_prompt(REASONING_SYSTEM))
    hot["payload"]["options"]["temperature"] = 1.3
    assert attribute([hot])["fired_rules"] == ["R2"]


def test_candidates_cover_every_combination_once() -> None:
    cs = candidates()
    names = [name_of(c) for c in cs]
    assert len(cs) == 4 * 5 * 16 and len(set(names)) == len(names)
    assert FULL_NAME in names and "skeleton" in names and "C2x+C3l+C6" in names


def test_apply_profile_substitutes_only_the_chosen_category() -> None:
    skeleton = {"run_id": "r", "step": "run", "span_id": "run", "category": "skeleton",
                "timestamp": "t", "payload": {"event": "run_start"}}  # fmt: skip
    prompt = _prompt(REASONING_SYSTEM)
    tool = {**skeleton, "step": "evidence_agent", "category": "tool_calls", "payload": {"a": 1}}
    events = [skeleton, tool, prompt]
    alt = {"prompt_shown:light": {2: light_prompt(prompt)}}
    ch = {"C1": None, "C2": "raw", "C3": "light", "C4": None, "C5": None, "C6": None}
    out = apply_profile(events, alt, ch)
    assert out[0] is skeleton and out[1] is tool
    assert set(out[2]["payload"]) == {"template_id", "system_sha256", "options"}
    ch["C2"] = None
    assert [e["category"] for e in apply_profile(events, alt, ch)] == ["skeleton", "prompt_shown"]
