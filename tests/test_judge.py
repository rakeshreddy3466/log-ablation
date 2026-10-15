import json
from typing import Any

from logablation.agents.llm import LLMReply
from logablation.attribution import judge
from logablation.attribution.render import (
    BUDGET_TOKENS,
    CHARS_PER_TOKEN,
    ORDER,
    render_category,
    render_trace,
    shorten,
)
from logablation.tracing.categories import ABLATABLE, CATEGORY_IDS
from logablation.tracing.filter import filter_trace

SECRET_RUN = "sc-01100-R1"


def make_events(n_each: int = 40, size: int = 3000) -> list[dict[str, Any]]:
    evs = []
    for cat in ("skeleton", *ABLATABLE):
        for i in range(n_each):
            evs.append({
                "run_id": SECRET_RUN, "step": "evidence_agent", "span_id": f"s{i}",
                "category": cat, "timestamp": "t",
                "payload": {"text": f"{cat}-{i}-" + "x" * size + "-END", "n": i},
            })  # fmt: skip
    return evs


def sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for block in text.split("\n\n## "):
        head, _, body = block.removeprefix("## ").partition("\n")
        out[head.split()[0]] = body
    return out


def test_each_category_stays_within_its_budget() -> None:
    sec = sections(render_trace(make_events()))
    ids = {"skeleton": "skeleton", **{v: k for k, v in CATEGORY_IDS.items()}}
    for cat in ORDER:
        assert len(sec[ids[cat]]) <= BUDGET_TOKENS[cat] * CHARS_PER_TOKEN, cat


def test_removing_a_category_leaves_the_others_byte_identical() -> None:
    evs = make_events()
    full = sections(render_trace(evs))
    for cat_id, cat in CATEGORY_IDS.items():
        part = sections(render_trace(filter_trace(evs, [cat])))
        assert part[cat_id] == "(not logged)"
        for other in full:
            if other != cat_id:
                assert part[other] == full[other], (cat_id, other)


def test_skeleton_cannot_be_removed() -> None:
    sec = sections(render_trace(filter_trace(make_events(), list(ABLATABLE))))
    assert sec["skeleton"] != "(not logged)"


def test_head_and_tail_survive_shortening() -> None:
    s = "HEAD" + "m" * 5000 + "TAIL-appended-note"
    out = shorten(s, 120)
    assert len(out) <= 120 and out.startswith("HEAD") and out.endswith("appended-note")
    assert shorten("short", 120) == "short"


def test_render_is_deterministic_and_event_level_fallback_marks_omissions() -> None:
    evs = [e for e in make_events(n_each=400, size=10) if e["category"] == "tool_calls"]
    a = render_category(evs, 2000)
    assert a == render_category(evs, 2000) and len(a) <= 2000
    assert "events left out" in a


class FakeLLM:
    model = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, schema: dict[str, Any], **kw: Any) -> LLMReply:
        self.prompts.append((system, user))
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return LLMReply(text, "fake", [], "", [], {"prompt_eval_count": 100, "eval_count": 20},
                        {"num_ctx": 8192})  # fmt: skip


GOOD = json.dumps({"observations": "handover 1 passed fewer findings", "first_error_step":
                   "handover_1", "root_cause_variant": "I2", "confidence": 0.8})  # fmt: skip


def test_prompt_never_contains_labels() -> None:
    llm = FakeLLM([GOOD])
    out = judge.judge_events(llm, make_events(n_each=3, size=20))
    system, user = llm.prompts[0]
    assert out["valid"] and out["pred_variant"] == "I2"
    assert SECRET_RUN not in system + user
    assert "run_id" not in user
    for word in ("ground_truth", "outcome", "manifested", "masked", "twin"):
        assert word not in (system + user).lower()


def test_one_retry_on_invalid_output_then_recorded_failure() -> None:
    llm = FakeLLM(["not json", GOOD])
    assert judge.judge_events(llm, make_events(2, 10))["attempts"] == 2
    bad = FakeLLM(["nope"])
    out = judge.judge_events(bad, make_events(2, 10))
    assert not out["valid"] and out["attempts"] == 2 and out["pred_variant"] == "invalid"


def test_conditions_are_full_minus_each_and_skeleton() -> None:
    cs = judge.conditions()
    assert [n for n, _ in cs] == ["full", "-C1", "-C2", "-C3", "-C4", "-C5", "-C6", "skeleton"]
    assert len(cs[0][1]) == 6 and all(len(k) == 5 for _, k in cs[1:7]) and cs[7][1] == ()


def test_instructions_and_taxonomy_come_first_and_are_never_shortened() -> None:
    small, huge = make_events(2, 10), make_events(80, 20000)
    for evs in (small, huge):
        llm = FakeLLM([GOOD])
        judge.judge_events(llm, evs)
        system, user = llm.prompts[0]
        assert system == judge.system_prompt()  # identical whatever the trace looks like
        assert user.startswith("Log of the run:")  # trace after, never before, the instructions
        for s in judge.STEPS:
            assert f"- {s}: {judge.STEP_TEXT[s]}" in system
        for v in judge.VARIANTS:
            assert judge.VARIANT_TEXT[v] in system


def test_prompt_that_fills_the_context_is_invalid_never_scored() -> None:
    class Full(FakeLLM):
        def chat(self, system: str, user: str, schema: dict[str, Any], **kw: Any) -> LLMReply:
            r = super().chat(system, user, schema, **kw)
            r.timings["prompt_eval_count"] = 8192
            return r

    out = judge.judge_events(Full([GOOD]), make_events(2, 10))
    assert not out["valid"] and out["context_full"] and out["pred_variant"] == "invalid"
    assert out["error"] == "context_full"


def test_v2_schema_writes_observations_before_the_labels() -> None:
    props = list(judge.JudgeVerdictV2.model_json_schema()["properties"])
    assert props == ["observations", "first_error_step", "root_cause_variant", "confidence"]


def test_variant_must_belong_to_the_chosen_step() -> None:
    wrong = json.dumps({"observations": "x", "first_error_step": "reasoning_agent",
                        "root_cause_variant": "E2", "confidence": 0.5})  # fmt: skip
    out = judge.judge_events(FakeLLM([wrong, wrong]), make_events(2, 10))
    assert not out["valid"] and out["attempts"] == 2
    assert judge.judge_events(FakeLLM([wrong, GOOD]), make_events(2, 10))["valid"]


def test_variant_order_is_shuffled_by_key_deterministically() -> None:
    a1 = judge.system_prompt(order_key="run-a|full")
    assert a1 == judge.system_prompt(order_key="run-a|full")
    orders = {judge.system_prompt(order_key=f"run-{i}|full") for i in range(20)}
    assert len(orders) > 10  # position varies across calls

    # the set of lines never changes, only their order
    def lines(p: str) -> list[str]:
        return sorted(x for x in p.splitlines() if x[:3] in ("- E", "- R", "- I"))

    assert lines(a1) == lines(judge.system_prompt(order_key="other"))


def test_order_key_is_not_shown_to_the_model() -> None:
    llm = FakeLLM([GOOD])
    judge.judge_events(llm, make_events(2, 10), order_key=SECRET_RUN + "|full")
    assert SECRET_RUN not in llm.prompts[0][0] + llm.prompts[0][1]


def test_dev_set_is_two_per_variant_on_the_design_side() -> None:
    from collections import Counter
    from pathlib import Path

    dev = json.loads(Path("results/judge_dev_set.json").read_text())
    design = set(json.loads(Path("results/split.json").read_text())["design"])
    assert len(dev["runs"]) == 16
    assert set(Counter(r["variant"] for r in dev["runs"]).values()) == {2}
    assert all(r["run_id"].rsplit("-", 1)[0] in design for r in dev["runs"])
