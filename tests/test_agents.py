"""Agent tests with a fake LLM: no Ollama needed."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from logablation.agents import (
    ActionPlan,
    Diagnosis,
    EvidenceBundle,
    LLMError,
    LLMReply,
    OllamaClient,
    action_agent,
    evidence_agent,
    generate_structured,
    reasoning_agent,
)
from logablation.agents.agents import plan_services
from logablation.agents.llm import NUM_CTX, render_prompt
from logablation.agents.schemas import EvidenceDraft
from logablation.scenarios.generator import generate_scenario
from logablation.scenarios.taxonomy import ACTIONS, INCIDENT_TYPES, SERVICES
from logablation.tools import ToolContext, build_ticket


@dataclass
class FakeLLM:
    """Replays scripted replies (str or Exception) and records every request."""

    script: list[str | Exception]
    model: str = "fake"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMReply:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "schema": schema,
                "temperature": temperature,
                "seed": seed,
            }
        )
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return LLMReply(
            text=item,
            model=self.model,
            messages=msgs,
            prompt=render_prompt(msgs),
            logprobs=[{"token": "x", "logprob": -0.1, "bytes": [120]}],
            timings={"prompt_eval_count": 10, "eval_count": 5},
            options={"num_ctx": NUM_CTX},
        )


def _diag(**kw: Any) -> str:
    base = {
        "root_cause": "bad_deploy",
        "suspected_service": "checkout",
        "reasoning_summary": "x",
        "confidence": 0.8,
        "evidence_used": ["f1"],
    }
    return json.dumps(base | kw)


BUNDLE = {
    "incident_id": "INC-1",
    "affected_service": "checkout",
    "time_window": ["2025-03-10T09:00:00Z", "2025-03-10T09:25:00Z"],
    "findings": [{"id": "f1", "source": "get_deploys", "value": "deploy d1 on checkout"}],
    "evidence_confidence": 0.9,
}


# ---- schemas -------------------------------------------------------------------------------


def test_enums_are_exactly_the_taxonomy() -> None:
    props = Diagnosis.model_json_schema()["$defs"]
    assert set(props["RootCause"]["enum"]) == set(INCIDENT_TYPES)
    assert set(ActionPlan.model_json_schema()["$defs"]["Action"]["enum"]) == set(ACTIONS.values())
    assert set(props["Service"]["enum"]) == set(SERVICES)


@pytest.mark.parametrize(
    "bad", [{"root_cause": "disk_full"}, {"confidence": 1.5}, {"suspected_service": "billing"}]
)
def test_diagnosis_rejects_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Diagnosis.model_validate_json(_diag(**bad))


def test_bundle_has_the_spec_fields() -> None:
    assert list(EvidenceBundle.model_fields) == [
        "incident_id",
        "affected_service",
        "time_window",
        "findings",
        "evidence_confidence",
    ]
    assert list(Diagnosis.model_fields) == [
        "root_cause",
        "suspected_service",
        "reasoning_summary",
        "confidence",
        "evidence_used",
    ]
    assert list(ActionPlan.model_fields) == [
        "action",
        "target_service",
        "justification",
        "confidence",
    ]


# ---- retries -------------------------------------------------------------------------------


def test_valid_first_try_makes_one_call() -> None:
    llm = FakeLLM([_diag()])
    r = generate_structured(llm, Diagnosis, "s", "u")
    assert r.ok and len(r.attempts) == 1 and len(llm.calls) == 1
    assert r.raw_text == _diag() and r.prompt.startswith("[system]\ns") and r.logprobs


def test_invalid_then_valid_retries_once_with_feedback() -> None:
    llm = FakeLLM(["not json", _diag()])
    r = generate_structured(llm, Diagnosis, "s", "original user", seed=3)
    assert r.ok and len(r.attempts) == 2
    assert r.attempts[0].error and r.attempts[1].error is None
    assert "original user" in llm.calls[1]["user"] and "rejected" in llm.calls[1]["user"]
    assert llm.calls[1]["seed"] != llm.calls[0]["seed"]


def test_two_failures_are_recorded_not_raised() -> None:
    llm = FakeLLM([_diag(root_cause="nope"), "{}"])
    r = generate_structured(llm, Diagnosis, "s", "u")
    assert not r.ok and r.parsed is None and len(r.attempts) == 2 and r.error
    assert len(llm.calls) == 2  # exactly one retry


def test_transport_error_is_a_recorded_failure() -> None:
    r = generate_structured(FakeLLM([LLMError("down")]), Diagnosis, "s", "u")
    assert not r.ok and r.error is not None and "llm_error" in r.error


def test_unknown_evidence_ids_trigger_retry() -> None:
    llm = FakeLLM([_diag(evidence_used=["f9"]), _diag()])
    r = reasoning_agent(BUNDLE, llm)
    assert r.ok and len(r.llm.attempts) == 2
    assert "f9" in (r.llm.attempts[0].error or "")


# ---- prompt rendering ----------------------------------------------------------------------


def _scenario_parts(seed: int = 1002):
    sc = generate_scenario(seed)
    return sc, build_ticket(sc), ToolContext(sc)


def test_evidence_agent_plan_prompt_and_bundle() -> None:
    sc, ticket, tools = _scenario_parts()
    draft = {
        "affected_service": "checkout",
        "evidence_confidence": 0.7,
        "findings": [
            {"source": "get_metrics", "value": "pool at 50"},
            {"source": "get_logs", "value": "timeouts"},
        ],
    }
    llm = FakeLLM([json.dumps(draft)])
    res = evidence_agent(ticket, tools, llm)
    bundle = res.output
    assert isinstance(bundle, EvidenceBundle)
    assert bundle.incident_id == ticket.incident_id
    assert bundle.time_window == ticket.time_window
    assert [f.id for f in bundle.findings] == ["f1", "f2"]
    names = [c["tool"] for c in res.tool_calls]
    assert names[:2] == ["get_topology", "get_alerts"] and names[-1] == "get_deploys"
    services = plan_services(ticket, res.tool_calls[0]["result"])
    assert services[0] == ticket.symptom_service
    assert [c["args"]["service"] for c in res.tool_calls if c["tool"] == "get_metrics"] == services
    user = llm.calls[0]["user"]
    for c in res.tool_calls:
        assert f"## {c['tool']}(" in user
    assert ticket.summary in user and ticket.time_window[0] in user
    # the ticket must not leak ground truth
    gt = sc.ground_truth
    assert gt.correct_action not in user
    assert llm.calls[0]["schema"] == EvidenceDraft.model_json_schema()


def test_plan_services_follows_dependencies_only() -> None:
    sc, ticket, tools = _scenario_parts()
    topo = tools.get_topology()
    order = plan_services(ticket, topo)
    assert order[0] == ticket.symptom_service and len(order) == len(set(order))
    leaf = ticket.model_copy(update={"symptom_service": "payment-db"})
    assert plan_services(leaf, topo) == ["payment-db"]


def test_reasoning_prompt_uses_only_the_bundle() -> None:
    llm = FakeLLM([_diag()])
    reasoning_agent(BUNDLE, llm)
    user = llm.calls[0]["user"]
    assert "deploy d1 on checkout" in user and "INC-1" in user
    assert "Evidence bundle:" in user and "Diagnosis:" not in user


def test_reasoning_accepts_a_partial_dict_from_a_handover_fault() -> None:
    partial = {k: v for k, v in BUNDLE.items() if k != "time_window"}
    llm = FakeLLM([_diag()])
    assert reasoning_agent(partial, llm).ok
    assert "time_window" not in llm.calls[0]["user"]


def test_action_prompt_uses_only_the_diagnosis() -> None:
    plan = {
        "action": "rollback_deploy",
        "target_service": "checkout",
        "justification": "x",
        "confidence": 0.9,
    }
    llm = FakeLLM([json.dumps(plan)])
    diag = Diagnosis.model_validate_json(_diag())
    res = action_agent(diag, llm)
    assert isinstance(res.output, ActionPlan)
    user = llm.calls[0]["user"]
    assert "Diagnosis:" in user and "bad_deploy" in user and "Evidence bundle" not in user
    # runbook lives in the system prompt and covers every root cause
    assert all(a in llm.calls[0]["system"] for a in ACTIONS.values())


def test_agent_failure_returns_none_output() -> None:
    res = action_agent({"root_cause": "bad_deploy"}, FakeLLM(["garbage"]))
    assert not res.ok and res.output is None and res.llm.error


# ---- client request shape ------------------------------------------------------------------


class _Capture(OllamaClient):
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent = (path, payload)
        return {
            "message": {"content": "{}"},
            "logprobs": [{"token": "{", "logprob": -0.1}],
            "prompt_eval_count": 3,
            "eval_count": 1,
        }


def test_client_always_sends_ctx_seed_format_and_logprobs() -> None:
    c = _Capture()
    schema: Callable[..., Any] = Diagnosis.model_json_schema
    rep = c.chat("sys", "usr", schema())
    path, p = c.sent
    assert path == "/api/chat"
    assert p["options"]["num_ctx"] == 8192 and p["options"]["temperature"] == 0.0
    assert p["options"]["seed"] == 7 and p["logprobs"] is True and p["format"] == schema()
    assert p["stream"] is False
    assert rep.prompt == "[system]\nsys\n\n[user]\nusr" and rep.logprobs[0]["token"] == "{"
    assert rep.timings["prompt_eval_count"] == 3 and "wall_s" in rep.timings


def test_client_refuses_oversized_context_and_unload_uses_keep_alive_zero() -> None:
    with pytest.raises(ValueError):
        OllamaClient(num_ctx=131072)
    c = _Capture()
    c.unload("qwen2.5:7b")
    path, p = c.sent
    assert path == "/api/generate" and p["keep_alive"] == 0 and p["model"] == "qwen2.5:7b"
    assert p["options"]["num_ctx"] == 8192
