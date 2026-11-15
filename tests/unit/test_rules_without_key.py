"""Coverage and draft check return with the classifier off and no API key.

A separate live check uses a key already in the environment and does not
replace this test. This module does not call that live check.
"""

from __future__ import annotations

import json
import socket

import pytest

import decision
from decision import (
    Action,
    CoverageRequest,
    Decision,
    DecisionConfig,
    FactCheckInput,
    LayerStatus,
    ModelMode,
)
from tests.support import citation_check, retrieval

_KEY_NAMES = ("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
_LAWYER_KEYS = {"user_facing_text", "questions", "answer"}
_QUESTION = "What is the monthly fee?"
_MATCHING_DRAFT = "The monthly fee is 100."


def _refuse_socket(*args, **kwargs):
    raise AssertionError("the test opened a socket")


def _drop_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delete the key variables. Do not read them except to delete them."""

    for name in _KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def _deleted_values(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Values MonkeyPatch removed. Used only to prove they were not copied."""

    saved: list[str] = []
    for _mapping, name, old in getattr(monkeypatch, "_setitem", ()):
        if name in _KEY_NAMES and isinstance(old, str) and old:
            saved.append(old)
    return saved


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _assert_result_has_no_key(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    payload = json.dumps(result.model_dump(mode="json"))
    if "sk-test-secret" in payload or "Authorization" in payload:
        raise AssertionError("key material was written into the result")
    for secret in _deleted_values(monkeypatch):
        if secret in payload:
            raise AssertionError("key value was written into the result")
    assert set(_walk_keys(json.loads(payload))).isdisjoint(_LAWYER_KEYS)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setattr(socket, "socket", _refuse_socket)
    _drop_keys(monkeypatch)
    return monkeypatch


def _finished_draft() -> FactCheckInput:
    return FactCheckInput(
        question=_QUESTION,
        answer=_MATCHING_DRAFT,
        retrieval=retrieval(),
        citation_check=citation_check(),
    )


def test_evaluate_returns_without_a_key(offline):
    result = decision.evaluate(
        CoverageRequest(question=_QUESTION, retrieval=retrieval()),
        config=DecisionConfig(model_mode="off"),
        transport=None,
    )
    assert result is not None
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off
    _assert_result_has_no_key(offline, result)


def test_fact_check_returns_without_a_key(offline):
    result = decision.fact_check(
        _finished_draft(),
        config=DecisionConfig(model_mode="off"),
        transport=None,
    )
    assert result is not None
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off
    _assert_result_has_no_key(offline, result)
