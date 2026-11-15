"""Shared helpers for live classifier tests under ``tests/live/`` only."""

from __future__ import annotations

import json
import logging
import os
from typing import Callable
from urllib.parse import urlparse

import httpx
import pytest

from decision.coverage import Recommendation
from decision.factcheck import FactCheckVerdict

_LAWYER_KEYS = {"user_facing_text", "questions", "answer"}
_CLOSED_DECISIONS = {
    "allow_answer",
    "reject_insufficient_kb",
    "reject_partial_coverage",
    "clarify",
}
_CLOSED_ACTIONS = {"accept", "rewrite", "refuse"}


def read_key() -> str:
    try:
        value = os.environ["DEEPSEEK_API_KEY"]
    except KeyError:
        pytest.fail("DEEPSEEK_API_KEY is not set")
    if not value.strip():
        pytest.fail("DEEPSEEK_API_KEY is not set")
    return value


def hook_client_send(monkeypatch) -> list[httpx.Request]:
    sent: list[httpx.Request] = []
    original = httpx.Client.send

    def _send(self, request, *args, **kwargs):
        sent.append(request)
        return original(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", _send)
    return sent


def assert_request_target(request: httpx.Request) -> None:
    parsed = urlparse(str(request.url))
    base = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    if not base:
        if parsed.hostname != "api.deepseek.com":
            raise AssertionError(f"request host is {parsed.hostname!r}")
    else:
        expected = urlparse(base if "://" in base else f"https://{base}").hostname
        if parsed.hostname != expected:
            raise AssertionError(f"request host is {parsed.hostname!r}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        raise AssertionError(f"request path is {parsed.path!r}")
    configured = os.environ.get("DEEPSEEK_MODEL", "").strip()
    expected_model = configured or "deepseek-flash"
    body = json.loads(request.content.decode())
    if body.get("model") != expected_model:
        raise AssertionError(f"model is {body.get('model')!r}")


def walk_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def assert_no_lawyer_fields(payload: dict) -> None:
    assert set(walk_keys(payload)).isdisjoint(_LAWYER_KEYS)


def assert_secret_absent(api_key: str, text: str, where: str) -> None:
    if api_key in text:
        raise AssertionError(f"key value was written into the {where}")


def validate_recommendation(result, caplog: logging.LogCaptureFixture, api_key: str) -> None:
    payload = result.model_dump(mode="json")
    Recommendation.model_validate(payload)
    assert payload["suggested_decision"] in _CLOSED_DECISIONS
    assert_no_lawyer_fields(payload)
    serialized = json.dumps(payload)
    assert_secret_absent(api_key, serialized, "result")
    log_text = caplog.text
    for record in caplog.records:
        log_text += "\n" + record.getMessage()
        if record.args:
            log_text += "\n" + repr(record.args)
    assert_secret_absent(api_key, log_text, "log")


def validate_fact_check(result, caplog: logging.LogCaptureFixture, api_key: str) -> None:
    payload = result.model_dump(mode="json")
    FactCheckVerdict.model_validate(payload)
    assert payload["suggested_action"] in _CLOSED_ACTIONS
    assert_no_lawyer_fields(payload)
    serialized = json.dumps(payload)
    assert_secret_absent(api_key, serialized, "result")
    log_text = caplog.text
    for record in caplog.records:
        log_text += "\n" + record.getMessage()
        if record.args:
            log_text += "\n" + repr(record.args)
    assert_secret_absent(api_key, log_text, "log")


def assert_one_request(sent: list[httpx.Request]) -> None:
    assert len(sent) == 1
    assert_request_target(sent[0])


def assert_request_json_shape(request: httpx.Request) -> dict:
    body = json.loads(request.content.decode())
    messages = body.get("messages", [])
    system = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    if "json" not in system.lower():
        raise AssertionError("system prompt must mention json")
    if body.get("temperature") != 0:
        raise AssertionError("temperature must be 0")
    if body.get("thinking", {}).get("type") != "disabled":
        raise AssertionError("thinking must be disabled")
    if body.get("response_format", {}).get("type") != "json_object":
        raise AssertionError("response_format must be json_object")
    for message in messages:
        if message.get("role") != "user":
            continue
        try:
            user = json.loads(message.get("content", "{}"))
        except json.JSONDecodeError:
            user = {}
        for banned in (
            "gold",
            "expected_decision",
            "user_facing_text",
            "questions",
            "answer",
        ):
            if banned in user:
                raise AssertionError(f"user JSON must not include {banned!r}")
    return body


def run_live(
    monkeypatch,
    caplog,
    *,
    call: Callable[[], object],
    validate: Callable[[object, logging.LogCaptureFixture, str], None],
) -> list[httpx.Request]:
    api_key = read_key()
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="decision.llm")
    sent = hook_client_send(monkeypatch)
    result = call()
    assert sent, "enabled mode did not send a classifier request"
    for request in sent:
        assert_request_target(request)
    validate(result, caplog, api_key)
    return sent