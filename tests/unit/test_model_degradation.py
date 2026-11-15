"""Degradation and gate tests for the optional classifier.

``evaluate`` and ``fact_check`` take an injected ``httpx.MockTransport``.
These tests do not open a live socket and do not construct a client of
their own. The key is set with ``monkeypatch.setenv`` only when a request
must be sent.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

import decision
from contracts import Unresolved
from decision import (
    Action,
    Claim,
    ClaimKind,
    CoverageRequest,
    Decision,
    DecisionConfig,
    FactCheckInput,
    FailureReason,
    FindingCode,
    LayerStatus,
    ModelMode,
    ReasonCode,
)
from tests.support import (
    ambiguity,
    candidate,
    citation_check,
    document,
    evidence,
    reference_error,
    retrieval,
    signals,
)

_TEST_KEY = "sk-test-secret"
_QUESTION = "What is the monthly fee?"
_MATCHING_DRAFT = "The monthly fee is 100."
_KEY_NAMES = ("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
_LAWYER_KEYS = {"user_facing_text", "questions", "answer"}
_GOLD_KEYS = {"gold", "gold_label", "expected_label"}
_SOLICITATIONS = (
    "write a question",
    "write an answer",
    "draft a question",
    "draft an answer",
    "ask the lawyer",
    "lawyer-facing question",
    "lawyer-facing answer",
    "question for the lawyer",
    "answer for the lawyer",
    "user_facing_text",
    "напиши вопрос",
    "напишите вопрос",
    "напиши ответ",
    "напишите ответ",
    "вопрос юристу",
    "ответ юристу",
    "сформулируй вопрос",
    "сформулируйте вопрос",
    "сформулируй ответ",
    "сформулируйте ответ",
)
_NEGATIONS = ("do not", "don't", "must not", "never", "without", "не ", "нельзя", "запрещ")

# The core left this rule unverifiable. Expected 100 is in the fragment and
# actual 250 is in the draft, so this object is inside the gates.
_SUPPORTED_RULE_FINDING = {
    "kind": "fact_check",
    "suggested_action": "rewrite",
    "findings": [
        {
            "code": "wrong_number",
            "severity": "high",
            "claim_id": "rule-1",
            "quote": "250",
            "expected": "100",
            "actual": "250",
            "fixable": True,
            "hint": "use the fee from the fragment",
        }
    ],
}

_FAILURE_CALLS = {
    "invalid_json": 1,
    "timeout": 1,
    "connect": 1,
    "empty": 2,
    "status_429": 2,
    "status_500": 2,
}
_FAILURE_REASONS = {
    "invalid_json": FailureReason.model_invalid_json,
    "timeout": FailureReason.model_timeout,
    "connect": FailureReason.model_network,
    "empty": FailureReason.model_invalid_json,
    "status_429": FailureReason.model_network,
    "status_500": FailureReason.model_network,
}


def _on(**overrides: object) -> DecisionConfig:
    values: dict[str, object] = {"model_mode": "on"}
    values.update(overrides)
    return DecisionConfig(**values)


def _off() -> DecisionConfig:
    return DecisionConfig(model_mode="off")


def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _TEST_KEY)


def _drop_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def _coverage(decision: str, reasons: list[str] | None = None, **extra: object) -> dict:
    body: dict[str, object] = {
        "kind": "coverage",
        "suggested_decision": decision,
        "reason_codes": [] if reasons is None else reasons,
    }
    body.update(extra)
    return body


def _fact(action: str, findings: list[dict] | None = None) -> dict:
    return {
        "kind": "fact_check",
        "suggested_action": action,
        "findings": [] if findings is None else findings,
    }


def _finding(code: str, **extra: object) -> dict:
    body: dict[str, object] = {
        "code": code,
        "severity": "medium",
        "claim_id": "fee-1",
        "quote": "100",
        "fixable": False,
        "hint": "model-only-hint",
    }
    body.update(extra)
    return body


def _rule_claim() -> Claim:
    return Claim(claim_id="rule-1", text="250", kind=ClaimKind.rule, quote="250")


def _rule_input(found: object) -> FactCheckInput:
    return FactCheckInput(
        question=_QUESTION,
        answer="The monthly fee is 250.",
        retrieval=found,
        citation_check=citation_check(),
        claims=[_rule_claim()],
        check_rules=False,
    )


def _accept_input() -> FactCheckInput:
    return FactCheckInput(
        question=_QUESTION,
        answer=_MATCHING_DRAFT,
        retrieval=retrieval(),
        citation_check=citation_check(),
    )


def _script(mode: str) -> httpx.MockTransport:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if mode == "invalid_json":
            return httpx.Response(200, content=b"not-json")
        if mode == "timeout":
            raise httpx.TimeoutException("timed out")
        if mode == "connect":
            raise httpx.ConnectError("down")
        if mode == "empty":
            return httpx.Response(200, content=b"")
        if mode == "status_429":
            return httpx.Response(429, content=b"busy")
        if mode == "status_500":
            return httpx.Response(500, content=b"down")
        raise AssertionError(mode)

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def _json_transport(payload: dict) -> httpx.MockTransport:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def _calls(transport: httpx.MockTransport) -> list[httpx.Request]:
    return transport.calls  # type: ignore[attr-defined]


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _dump(result: object) -> dict:
    return result.model_dump(mode="json")


def _assert_no_lawyer_text(result: object) -> None:
    keys = set(_walk_keys(_dump(result)))
    assert keys.isdisjoint(_LAWYER_KEYS)


def _has_code(result: object, code: ReasonCode) -> bool:
    if code in result.reason_codes:
        return True
    return any(gap.code == code for gap in result.gaps)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _solicits(text: str) -> str | None:
    lowered = text.lower()
    for phrase in _SOLICITATIONS:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            window = lowered[max(0, index - 48) : index]
            if not any(negation in window for negation in _NEGATIONS):
                return phrase
            start = index + len(phrase)
    return None


def _assert_request(request: httpx.Request) -> None:
    """Classifier body: json mode, no gold label, no lawyer question or answer."""

    if _TEST_KEY in str(request.url):
        raise AssertionError("test key was written into the request URL")
    raw = request.content.decode()
    if _TEST_KEY in raw:
        raise AssertionError("test key was written into the request body")
    if request.method != "POST":
        raise AssertionError(f"method is {request.method!r}")
    if not request.url.path.rstrip("/").endswith("/chat/completions"):
        raise AssertionError(f"path is {request.url.path!r}")
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer ") or header[len("Bearer ") :] != _TEST_KEY:
        raise AssertionError("classifier request is missing the test bearer token")
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise AssertionError(f"content type is {content_type!r}")
    body = json.loads(raw)
    for key in _walk_keys(body):
        assert key not in _GOLD_KEYS
    assert "tools" not in body
    assert body["temperature"] == 0
    assert body["response_format"]["type"] == "json_object"
    assert body["thinking"]["type"] == "disabled"
    messages = body["messages"]
    system = "\n".join(
        _content_text(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    assert "json" in system.lower()
    combined = "\n".join(
        _content_text(message.get("content", "")) for message in messages
    )
    solicited = _solicits(combined)
    assert solicited is None
    assert "gold_label" not in combined
    assert "expected_label" not in combined


def _assert_logged_clean(caplog: pytest.LogCaptureFixture, mode: str) -> None:
    if _TEST_KEY in caplog.text:
        raise AssertionError("log contains the test key")
    if "Authorization" in caplog.text:
        raise AssertionError("log contains Authorization")
    if mode == "timeout":
        assert "TimeoutException" in caplog.text
    if mode == "status_500":
        assert "500" in caplog.text


def _run(handle: str, transport: httpx.MockTransport, request: object | None = None):
    config = _on()
    if handle == "evaluate":
        coverage = request or CoverageRequest(question=_QUESTION, retrieval=retrieval())
        return decision.evaluate(coverage, config=config, transport=transport)
    fact_input = request or _accept_input()
    return decision.fact_check(fact_input, config=config, transport=transport)


def _evaluate(monkeypatch: pytest.MonkeyPatch, found: object, payload: dict, **request_fields):
    _set_key(monkeypatch)
    transport = _json_transport(payload)
    request = CoverageRequest(question=_QUESTION, retrieval=found, **request_fields)
    result = decision.evaluate(request, config=_on(), transport=transport)
    return result, _calls(transport)


def _check(monkeypatch: pytest.MonkeyPatch, fact_input: object, payload: dict):
    _set_key(monkeypatch)
    transport = _json_transport(payload)
    result = decision.fact_check(fact_input, config=_on(check_rules=False), transport=transport)
    return result, _calls(transport)


def _assert_discarded_decision(result: object, expected: Decision) -> None:
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_schema
    assert result.model_mode == ModelMode.on
    assert result.suggested_decision == expected
    _assert_no_lawyer_text(result)


def _assert_requests(calls: list[httpx.Request], expected: int) -> None:
    assert len(calls) == expected
    for request in calls:
        _assert_request(request)


@pytest.mark.parametrize("handle", ["evaluate", "fact_check"])
@pytest.mark.parametrize(
    "mode",
    ["invalid_json", "timeout", "connect", "empty", "status_429", "status_500"],
)
def test_model_failure_keeps_core(handle, mode, monkeypatch, caplog):
    """Unreadable JSON, timeout, network, empty body, 429, and 500 degrade.

    Empty body, HTTP 429, and HTTP 500 are each retried once. The other
    failures are not retried. The core allow/accept result is kept and the
    call does not raise. HTTP 429 and HTTP 500 degrade as ``model_network``.
    """

    _set_key(monkeypatch)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="decision.llm")
    transport = _script(mode)
    result = _run(handle, transport)
    if handle == "evaluate":
        assert result.suggested_decision == Decision.allow_answer
    else:
        assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == _FAILURE_REASONS[mode]
    assert result.model_mode == ModelMode.on
    _assert_no_lawyer_text(result)
    _assert_logged_clean(caplog, mode)
    _assert_requests(_calls(transport), _FAILURE_CALLS[mode])


@pytest.mark.parametrize("mode", ["invalid_json", "timeout", "connect"])
def test_failure_does_not_silently_allow(mode, monkeypatch, caplog):
    """A core reject stays reject when the model times out, disconnects, or returns junk."""

    _set_key(monkeypatch)
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="decision.llm")
    found = retrieval(signals=signals(topic_match=False))
    request = CoverageRequest(question=_QUESTION, retrieval=found)
    transport = _script(mode)
    result = decision.evaluate(request, config=_on(), transport=transport)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.topic_mismatch in result.reason_codes
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == _FAILURE_REASONS[mode]
    _assert_no_lawyer_text(result)
    _assert_logged_clean(caplog, mode)
    _assert_requests(_calls(transport), 1)


def test_supported_finding_recomputes_unverifiable_rule(monkeypatch):
    """A gated finding fills the unverifiable rule and the action becomes rewrite."""

    _set_key(monkeypatch)
    fact_input = _rule_input(retrieval())
    result, calls = _check(monkeypatch, fact_input, _SUPPORTED_RULE_FINDING)
    assert result.suggested_action == Action.rewrite
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.on
    wrong = [item for item in result.findings if item.code == FindingCode.wrong_number]
    assert len(wrong) == 1
    assert wrong[0].fixable is True
    assert wrong[0].claim_id == "rule-1"
    assert wrong[0].expected == "100"
    assert wrong[0].actual == "250"
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)

    def _refuse_client(*args, **kwargs):
        raise AssertionError("httpx.Client was constructed")

    monkeypatch.setattr(httpx, "Client", _refuse_client)
    core = decision.fact_check(fact_input, config=_off(), transport=None)
    assert core.suggested_action == Action.accept
    assert core.layer_status == LayerStatus.ok
    assert core.failure_reason is None
    assert core.model_mode == ModelMode.off


def test_empty_findings_do_not_mark_rule_checked(monkeypatch):
    """Empty findings and accept are inside the gates and do not check the rule."""

    fact_input = _rule_input(retrieval())
    result, calls = _check(monkeypatch, fact_input, _fact("accept", []))
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert "rule-1" not in result.checked_claims
    assert "rule-1" in result.unverified_claims or any(
        item.claim_id == "rule-1" and item.code == FindingCode.unverifiable
        for item in result.findings
    )
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


@pytest.mark.parametrize(
    ("case", "expected", "code"),
    [
        ("topic_match_false", Decision.reject_insufficient_kb, ReasonCode.topic_mismatch),
        (
            "order_only",
            Decision.reject_insufficient_kb,
            ReasonCode.sufficiency_scale_unavailable,
        ),
        ("low_confidence", Decision.reject_insufficient_kb, ReasonCode.low_confidence),
        (
            "low_relevance",
            Decision.reject_insufficient_kb,
            ReasonCode.insufficient_relevance,
        ),
        ("unresolved", Decision.reject_insufficient_kb, ReasonCode.named_not_found),
        ("reference_errors", Decision.reject_insufficient_kb, ReasonCode.reference_error),
        (
            "ambiguous_edition",
            Decision.reject_insufficient_kb,
            ReasonCode.ambiguous_edition,
        ),
        ("date_conflict", Decision.reject_insufficient_kb, ReasonCode.date_conflict),
        ("empty_retrieval", Decision.reject_insufficient_kb, ReasonCode.missing_facts),
    ],
)
def test_allow_answer_outside_gate_is_discarded(case, expected, code, monkeypatch):
    found = {
        "topic_match_false": lambda: retrieval(signals=signals(topic_match=False)),
        "order_only": lambda: retrieval(
            signals=signals(score_kind="order_only"),
            evidence=[evidence(score_kind="order_only")],
        ),
        "low_confidence": lambda: retrieval(signals=signals(low_confidence=True)),
        "low_relevance": lambda: retrieval(signals=signals(max_relevance=0.1)),
        "unresolved": lambda: retrieval(
            unresolved=Unresolved(contract_numbers=["K-404"])
        ),
        "reference_errors": lambda: retrieval(reference_errors=[reference_error()]),
        "ambiguous_edition": lambda: retrieval(
            signals=signals(applicable_editions=["edition-1", "edition-2"]),
            documents=[
                document(doc_id="doc-1", applicable_edition=True),
                document(
                    doc_id="doc-2",
                    title="Second edition",
                    doc_number="K-2",
                    applicable_edition=True,
                ),
            ],
        ),
        "date_conflict": lambda: retrieval(
            signals=signals(event_date="2025-06-01"),
            documents=[
                document(
                    applicable_edition=True,
                    effective_from="2024-01-01",
                    effective_to="2024-12-31",
                )
            ],
        ),
        "empty_retrieval": lambda: retrieval(documents=[], evidence=[]),
    }[case]()
    result, calls = _evaluate(monkeypatch, found, _coverage("allow_answer"))
    _assert_discarded_decision(result, expected)
    assert result.suggested_decision != Decision.allow_answer
    if case == "empty_retrieval":
        assert result.suggested_decision != Decision.clarify
        assert _has_code(result, code)
    else:
        assert code in result.reason_codes
    _assert_requests(calls, 1)


def test_empty_entity_match_does_not_discard_allow(monkeypatch):
    found = retrieval(signals=signals(entity_match=[]), unresolved=Unresolved())
    result, calls = _evaluate(monkeypatch, found, _coverage("allow_answer"))
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.on
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


def test_clarify_candidate_absent_from_ambiguity_is_discarded(monkeypatch):
    payload = _coverage(
        "clarify",
        ["ambiguous_entity"],
        missing_fields=["contract"],
        candidate_doc_ids=["doc-missing"],
    )
    result, calls = _evaluate(monkeypatch, retrieval(), payload)
    _assert_discarded_decision(result, Decision.allow_answer)
    assert result.suggested_decision != Decision.clarify
    assert "doc-missing" not in json.dumps(_dump(result))
    _assert_requests(calls, 1)


def test_clarify_on_empty_retrieval_is_discarded(monkeypatch):
    payload = _coverage(
        "clarify",
        ["missing_identifier"],
        missing_fields=["contract"],
        candidate_doc_ids=["doc-1"],
    )
    result, calls = _evaluate(monkeypatch, retrieval(documents=[], evidence=[]), payload)
    _assert_discarded_decision(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.clarify
    _assert_requests(calls, 1)


def test_partial_coverage_without_policy_is_discarded(monkeypatch):
    payload = _coverage("reject_partial_coverage", ["partial_clause_coverage"])
    result, calls = _evaluate(monkeypatch, retrieval(), payload)
    _assert_discarded_decision(result, Decision.allow_answer)
    _assert_requests(calls, 1)


def test_partial_coverage_while_clarification_holds_is_discarded(monkeypatch):
    """Clarification conditions hold, so a model partial reject is not accepted.

    Two ambiguity candidates, margin 0.01, two document numbers, and a
    question that does not name a contract. Until the questioner is real the
    core may be ``reject_partial_coverage`` or ``allow_answer``. The model
    answer is discarded: ``failure_reason`` is ``model_schema``. A model
    partial reject must not drop uncovered clause B.
    """

    found = retrieval(
        signals=signals(
            margin_top1_top2=0.01,
            distinct_docs=2,
            applicable_editions=["edition-1"],
        ),
        documents=[
            document(
                doc_id="doc-1",
                doc_number="K-1",
                applicable_edition=True,
                matched_clauses=["A"],
            ),
            document(
                doc_id="doc-2",
                title="Other rule",
                doc_number="K-2",
                applicable_edition=False,
                matched_clauses=["A"],
            ),
        ],
        ambiguity=[
            ambiguity(
                "contract",
                [
                    candidate(
                        doc_id="doc-1",
                        doc_number="K-1",
                        label="North Warehouse LLC",
                    ),
                    candidate(
                        doc_id="doc-2",
                        title="Other rule",
                        doc_number="K-2",
                        label="South Warehouse LLC",
                    ),
                ],
            )
        ],
    )
    payload = _coverage("reject_partial_coverage", ["partial_clause_coverage"])
    result, calls = _evaluate(
        monkeypatch,
        found,
        payload,
        question_kind="fee",
        policy={"fee": ["A", "B"]},
    )
    assert result.failure_reason == FailureReason.model_schema
    assert result.layer_status == LayerStatus.degraded
    assert result.model_mode == ModelMode.on
    kept_clause = "B" in list(result.coverage.missing)
    if result.suggested_decision == Decision.reject_partial_coverage:
        assert kept_clause
    if result.suggested_decision == Decision.clarify:
        assert kept_clause
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


def test_reason_code_outside_closed_list_is_discarded(monkeypatch):
    result, calls = _evaluate(
        monkeypatch, retrieval(), _coverage("allow_answer", ["no_chunks"])
    )
    _assert_discarded_decision(result, Decision.allow_answer)
    assert "no_chunks" not in json.dumps(_dump(result))
    _assert_requests(calls, 1)


@pytest.mark.parametrize(
    "extra",
    [
        {"user_facing_text": "Pay the fee now."},
        {"question": "Which contract applies?"},
    ],
)
def test_lawyer_text_field_is_discarded(extra, monkeypatch):
    result, calls = _evaluate(monkeypatch, retrieval(), _coverage("allow_answer", **extra))
    _assert_discarded_decision(result, Decision.allow_answer)
    dumped = json.dumps(_dump(result))
    assert "Pay the fee now." not in dumped
    assert "Which contract applies?" not in dumped
    _assert_requests(calls, 1)


def test_accept_without_citation_check_is_discarded(monkeypatch):
    fact_input = FactCheckInput(
        question=_QUESTION,
        answer=_MATCHING_DRAFT,
        retrieval=retrieval(),
        citation_check=None,
    )
    result, calls = _check(monkeypatch, fact_input, _fact("accept", []))
    assert result.suggested_action == Action.rewrite
    assert result.suggested_action != Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_schema
    assert any(item.code == FindingCode.citation_check_missing for item in result.findings)
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


def test_fixable_without_replacement_is_discarded(monkeypatch):
    found = retrieval(
        documents=[
            document(
                doc_number="",
                effective_from=None,
                effective_to=None,
                parties=["North Warehouse LLC"],
            )
        ],
        evidence=[
            evidence(
                text="The monthly fee is described in the schedule.",
                section="fee",
            )
        ],
    )
    fact_input = FactCheckInput(
        question=_QUESTION,
        answer="The monthly fee is 250.",
        retrieval=found,
        citation_check=citation_check(),
        claims=[
            Claim(
                claim_id="num-1",
                text="250",
                kind=ClaimKind.number,
                quote="250",
                subject="monthly fee",
                value="250",
            )
        ],
        check_rules=False,
    )
    payload = _fact(
        "rewrite",
        [
            _finding(
                "wrong_number",
                severity="high",
                claim_id="num-1",
                quote="250",
                expected="999",
                actual="250",
                fixable=True,
            )
        ],
    )
    result, calls = _check(monkeypatch, fact_input, payload)
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_schema
    assert FindingCode.wrong_number not in [item.code for item in result.findings]
    assert all(item.fixable is False for item in result.findings)
    assert any(item.code == FindingCode.unverifiable for item in result.findings)
    assert "model-only-hint" not in json.dumps(_dump(result))
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


def test_wrong_number_without_subject_support_is_discarded(monkeypatch):
    fact_input = FactCheckInput(
        question=_QUESTION,
        answer="The zirconium quota is 250.",
        retrieval=retrieval(),
        citation_check=citation_check(),
        claims=[
            Claim(
                claim_id="zq-1",
                text="The zirconium quota is 250.",
                kind=ClaimKind.number,
                quote="250",
                subject="zirconium quota",
                value="250",
            )
        ],
    )
    payload = _fact(
        "rewrite",
        [
            _finding(
                "wrong_number",
                severity="high",
                claim_id="zq-1",
                quote="250",
                expected="100",
                actual="250",
                fixable=True,
            )
        ],
    )
    result, calls = _check(monkeypatch, fact_input, payload)
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_schema
    codes = [item.code for item in result.findings]
    assert FindingCode.unsupported_claim in codes
    assert FindingCode.wrong_number not in codes
    assert "model-only-hint" not in json.dumps(_dump(result))
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


@pytest.mark.parametrize(
    "code",
    [
        "citation_not_in_context",
        "citation_check_missing",
        "missing_citation",
        "unsupported_claim",
        "unverifiable",
    ],
)
def test_high_severity_on_non_refusal_code_is_discarded(code, monkeypatch):
    payload = _fact("refuse", [_finding(code, severity="high", fixable=False)])
    result, calls = _check(monkeypatch, _accept_input(), payload)
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_schema
    assert all(item.severity.value != "high" for item in result.findings)
    assert "model-only-hint" not in json.dumps(_dump(result))
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


def test_rule_wrong_number_without_support_is_discarded(monkeypatch):
    """wrong_number on an unchecked rule is discarded when 100 is not in the materials."""

    found = retrieval(
        evidence=[evidence(text="The schedule names the fee.", section="fee")]
    )
    fact_input = _rule_input(found)
    result, calls = _check(monkeypatch, fact_input, _SUPPORTED_RULE_FINDING)
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_schema
    codes = [item.code for item in result.findings]
    assert FindingCode.wrong_number not in codes
    assert FindingCode.unverifiable in codes
    _assert_no_lawyer_text(result)
    _assert_requests(calls, 1)


def test_model_mode_off_evaluate_does_not_construct_client(monkeypatch):
    def _refuse_client(*args, **kwargs):
        raise AssertionError("httpx.Client was constructed")

    monkeypatch.setattr(httpx, "Client", _refuse_client)
    _drop_keys(monkeypatch)
    result = decision.evaluate(
        CoverageRequest(question=_QUESTION, retrieval=retrieval()),
        config=_off(),
        transport=None,
    )
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off
    _assert_no_lawyer_text(result)


def test_model_mode_off_fact_check_does_not_construct_client(monkeypatch):
    def _refuse_client(*args, **kwargs):
        raise AssertionError("httpx.Client was constructed")

    monkeypatch.setattr(httpx, "Client", _refuse_client)
    _drop_keys(monkeypatch)
    result = decision.fact_check(_accept_input(), config=_off(), transport=None)
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off
    _assert_no_lawyer_text(result)


def test_model_on_without_key_evaluate_sends_nothing(monkeypatch):
    _drop_keys(monkeypatch)
    transport = _json_transport(_coverage("allow_answer"))
    result = decision.evaluate(
        CoverageRequest(question=_QUESTION, retrieval=retrieval()),
        config=_on(),
        transport=transport,
    )
    assert _calls(transport) == []
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_unconfigured
    assert result.model_mode == ModelMode.on
    _assert_no_lawyer_text(result)


def test_model_on_without_key_fact_check_sends_nothing(monkeypatch):
    _drop_keys(monkeypatch)
    transport = _json_transport(_fact("accept", []))
    result = decision.fact_check(_accept_input(), config=_on(), transport=transport)
    assert _calls(transport) == []
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.degraded
    assert result.failure_reason == FailureReason.model_unconfigured
    assert result.model_mode == ModelMode.on
    _assert_no_lawyer_text(result)


def test_evaluate_request_has_no_gold_and_asks_for_json(monkeypatch):
    result, calls = _evaluate(monkeypatch, retrieval(), _coverage("allow_answer"))
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status == LayerStatus.ok
    _assert_requests(calls, 1)
    user = "\n".join(
        _content_text(message.get("content", ""))
        for message in json.loads(calls[0].content.decode())["messages"]
        if message.get("role") == "user"
    )
    assert _QUESTION in user


def test_fact_check_request_has_no_gold_and_asks_for_json(monkeypatch):
    result, calls = _check(monkeypatch, _accept_input(), _fact("accept", []))
    assert result.suggested_action == Action.accept
    assert result.layer_status == LayerStatus.ok
    _assert_requests(calls, 1)
    user = "\n".join(
        _content_text(message.get("content", ""))
        for message in json.loads(calls[0].content.decode())["messages"]
        if message.get("role") == "user"
    )
    assert _MATCHING_DRAFT in user


def test_valid_model_premise_errors_merge_without_changing_allow(monkeypatch):
    """Gated premise errors from the model are merged; allow_answer stays allow."""

    _set_key(monkeypatch)
    found = retrieval(evidence=[evidence(text="The monthly fee is 100.")])
    request = CoverageRequest(question="The monthly fee is 250.", retrieval=found)
    payload = _coverage(
        "allow_answer",
        ["premise_error"],
        premise_errors=[
            {
                "statement": "fee",
                "evidence_quote": "100",
                "expected": "100",
                "actual": "250",
            }
        ],
    )
    transport = _json_transport(payload)
    result = decision.evaluate(request, config=_on(), transport=transport)
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status == LayerStatus.ok
    assert ReasonCode.premise_error in result.reason_codes
    assert result.premise_errors
    assert result.premise_errors[0].expected == "100"
    assert result.premise_errors[0].actual == "250"
    _assert_no_lawyer_text(result)
