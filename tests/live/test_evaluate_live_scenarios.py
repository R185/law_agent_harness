"""Live coverage scenarios at ``model_mode=on``."""

from __future__ import annotations

import decision
from contracts import Unresolved
from decision import CoverageRequest, Decision, DecisionConfig, MissingField
from tests.live.live_helpers import assert_one_request, run_live, validate_recommendation
from tests.support import ambiguity, candidate, document, evidence, reference_error, retrieval, signals

_QUESTION = "What is the monthly fee?"


def _evaluate(found, *, question=_QUESTION):
    return decision.evaluate(
        CoverageRequest(question=question, retrieval=found),
        config=DecisionConfig(model_mode="on"),
    )


def test_baseline_sufficient_retrieval(monkeypatch, caplog):
    sent = run_live(
        monkeypatch,
        caplog,
        call=lambda: _evaluate(retrieval()),
        validate=validate_recommendation,
    )
    assert_one_request(sent)


def test_topic_mismatch_high_relevance_not_allow(monkeypatch, caplog):
    found = retrieval(signals=signals(topic_match=False, max_relevance=0.95))

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.allow_answer
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_order_only_not_allow(monkeypatch, caplog):
    found = retrieval(signals=signals(score_kind="order_only", max_relevance=0.99))

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.allow_answer
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_empty_retrieval_not_clarify(monkeypatch, caplog):
    found = retrieval(documents=[], evidence=[], signals=signals(distinct_docs=0))

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.clarify
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_reference_errors_not_allow(monkeypatch, caplog):
    found = retrieval(reference_errors=[reference_error(kind="unresolved_doc")])

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.allow_answer
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_unresolved_not_allow(monkeypatch, caplog):
    found = retrieval(unresolved=Unresolved(contract_numbers=["CN-1"]))

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.allow_answer
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_low_confidence_not_allow(monkeypatch, caplog):
    found = retrieval(signals=signals(low_confidence=True))

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.allow_answer
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_clarification_eligible_shape(monkeypatch, caplog):
    docs = [
        document(doc_id="doc-1", doc_number="A-1"),
        document(doc_id="doc-2", doc_number="B-2"),
    ]
    choices = [
        candidate(doc_id="doc-1", label="North", doc_number="A-1"),
        candidate(doc_id="doc-2", label="South", doc_number="B-2"),
    ]
    found = retrieval(
        documents=docs,
        signals=signals(margin_top1_top2=0.01, distinct_docs=2),
        ambiguity=[ambiguity("counterparty", choices)],
    )

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision in {
            Decision.clarify,
            Decision.reject_insufficient_kb,
            Decision.reject_partial_coverage,
            Decision.allow_answer,
        }
        assert len(result.missing_fields) <= 3
        for item in result.candidates:
            known = {
                cand.doc_id
                for group in found.ambiguity
                for cand in group.candidates
            }
            assert item.doc_id in known
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_ambiguous_editions_not_allow(monkeypatch, caplog):
    docs = [
        document(doc_id="doc-1", applicable_edition=True),
        document(doc_id="doc-2", doc_number="KB-2", applicable_edition=True),
    ]
    found = retrieval(
        documents=docs,
        signals=signals(
            applicable_editions=["edition-1", "edition-2"],
            distinct_docs=2,
            event_date="2024-06-01",
        ),
    )

    def _call():
        result = _evaluate(found)
        assert result.suggested_decision != Decision.allow_answer
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)