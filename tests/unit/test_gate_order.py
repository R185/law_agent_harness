"""Ordered coverage rules for ``decision.evaluate``.

``evaluate`` is resolved inside the test body so collection succeeds before
the function exists. Rules 0 and 1 fail the layer and leave reason codes
empty. From rule 2, the earliest match sets the decision. Later matches
still add their reason codes, and none of them opens ``allow_answer``.
Clarification stays closed: one document, margin 0.2, and empty ambiguity,
except where a test uses several editions.
"""

from __future__ import annotations

import logging

import pytest

from decision import (
    CoverageRequest,
    Decision,
    DecisionConfig,
    FailureReason,
    LayerStatus,
    ModelMode,
    ReasonCode,
)
from tests.support import document, evidence, reference_error, retrieval, signals

QUESTION = "What is the monthly fee?"
EDITION_QUESTION = "What is the monthly fee under KA-EDITION?"
_SECRET = "sk-synthetic-not-a-real-key"
_SIGNAL_FIELDS = (
    "low_confidence",
    "max_relevance",
    "mean_relevance",
    "mean_top3_relevance",
    "margin_top1_top2",
    "distinct_docs",
    "entity_match",
    "topic_match",
    "score_kind",
    "dense_used",
    "rerank_used",
    "event_date",
    "applicable_editions",
)


@pytest.fixture(autouse=True)
def _model_off_and_no_key(monkeypatch):
    monkeypatch.setenv("DECISION_MODEL_MODE", "off")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def _request(found, *, question=QUESTION, question_kind=None, policy=None):
    return CoverageRequest(
        question=question,
        retrieval=found,
        question_kind=question_kind,
        policy=policy,
    )


def _evaluate(request, config=None):
    import decision

    return decision.evaluate(request, config=config)


def _dump(request):
    return request.model_dump(mode="json")


def _assert_no_prose(result):
    dumped = result.model_dump()
    assert "user_facing_text" not in dumped
    assert "questions" not in dumped
    assert "answer" not in dumped


def _assert_failed(result, reason):
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert result.layer_status == LayerStatus.failed
    assert result.failure_reason == reason
    assert list(result.reason_codes) == []
    assert result.model_mode == ModelMode.off
    _assert_no_prose(result)


def _assert_ok(result, decision):
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off
    assert result.suggested_decision == decision
    assert len(list(result.reason_codes)) == len(set(result.reason_codes))
    _assert_no_prose(result)


def _contested():
    """Readable retrieval that trips every later refusal except scale and editions."""

    return retrieval(
        signals=signals(
            topic_match=False,
            score_kind="semantic",
            max_relevance=0.1,
            low_confidence=True,
            margin_top1_top2=0.2,
        ),
        unresolved={"contract_numbers": ["K-9"]},
        ambiguity=[],
    )


def _two_editions(start, end):
    docs = [
        document(
            doc_id="doc-1",
            doc_number="KA-EDITION",
            applicable_edition=True,
            effective_from=start,
            effective_to=end,
        ),
        document(
            doc_id="doc-2",
            doc_number="KB-EDITION",
            title="Addendum",
            applicable_edition=True,
            effective_from=start,
            effective_to=end,
        ),
    ]
    return retrieval(
        documents=docs,
        signals=signals(
            topic_match=True,
            score_kind="semantic",
            max_relevance=0.8,
            low_confidence=False,
            margin_top1_top2=0.2,
            distinct_docs=2,
            event_date="2024-06-01",
            applicable_editions=["edition-1", "edition-2"],
        ),
        ambiguity=[],
        reference_errors=[],
    )


def _logged(caplog, reason, secret):
    records = [rec for rec in caplog.records if rec.name == "decision.coverage"]
    assert records, "expected a decision.coverage log record"
    rendered = "\n".join(rec.getMessage() for rec in records)
    assert reason in rendered
    whole = "\n".join(rec.getMessage() for rec in caplog.records)
    assert secret not in whole
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_relevance_semantic", -0.1),
        ("min_relevance_semantic", "nope"),
        ("min_relevance_semantic", "0.5"),
        ("min_relevance_semantic", 1.1),
        ("min_relevance_semantic", True),
        ("min_relevance_semantic", False),
        ("min_relevance_semantic", float("nan")),
        ("min_relevance_semantic", float("inf")),
        ("min_relevance_semantic", float("-inf")),
        ("min_relevance_lexical", -0.1),
        ("min_relevance_lexical", "nope"),
        ("min_relevance_lexical", 1.1),
        ("min_relevance_lexical", True),
        ("min_relevance_lexical", False),
    ],
)
def test_invalid_sufficiency_threshold_is_config_invalid(field, value):
    request = _request(
        _contested(),
        question_kind="fees",
        policy={"fees": ["A", "B"]},
    )
    result = _evaluate(request, config=DecisionConfig(**{field: value}))
    _assert_failed(result, FailureReason.config_invalid)


def test_threshold_zero_stays_on_the_allow_path():
    result = _evaluate(
        _request(retrieval()),
        config=DecisionConfig(min_relevance_semantic=0, min_relevance_lexical=0),
    )
    _assert_ok(result, Decision.allow_answer)
    assert result.failure_reason != FailureReason.config_invalid
    assert result.reason_codes == []


def test_threshold_zero_does_not_reject_equal_relevance():
    found = retrieval(signals=signals(score_kind="semantic", max_relevance=0))
    result = _evaluate(
        _request(found),
        config=DecisionConfig(min_relevance_semantic=0),
    )
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.insufficient_relevance not in result.reason_codes
    assert result.failure_reason != FailureReason.config_invalid


def test_threshold_one_is_valid_when_relevance_equals_it():
    found = retrieval(signals=signals(score_kind="semantic", max_relevance=1))
    result = _evaluate(
        _request(found),
        config=DecisionConfig(min_relevance_semantic=1),
    )
    _assert_ok(result, Decision.allow_answer)
    assert result.failure_reason != FailureReason.config_invalid
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_threshold_zero_does_not_mask_a_later_rule():
    result = _evaluate(
        _request(_contested(), question_kind="fees", policy={"fees": ["A", "B"]}),
        config=DecisionConfig(min_relevance_semantic=0, min_relevance_lexical=0),
    )
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.failure_reason != FailureReason.config_invalid
    assert ReasonCode.topic_mismatch in result.reason_codes
    assert result.suggested_decision != Decision.allow_answer


@pytest.mark.parametrize(
    ("where", "field", "value"),
    [
        ("evidence", "relevance", -0.01),
        ("evidence", "relevance", 1.5),
        ("signals", "max_relevance", -0.01),
        ("signals", "max_relevance", 1.5),
        ("signals", "mean_relevance", 1.2),
    ],
)
def test_relevance_outside_unit_interval_is_contract_invalid(where, field, value):
    payload = _dump(_request(_contested(), question_kind="fees", policy={"fees": ["A", "B"]}))
    if where == "evidence":
        payload["retrieval"]["evidence"][0][field] = value
    else:
        payload["retrieval"]["signals"][field] = value
    result = _evaluate(payload)
    _assert_failed(result, FailureReason.contract_invalid)


@pytest.mark.parametrize("field", _SIGNAL_FIELDS)
def test_missing_signal_is_contract_invalid(field):
    payload = _dump(_request(_contested()))
    del payload["retrieval"]["signals"][field]
    result = _evaluate(payload)
    _assert_failed(result, FailureReason.contract_invalid)


def test_missing_signals_object_is_contract_invalid():
    payload = _dump(_request(retrieval()))
    del payload["retrieval"]["signals"]
    result = _evaluate(payload)
    _assert_failed(result, FailureReason.contract_invalid)


@pytest.mark.parametrize("where", ["signals", "evidence"])
def test_mystery_score_kind_is_contract_invalid(where):
    payload = _dump(_request(_contested(), question_kind="fees", policy={"fees": ["A", "B"]}))
    if where == "signals":
        payload["retrieval"]["signals"]["score_kind"] = "mystery"
    else:
        payload["retrieval"]["evidence"][0]["score_kind"] = "mystery"
    result = _evaluate(payload)
    _assert_failed(result, FailureReason.contract_invalid)


@pytest.mark.parametrize(
    "retrieval_payload",
    [
        {"score": 4},
        [{"score": 4}],
        {"documents": [{"score": 4}]},
    ],
)
def test_score_only_payload_is_contract_invalid(retrieval_payload):
    result = _evaluate({"question": QUESTION, "retrieval": retrieval_payload})
    _assert_failed(result, FailureReason.contract_invalid)


def test_unknown_signal_field_is_ignored():
    payload = _dump(_request(retrieval()))
    payload["retrieval"]["signals"]["bundle_note"] = "ignored"
    result = _evaluate(payload)
    _assert_ok(result, Decision.allow_answer)
    assert result.failure_reason != FailureReason.contract_invalid
    assert result.reason_codes == []


def test_topic_mismatch_keeps_clarification_fields_without_clarify_decision():
    from tests.support import ambiguity as amb, candidate as cand

    docs = [
        document(doc_id="doc-1", doc_number="A-15"),
        document(doc_id="doc-2", doc_number="B-20"),
    ]
    choices = [
        cand(doc_id="doc-1", label="North", doc_number="A-15"),
        cand(doc_id="doc-2", label="South", doc_number="B-20"),
    ]
    found = retrieval(
        documents=docs,
        evidence=[evidence(doc_id="doc-1"), evidence(doc_id="doc-2")],
        signals=signals(
            topic_match=False,
            max_relevance=0.9,
            margin_top1_top2=0.01,
            distinct_docs=2,
        ),
        ambiguity=[amb("counterparty", choices)],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert ReasonCode.topic_mismatch in result.reason_codes
    assert ReasonCode.ambiguous_entity in result.reason_codes
    assert result.missing_fields
    assert result.candidates
    assert result.suggested_decision != Decision.clarify


def test_topic_mismatch_rejects_even_when_relevance_is_high():
    found = retrieval(
        signals=signals(topic_match=False, max_relevance=0.99, score_kind="semantic"),
        evidence=[evidence(relevance=0.99, score=100)],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.topic_mismatch]
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_order_only_high_relevance_stays_rejected_for_the_scale():
    found = retrieval(
        signals=signals(
            topic_match=True,
            score_kind="order_only",
            max_relevance=0.99,
            margin_top1_top2=0.2,
        ),
        evidence=[evidence(score_kind="order_only", relevance=0.99, score=100)],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.sufficiency_scale_unavailable]
    assert ReasonCode.topic_mismatch not in result.reason_codes
    assert ReasonCode.insufficient_relevance not in result.reason_codes
    assert result.suggested_decision != Decision.allow_answer


@pytest.mark.parametrize("dense_used", [True, False])
@pytest.mark.parametrize("rerank_used", [True, False])
def test_dense_and_rerank_flags_do_not_change_allow(dense_used, rerank_used):
    found = retrieval(
        signals=signals(dense_used=dense_used, rerank_used=rerank_used, score_kind="semantic")
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.allow_answer)
    assert result.reason_codes == []


@pytest.mark.parametrize("dense_used", [True, False])
@pytest.mark.parametrize("rerank_used", [True, False])
def test_dense_and_rerank_flags_do_not_change_order_only(dense_used, rerank_used):
    found = retrieval(
        signals=signals(
            topic_match=True,
            score_kind="order_only",
            max_relevance=0.99,
            dense_used=dense_used,
            rerank_used=rerank_used,
            margin_top1_top2=0.2,
        ),
        evidence=[evidence(score_kind="order_only", relevance=0.99, score=100)],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.sufficiency_scale_unavailable]


def test_empty_retrieval_names_missing_facts_and_does_not_clarify():
    found = retrieval(
        documents=[],
        evidence=[],
        signals=signals(
            topic_match=True,
            score_kind="semantic",
            max_relevance=0.8,
            low_confidence=False,
            margin_top1_top2=0.2,
            distinct_docs=0,
            entity_match=[],
            event_date=None,
            applicable_editions=[],
        ),
        ambiguity=[],
        reference_errors=[],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.clarify
    assert result.reason_codes == [ReasonCode.missing_facts]
    assert any(gap.code == ReasonCode.missing_facts for gap in result.gaps)
    assert ReasonCode.ambiguous_entity not in result.reason_codes
    assert ReasonCode.missing_identifier not in result.reason_codes
    assert result.event_date is None


@pytest.mark.parametrize(
    ("kind", "code"),
    [
        ("edition_missing", ReasonCode.edition_missing),
        ("unresolved_doc", ReasonCode.reference_error),
        ("unresolved_clause", ReasonCode.reference_error),
        ("ambiguous", ReasonCode.reference_error),
        ("broken_internal_ref", ReasonCode.reference_error),
        ("orphan_amendment", ReasonCode.reference_error),
    ],
)
def test_reference_error_blocks_allow_without_a_dependence_claim(kind, code):
    found = retrieval(reference_errors=[reference_error(kind=kind)])
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert result.reason_codes == [code]


def test_edition_missing_and_other_reference_errors_both_remain():
    found = retrieval(
        reference_errors=[
            reference_error(kind="edition_missing", raw="old statute"),
            reference_error(kind="broken_internal_ref", raw="clause 4"),
        ]
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert set(result.reason_codes) == {
        ReasonCode.edition_missing,
        ReasonCode.reference_error,
    }
    assert result.suggested_decision != Decision.allow_answer


def test_repeated_reference_kinds_do_not_duplicate_the_reason():
    found = retrieval(
        reference_errors=[
            reference_error(kind="unresolved_doc", raw="doc A"),
            reference_error(kind="orphan_amendment", raw="amendment 2"),
        ]
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.reference_error]


def test_several_applicable_editions_are_ambiguous():
    found = _two_editions("2024-01-01", "2024-12-31")
    result = _evaluate(_request(found, question=EDITION_QUESTION))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert result.reason_codes == [ReasonCode.ambiguous_edition]
    assert ReasonCode.date_conflict not in result.reason_codes


def test_silence_about_the_applicable_edition_is_ambiguous():
    found = retrieval(
        documents=[document(applicable_edition=False)],
        signals=signals(
            topic_match=True,
            score_kind="semantic",
            max_relevance=0.8,
            low_confidence=False,
            margin_top1_top2=0.2,
            event_date="2024-06-01",
            applicable_editions=[],
            distinct_docs=1,
        ),
        ambiguity=[],
        reference_errors=[],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert result.reason_codes == [ReasonCode.ambiguous_edition]
    assert ReasonCode.date_conflict not in result.reason_codes


def test_ambiguous_edition_and_reference_error_both_remain():
    found = _two_editions("2024-01-01", "2024-12-31")
    found = retrieval(
        documents=found.documents,
        evidence=found.evidence,
        signals=found.signals,
        ambiguity=[],
        reference_errors=[reference_error(kind="edition_missing", raw="old statute")],
    )
    result = _evaluate(_request(found, question=EDITION_QUESTION))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert set(result.reason_codes) == {
        ReasonCode.ambiguous_edition,
        ReasonCode.edition_missing,
    }
    assert ReasonCode.date_conflict not in result.reason_codes
    assert result.suggested_decision != Decision.allow_answer


def test_silence_and_reference_error_both_remain():
    found = retrieval(
        documents=[document(applicable_edition=False)],
        signals=signals(
            event_date="2024-06-01",
            applicable_editions=[],
            margin_top1_top2=0.2,
            topic_match=True,
            max_relevance=0.8,
        ),
        ambiguity=[],
        reference_errors=[reference_error(kind="orphan_amendment")],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert set(result.reason_codes) == {
        ReasonCode.ambiguous_edition,
        ReasonCode.reference_error,
    }


def test_ambiguous_edition_and_date_conflict_both_remain():
    found = _two_editions("2020-01-01", "2020-12-31")
    result = _evaluate(_request(found, question=EDITION_QUESTION))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert set(result.reason_codes) == {
        ReasonCode.ambiguous_edition,
        ReasonCode.date_conflict,
    }


def test_partial_clause_when_clarification_stays_closed():
    found = retrieval()
    assert len(found.documents) == 1
    assert found.signals.margin_top1_top2 == 0.2
    assert found.ambiguity == []
    assert found.documents[0].matched_clauses == ["A"]
    assert found.evidence[0].clause_ids == ["A"]
    result = _evaluate(
        _request(found, question_kind="fees", policy={"fees": ["A", "B"]})
    )
    _assert_ok(result, Decision.reject_partial_coverage)
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert result.reason_codes == [ReasonCode.partial_clause_coverage]
    assert list(result.coverage.missing) == ["B"]


def test_order_only_keeps_rejection_when_a_clause_is_also_missing():
    found = retrieval(
        signals=signals(
            topic_match=True,
            score_kind="order_only",
            max_relevance=0.99,
            margin_top1_top2=0.2,
        ),
        evidence=[evidence(score_kind="order_only", relevance=0.99, score=100)],
    )
    result = _evaluate(
        _request(found, question_kind="fees", policy={"fees": ["A", "B"]})
    )
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.reject_partial_coverage
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.topic_mismatch not in result.reason_codes
    assert set(result.reason_codes) == {
        ReasonCode.sufficiency_scale_unavailable,
        ReasonCode.partial_clause_coverage,
    }
    assert list(result.coverage.missing) == ["B"]


def test_earliest_rule_wins_and_later_codes_accumulate():
    result = _evaluate(
        _request(_contested(), question_kind="fees", policy={"fees": ["A", "B"]})
    )
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.reject_partial_coverage
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert set(result.reason_codes) == {
        ReasonCode.topic_mismatch,
        ReasonCode.partial_clause_coverage,
        ReasonCode.named_not_found,
        ReasonCode.low_confidence,
        ReasonCode.insufficient_relevance,
    }
    assert list(result.coverage.missing) == ["B"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_numbers", ["K-9"]),
        ("party_names", ["South Dock LLC"]),
        ("law_articles", ["art-12"]),
        ("clauses", ["Z-4"]),
    ],
)
def test_unresolved_entry_is_named_not_found(field, value):
    found = retrieval(unresolved={field: value})
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert result.reason_codes == [ReasonCode.named_not_found]


def test_empty_entity_match_and_empty_unresolved_do_not_refuse():
    found = retrieval(signals=signals(entity_match=[]), unresolved={})
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.named_not_found not in result.reason_codes
    assert result.reason_codes == []


def test_name_only_in_the_question_does_not_refuse():
    found = retrieval(
        signals=signals(entity_match=["entity-a"]),
        unresolved={},
    )
    result = _evaluate(
        _request(found, question="What is the fee for South Dock LLC?")
    )
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.named_not_found not in result.reason_codes


def test_low_confidence_forbids_allow():
    found = retrieval(signals=signals(low_confidence=True, max_relevance=0.8))
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert result.reason_codes == [ReasonCode.low_confidence]


def test_semantic_relevance_below_default_threshold():
    found = retrieval(
        signals=signals(score_kind="semantic", max_relevance=0.34),
        evidence=[evidence(score_kind="semantic", relevance=0.9, score=100)],
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.insufficient_relevance]


def test_semantic_relevance_at_threshold_does_not_fire():
    found = retrieval(
        signals=signals(score_kind="semantic", max_relevance=0.35),
        evidence=[evidence(score_kind="semantic", relevance=0.1, score=100)],
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.insufficient_relevance not in result.reason_codes
    assert result.reason_codes == []


def test_lexical_relevance_below_default_threshold():
    found = retrieval(
        signals=signals(score_kind="lexical", max_relevance=0.19),
        evidence=[evidence(score_kind="lexical", relevance=0.9, score=100)],
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.insufficient_relevance]


def test_lexical_relevance_at_threshold_does_not_fire():
    found = retrieval(
        signals=signals(score_kind="lexical", max_relevance=0.20),
        evidence=[evidence(score_kind="lexical", relevance=0.1, score=100)],
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.insufficient_relevance not in result.reason_codes
    assert result.reason_codes == []


def test_lexical_score_ignores_the_semantic_threshold():
    found = retrieval(
        signals=signals(score_kind="lexical", max_relevance=0.34),
        evidence=[evidence(score_kind="lexical", relevance=0.34, score=100)],
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_mean_relevance_alone_does_not_fire():
    found = retrieval(
        signals=signals(
            score_kind="semantic",
            max_relevance=0.8,
            mean_relevance=0.0,
            mean_top3_relevance=0.0,
        )
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.allow_answer)
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_high_mean_does_not_hide_low_max_relevance():
    found = retrieval(
        signals=signals(
            score_kind="semantic",
            max_relevance=0.34,
            mean_relevance=0.99,
            mean_top3_relevance=0.99,
        ),
        evidence=[evidence(relevance=0.99, score=100)],
    )
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.reason_codes == [ReasonCode.insufficient_relevance]


def test_event_date_outside_the_applicable_term_is_date_conflict():
    found = retrieval(
        documents=[
            document(
                applicable_edition=True,
                effective_from="2020-01-01",
                effective_to="2020-12-31",
            )
        ],
        signals=signals(
            topic_match=True,
            score_kind="semantic",
            max_relevance=0.8,
            low_confidence=False,
            margin_top1_top2=0.2,
            distinct_docs=1,
            event_date="2024-06-01",
            applicable_editions=["edition-1"],
        ),
        reference_errors=[],
        ambiguity=[],
    )
    result = _evaluate(_request(found))
    _assert_ok(result, Decision.reject_insufficient_kb)
    assert result.suggested_decision != Decision.allow_answer
    assert result.reason_codes == [ReasonCode.date_conflict]
    assert ReasonCode.ambiguous_edition not in result.reason_codes
    assert result.event_date == "2024-06-01"


def test_quiet_retrieval_allows_the_answer():
    found = retrieval()
    assert found.evidence[0].score > 1
    result = _evaluate(_request(found), config=None)
    _assert_ok(result, Decision.allow_answer)
    assert result.reason_codes == []
    assert result.event_date == found.signals.event_date
    assert result.event_date == "2024-06-01"


@pytest.mark.parametrize(
    ("alias", "max_relevance", "decision", "code"),
    [
        ("bm25", 0.19, Decision.reject_insufficient_kb, ReasonCode.insufficient_relevance),
        ("bm25", 0.20, Decision.allow_answer, None),
        ("rerank", 0.34, Decision.reject_insufficient_kb, ReasonCode.insufficient_relevance),
        ("rerank", 0.35, Decision.allow_answer, None),
        ("rrf", 0.99, Decision.reject_insufficient_kb, ReasonCode.sufficiency_scale_unavailable),
        ("order", 0.99, Decision.reject_insufficient_kb, ReasonCode.sufficiency_scale_unavailable),
    ],
)
def test_score_kind_alias_selects_the_rule(alias, max_relevance, decision, code):
    payload = _dump(_request(retrieval()))
    payload["retrieval"]["signals"]["score_kind"] = alias
    payload["retrieval"]["signals"]["max_relevance"] = max_relevance
    payload["retrieval"]["evidence"][0]["score_kind"] = alias
    payload["retrieval"]["evidence"][0]["relevance"] = max_relevance
    payload["retrieval"]["evidence"][0]["score"] = 100
    result = _evaluate(payload)
    _assert_ok(result, decision)
    assert result.failure_reason != FailureReason.contract_invalid
    if code is None:
        assert result.reason_codes == []
    else:
        assert result.reason_codes == [code]


def test_config_invalid_is_logged_without_an_api_key(caplog, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", _SECRET)
    caplog.set_level(logging.DEBUG, logger="decision.coverage")
    result = _evaluate(
        _request(retrieval()),
        config=DecisionConfig(min_relevance_semantic="nope"),
    )
    _assert_failed(result, FailureReason.config_invalid)
    _logged(caplog, "config_invalid", _SECRET)
    assert _SECRET not in result.model_dump_json()


def test_contract_invalid_is_logged_without_an_api_key(caplog, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", _SECRET)
    caplog.set_level(logging.DEBUG, logger="decision.coverage")
    payload = _dump(_request(retrieval()))
    payload["retrieval"]["signals"]["score_kind"] = "mystery"
    result = _evaluate(payload)
    _assert_failed(result, FailureReason.contract_invalid)
    _logged(caplog, "contract_invalid", _SECRET)
    assert _SECRET not in result.model_dump_json()
