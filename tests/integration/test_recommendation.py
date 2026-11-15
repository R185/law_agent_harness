"""Recommendation scenarios for ``decision.evaluate``.

Fixtures are synthetic. ``decision.evaluate`` is imported inside the test
body so collection succeeds before the function exists.
"""

from __future__ import annotations

import datetime
import re
import sys

from decision import (
    CoverageRequest,
    Decision,
    DecisionConfig,
    FailureReason,
    LayerStatus,
    ModelMode,
    ReasonCode,
)
from contracts import Unresolved
from tests.support import (
    ambiguity,
    candidate,
    document,
    evidence,
    reference_error,
    retrieval,
    signals,
)

_SEMANTIC = 0.35
_LEXICAL = 0.20
_FORBIDDEN = ("user_facing_text", "questions", "answer")


def _config() -> DecisionConfig:
    return DecisionConfig(model_mode=ModelMode.off)


def _evaluate(request):
    import decision

    return decision.evaluate(request, config=_config())


def _request(
    question: str,
    *,
    dialogue=None,
    question_kind=None,
    policy=None,
    signal_overrides=None,
    document_overrides=None,
    evidence_overrides=None,
    documents=None,
    evidence_items=None,
    unresolved=None,
    ambiguity_items=None,
    reference_errors=None,
) -> CoverageRequest:
    found = retrieval(
        documents=documents
        if documents is not None
        else [document(**(document_overrides or {}))],
        evidence=evidence_items
        if evidence_items is not None
        else [evidence(**(evidence_overrides or {}))],
        signals=signals(**(signal_overrides or {})),
        unresolved=unresolved if unresolved is not None else Unresolved(),
        ambiguity=ambiguity_items if ambiguity_items is not None else [],
        reference_errors=reference_errors if reference_errors is not None else [],
    )
    return CoverageRequest(
        question=question,
        dialogue=list(dialogue or []),
        retrieval=found,
        question_kind=question_kind,
        policy=policy,
    )


def _plain_document(**overrides):
    base = {
        "doc_id": "doc",
        "title": "Storage rule",
        "doc_number": "KW",
        "doc_type": "rule",
        "parties": ["North Warehouse LLC"],
        "status": "active",
        "effective_from": None,
        "effective_to": None,
        "applicable_edition": True,
        "edition_note": "",
        "matched_clauses": ["A"],
        "is_direct": True,
        "why": ["selected"],
    }
    base.update(overrides)
    return document(**base)


def _plain_evidence(**overrides):
    base = {
        "evidence_id": "ev",
        "doc_id": "doc",
        "title": "Storage rule",
        "section": "main",
        "clause_ids": ["A"],
        "text": "The storage rule applies.",
        "start": 0,
        "end": 1,
        "score": 8.0,
        "score_kind": "semantic",
        "relevance": 0.8,
        "source": "bm25",
        "why": ["matched"],
    }
    base.update(overrides)
    return evidence(**base)


def _assert_readable(result, found) -> None:
    """Successful read: one decision, closed unique codes, no lawyer text."""

    dumped = result.model_dump(mode="json")
    decision_value = dumped["suggested_decision"]
    assert isinstance(decision_value, str)
    assert decision_value in {item.value for item in Decision}
    assert not isinstance(result.suggested_decision, (list, tuple, set))
    assert result.suggested_decision == Decision(decision_value)
    codes = dumped["reason_codes"]
    assert isinstance(codes, list)
    assert codes == list(dict.fromkeys(codes))
    assert set(codes) <= {code.value for code in ReasonCode}
    assert result.event_date == found.signals.event_date
    assert list(result.applicable_editions) == list(found.signals.applicable_editions)
    for key in _FORBIDDEN:
        assert key not in dumped
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off


def _gaps(result, code):
    return [gap for gap in result.gaps if gap.code == code]


def _pairs(result):
    return [(err.expected, err.actual) for err in result.premise_errors]


def _freeze_calendar(monkeypatch, year: int, month: int, day: int) -> list[str]:
    """Point ``date.today`` and ``datetime.now`` at a calendar day."""

    real_date = datetime.date
    real_datetime = datetime.datetime
    calls: list[str] = []

    class FrozenDate(real_date):
        @classmethod
        def today(cls):
            calls.append("date.today")
            return cls(year, month, day)

    class FrozenDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append("datetime.now")
            return cls(year, month, day)

        @classmethod
        def today(cls):
            calls.append("datetime.today")
            return cls(year, month, day)

    monkeypatch.setattr(datetime, "date", FrozenDate)
    monkeypatch.setattr(datetime, "datetime", FrozenDateTime)
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("decision"):
            continue
        if getattr(module, "date", None) is real_date:
            monkeypatch.setattr(module, "date", FrozenDate)
        if getattr(module, "datetime", None) is real_datetime:
            monkeypatch.setattr(module, "datetime", FrozenDateTime)
    return calls


def test_01_topic_mismatch_rejects_despite_high_relevance():
    request = _request(
        "What is the weather tomorrow?",
        signal_overrides={"topic_match": False, "max_relevance": 0.95},
        evidence_overrides={"relevance": 0.95, "score": 9.5},
    )
    assert request.retrieval.signals.topic_match is False
    assert request.retrieval.signals.max_relevance == 0.95
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.topic_mismatch in result.reason_codes
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_02_order_only_scale_rejects_even_when_relevance_is_high():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "topic_match": True,
            "score_kind": "order_only",
            "max_relevance": 0.95,
            "low_confidence": False,
        },
        evidence_overrides={
            "score_kind": "order_only",
            "relevance": 0.95,
            "score": 4.0,
        },
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.sufficiency_scale_unavailable in result.reason_codes
    assert ReasonCode.topic_mismatch not in result.reason_codes
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_03_semantic_threshold_allows_and_names_absent_fact():
    """Policy clauses are confirmed and relevance is exactly 0.35.

    The question states a fee and a subject the fragment does not contain.
    Materials have no number, so the fee is a missing_facts gap with empty
    expected and is not stored as the norm. The decision still allows.
    """

    question = (
        "Does the cold-storage surcharge of 250 apply to North Warehouse LLC?"
    )
    request = _request(
        question,
        question_kind="storage",
        policy={"storage": ["A"]},
        signal_overrides={
            "topic_match": True,
            "score_kind": "semantic",
            "max_relevance": _SEMANTIC,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
            "applicable_editions": ["edition-1"],
        },
        documents=[_plain_document(matched_clauses=["A"])],
        evidence_items=[
            _plain_evidence(
                text="The storage rule applies.",
                relevance=_SEMANTIC,
                score=8.0,
                clause_ids=["A"],
            )
        ],
    )
    assert request.retrieval.signals.max_relevance == _SEMANTIC
    assert "cold-storage" not in request.retrieval.evidence[0].text
    assert "250" not in request.retrieval.evidence[0].text
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.coverage.required == ["A"]
    assert result.coverage.missing == []
    assert "A" in result.coverage.confirmed
    assert "cold-storage" not in result.coverage.confirmed
    assert "250" not in result.coverage.confirmed
    missing = _gaps(result, ReasonCode.missing_facts)
    assert missing
    assert all(gap.detail.strip() for gap in missing)
    assert ReasonCode.missing_facts in result.reason_codes
    assert result.premise_errors
    assert all(err.expected == "" for err in result.premise_errors)
    assert any(err.actual == "250" for err in result.premise_errors)
    assert all("250" in err.statement for err in result.premise_errors if err.actual == "250")
    for gap in missing:
        for token in re.findall(r"\d+", gap.detail):
            assert token == "250"


def test_03_no_policy_allows_at_semantic_threshold():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        question_kind=None,
        policy=None,
        signal_overrides={
            "topic_match": True,
            "score_kind": "semantic",
            "max_relevance": _SEMANTIC,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"relevance": _SEMANTIC, "text": "The monthly fee is 100."},
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.suggested_decision != Decision.reject_partial_coverage
    assert result.coverage.required == []
    assert result.coverage.missing == []
    assert ReasonCode.partial_clause_coverage not in result.reason_codes


def test_03_lexical_threshold_allows():
    """0.20 meets the lexical threshold and is below the semantic one."""

    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        question_kind=None,
        policy=None,
        signal_overrides={
            "topic_match": True,
            "score_kind": "lexical",
            "max_relevance": _LEXICAL,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={
            "score_kind": "lexical",
            "relevance": _LEXICAL,
            "score": 6.0,
            "text": "The monthly fee is 100.",
        },
    )
    assert request.retrieval.signals.max_relevance == _LEXICAL
    assert _LEXICAL < _SEMANTIC
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_low_confidence_forbids_allow_when_relevance_is_high():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "topic_match": True,
            "low_confidence": True,
            "max_relevance": 0.95,
            "score_kind": "semantic",
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"relevance": 0.95, "text": "The monthly fee is 100."},
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.low_confidence in result.reason_codes
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_mean_relevance_does_not_switch_the_decision():
    question = "What monthly fee applies to North Warehouse LLC?"
    low = _request(
        question,
        signal_overrides={
            "mean_relevance": 0.0,
            "mean_top3_relevance": 0.0,
            "max_relevance": 0.8,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    high = _request(
        question,
        signal_overrides={
            "mean_relevance": 1.0,
            "mean_top3_relevance": 1.0,
            "max_relevance": 0.8,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"text": "The monthly fee is 100."},
    )

    def _without_means(payload: dict) -> dict:
        signals_payload = dict(payload["retrieval"]["signals"])
        signals_payload.pop("mean_relevance")
        signals_payload.pop("mean_top3_relevance")
        retrieval_payload = dict(payload["retrieval"])
        retrieval_payload["signals"] = signals_payload
        return {**payload, "retrieval": retrieval_payload}

    assert _without_means(low.model_dump(mode="json")) == _without_means(
        high.model_dump(mode="json")
    )
    low_result = _evaluate(low)
    high_result = _evaluate(high)
    _assert_readable(low_result, low.retrieval)
    _assert_readable(high_result, high.retrieval)
    assert low.retrieval.signals.mean_relevance == 0.0
    assert low.retrieval.signals.mean_top3_relevance == 0.0
    assert high.retrieval.signals.mean_relevance == 1.0
    assert high.retrieval.signals.mean_top3_relevance == 1.0
    assert low_result.suggested_decision == Decision.allow_answer
    assert high_result.suggested_decision == Decision.allow_answer
    assert low_result.suggested_decision == high_result.suggested_decision
    assert ReasonCode.insufficient_relevance not in low_result.reason_codes
    assert ReasonCode.insufficient_relevance not in high_result.reason_codes


def test_04_lawyer_fee_is_a_premise_error_and_stays_allow():
    evidence_text = "The monthly fee is 100."
    request = _request(
        "The fee is 250.",
        signal_overrides={"entity_match": [], "max_relevance": 0.8},
        documents=[_plain_document()],
        evidence_items=[_plain_evidence(text=evidence_text)],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.premise_error in result.reason_codes
    assert _pairs(result) == [("100", "250")]
    assert result.premise_errors[0].statement
    assert "250" in result.premise_errors[0].statement
    assert request.retrieval.evidence[0].text == evidence_text
    assert "250" not in request.retrieval.evidence[0].text


def test_04_lawyer_date_is_a_premise_error_and_stays_allow():
    evidence_text = "The schedule date is 2024-06-01."
    request = _request(
        "The schedule date is 01.02.2024.",
        signal_overrides={"entity_match": [], "max_relevance": 0.8},
        documents=[_plain_document()],
        evidence_items=[_plain_evidence(text=evidence_text)],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.premise_error in result.reason_codes
    assert _pairs(result) == [("2024-06-01", "01.02.2024")]
    assert all(err.expected != "01.02.2024" for err in result.premise_errors)
    assert all(err.expected != "2024-02-01" for err in result.premise_errors)
    assert request.retrieval.evidence[0].text == evidence_text


def test_04_lawyer_party_is_a_premise_error_and_stays_allow():
    request = _request(
        "The counterparty is «Other Party LLC».",
        signal_overrides={"entity_match": [], "max_relevance": 0.8},
        documents=[_plain_document(parties=["North Warehouse LLC"])],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.premise_error in result.reason_codes
    assert ReasonCode.named_not_found not in result.reason_codes
    assert _pairs(result) == [("North Warehouse LLC", "Other Party LLC")]
    assert request.retrieval.documents[0].parties == ["North Warehouse LLC"]
    assert all(err.expected != "Other Party LLC" for err in result.premise_errors)


def test_05_partial_policy_names_only_the_missing_clause():
    request = _request(
        "Are clauses A, B, and C confirmed for North Warehouse LLC?",
        question_kind="bundle",
        policy={"bundle": ["A", "B", "C"]},
        signal_overrides={
            "topic_match": True,
            "low_confidence": False,
            "max_relevance": 0.8,
            "margin_top1_top2": 0.2,
            "entity_match": ["North Warehouse LLC"],
            "applicable_editions": ["edition-1"],
        },
        documents=[
            _plain_document(
                matched_clauses=["A", "C"],
                effective_from="2024-01-01",
                effective_to="2024-12-31",
            )
        ],
        evidence_items=[
            _plain_evidence(
                text="Clauses A and C are confirmed.",
                clause_ids=["A", "C"],
            )
        ],
        ambiguity_items=[],
    )
    assert request.retrieval.signals.margin_top1_top2 == 0.2
    assert request.retrieval.ambiguity == []
    assert len(request.retrieval.documents) == 1
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_partial_coverage
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.partial_clause_coverage in result.reason_codes
    assert result.coverage.required == ["A", "B", "C"]
    assert result.coverage.missing == ["B"]
    assert set(result.coverage.confirmed) == {"A", "C"}
    partial = _gaps(result, ReasonCode.partial_clause_coverage)
    assert partial
    assert any("B" in gap.detail for gap in partial)


def test_06_event_date_and_addendum_ignore_the_run_date(monkeypatch):
    """The run dates sit outside the edition term. The event date does not."""

    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["addendum-4"],
            "topic_match": True,
            "low_confidence": False,
            "max_relevance": 0.8,
            "entity_match": ["North Warehouse LLC"],
            "margin_top1_top2": 0.2,
        },
        documents=[
            _plain_document(
                applicable_edition=True,
                effective_from="2024-01-01",
                effective_to="2024-12-31",
            )
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
    )
    assert len(request.retrieval.documents) == 1
    assert request.retrieval.documents[0].applicable_edition is True
    first_calls = _freeze_calendar(monkeypatch, 2031, 5, 5)
    first = _evaluate(request)
    second_calls = _freeze_calendar(monkeypatch, 1999, 1, 2)
    second = _evaluate(request)
    _assert_readable(first, request.retrieval)
    _assert_readable(second, request.retrieval)
    assert first_calls == []
    assert second_calls == []
    assert first.suggested_decision == Decision.allow_answer
    assert second.suggested_decision == Decision.allow_answer
    assert first.suggested_decision == second.suggested_decision
    assert first.event_date == second.event_date == "2024-06-01"
    assert first.event_date not in {"2031-05-05", "1999-01-02"}
    assert list(first.applicable_editions) == ["addendum-4"]
    assert list(second.applicable_editions) == ["addendum-4"]
    assert first.reason_codes == second.reason_codes
    assert ReasonCode.date_conflict not in first.reason_codes
    assert ReasonCode.ambiguous_edition not in first.reason_codes
    assert first.suggested_decision != Decision.clarify


def test_07_missing_statute_edition_is_a_named_refusal():
    edition = "statute-edition-1994"
    request = _request(
        f"What does {edition} require for this fee?",
        signal_overrides={"topic_match": True, "max_relevance": 0.8},
        reference_errors=[
            reference_error(
                kind="edition_missing",
                raw=edition,
                clause=None,
                doc_id="doc-1",
            )
        ],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.edition_missing in result.reason_codes
    named = [gap for gap in result.gaps if edition in gap.detail]
    assert named
    assert all(gap.code == ReasonCode.edition_missing for gap in named)
    assert edition not in result.coverage.confirmed
    assert "answer" not in result.model_dump(mode="json")


def test_08_empty_retrieval_names_a_gap_and_does_not_clarify():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "topic_match": True,
            "distinct_docs": 0,
            "max_relevance": 0.0,
            "applicable_editions": ["edition-1"],
        },
        documents=[],
        evidence_items=[],
        ambiguity_items=[],
    )
    assert request.retrieval.documents == []
    assert request.retrieval.evidence == []
    assert request.retrieval.signals.topic_match is True
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.clarify
    assert result.suggested_decision != Decision.allow_answer
    missing = _gaps(result, ReasonCode.missing_facts)
    assert missing
    assert all(gap.detail.strip() for gap in missing)
    assert ReasonCode.missing_facts in result.reason_codes
    assert result.candidates == []
    assert result.missing_fields == []


def test_09_unresolved_kinds_stay_separate_and_are_not_missing_fields():
    unresolved = Unresolved(
        contract_numbers=["CN-77"],
        party_names=["Absent Party LLC"],
        law_articles=["article-12"],
        clauses=["clause-9"],
    )
    request = _request(
        "What is the monthly fee?",
        signal_overrides={
            "topic_match": True,
            "max_relevance": 0.8,
            "margin_top1_top2": 0.2,
            "entity_match": [],
        },
        unresolved=unresolved,
        ambiguity_items=[],
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    found = request.retrieval.unresolved
    assert found.contract_numbers == ["CN-77"]
    assert found.party_names == ["Absent Party LLC"]
    assert found.law_articles == ["article-12"]
    assert found.clauses == ["clause-9"]
    assert found.contract_numbers != found.party_names
    assert found.law_articles != found.clauses
    assert found.contract_numbers != found.clauses
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.named_not_found in result.reason_codes
    assert result.missing_fields == []
    assert result.candidates == []
    dumped = result.model_dump(mode="json")
    assert "CN-77" not in dumped["missing_fields"]
    assert "Absent Party LLC" not in [item.get("label") for item in dumped["candidates"]]


def test_10_raw_score_above_one_uses_relevance():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "topic_match": True,
            "score_kind": "semantic",
            "max_relevance": 0.5,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={
            "score": 3.5,
            "relevance": 0.5,
            "score_kind": "semantic",
            "text": "The monthly fee is 100.",
        },
    )
    assert request.retrieval.evidence[0].score == 3.5
    assert request.retrieval.evidence[0].relevance == 0.5
    assert request.retrieval.signals.max_relevance == 0.5
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.layer_status != LayerStatus.failed
    assert ReasonCode.insufficient_relevance not in result.reason_codes


def test_11_broken_reference_is_a_gap_and_does_not_allow():
    raw = "broken-ref-clause-9"
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "topic_match": True,
            "max_relevance": 0.8,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"text": "The monthly fee is 100."},
        reference_errors=[reference_error(kind="broken_internal_ref", raw=raw)],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.reference_error in result.reason_codes
    named = [gap for gap in result.gaps if raw in gap.detail]
    assert named
    assert all(gap.code == ReasonCode.reference_error for gap in named)
    assert all(gap.code != ReasonCode.edition_missing for gap in named)


def test_11_missing_edition_the_conclusion_needs_does_not_allow():
    edition = "statute-edition-1994"
    request = _request(
        f"The conclusion depends on {edition}.",
        signal_overrides={"topic_match": True, "max_relevance": 0.9},
        reference_errors=[
            reference_error(kind="edition_missing", raw=edition, clause=None)
        ],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert ReasonCode.edition_missing in result.reason_codes
    named = [gap for gap in result.gaps if edition in gap.detail]
    assert named
    assert all(gap.code == ReasonCode.edition_missing for gap in named)


def test_12_other_subject_wording_does_not_invent_topic_mismatch():
    request = _request(
        "What weather poem applies if the maritime fee is 250?",
        signal_overrides={
            "topic_match": True,
            "max_relevance": 0.8,
            "low_confidence": False,
            "entity_match": [],
        },
        documents=[_plain_document()],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
    )
    assert request.retrieval.signals.topic_match is True
    assert "weather" not in request.retrieval.evidence[0].text
    assert "poem" not in request.retrieval.evidence[0].text
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.topic_mismatch not in result.reason_codes
    assert _gaps(result, ReasonCode.topic_mismatch) == []
    assert ("100", "250") in _pairs(result)
    assert ReasonCode.premise_error in result.reason_codes


def test_14_fragment_clauses_without_policy_are_not_partial():
    request = _request(
        "Which clauses apply to North Warehouse LLC?",
        question_kind=None,
        policy=None,
        signal_overrides={
            "topic_match": True,
            "max_relevance": 0.8,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        documents=[_plain_document(matched_clauses=["A"])],
        evidence_items=[
            _plain_evidence(
                text="Clause A applies. Clause B is printed in this fragment.",
                clause_ids=["A"],
            )
        ],
    )
    assert "Clause B" in request.retrieval.evidence[0].text
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.suggested_decision != Decision.reject_partial_coverage
    assert result.coverage.required == []
    assert result.coverage.missing == []
    assert "B" not in result.coverage.missing
    assert ReasonCode.partial_clause_coverage not in result.reason_codes


def test_16_question_with_no_entities_allows_when_entity_match_is_empty():
    request = _request(
        "What is the monthly fee?",
        signal_overrides={
            "topic_match": True,
            "entity_match": [],
            "max_relevance": 0.8,
            "low_confidence": False,
            "score_kind": "semantic",
        },
        unresolved=Unresolved(),
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    assert request.retrieval.signals.entity_match == []
    assert request.retrieval.unresolved.contract_numbers == []
    assert request.retrieval.unresolved.party_names == []
    assert request.retrieval.unresolved.law_articles == []
    assert request.retrieval.unresolved.clauses == []
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.named_not_found not in result.reason_codes
    assert request.retrieval.signals.entity_match == []


def test_17_event_date_outside_applicable_term_is_date_conflict():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["edition-1"],
            "topic_match": True,
            "max_relevance": 0.8,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        documents=[
            document(
                applicable_edition=True,
                effective_from="2024-01-01",
                effective_to="2024-03-31",
            )
        ],
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    assert len(request.retrieval.documents) == 1
    assert request.retrieval.documents[0].applicable_edition is True
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.date_conflict in result.reason_codes
    assert ReasonCode.ambiguous_edition not in result.reason_codes


def test_18_two_applicable_editions_are_ambiguous_not_a_date_conflict():
    shared = {
        "doc_number": "K-1",
        "title": "Internal rule",
        "parties": ["North Warehouse LLC"],
        "applicable_edition": True,
        "effective_from": "2024-01-01",
        "effective_to": "2024-12-31",
        "matched_clauses": ["A"],
        "status": "active",
    }
    request = _request(
        "What is the monthly fee?",
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["ed-1", "ed-2"],
            "margin_top1_top2": 0.9,
            "distinct_docs": 2,
            "topic_match": True,
            "max_relevance": 0.8,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        documents=[
            document(doc_id="doc-1", **shared),
            document(doc_id="doc-2", **shared),
        ],
        evidence_overrides={"text": "The monthly fee is 100."},
        ambiguity_items=[
            ambiguity(
                raw="counterparty",
                candidates=[
                    candidate(
                        doc_id="doc-1",
                        label="North Warehouse LLC",
                        doc_number="K-1",
                    ),
                    candidate(
                        doc_id="doc-2",
                        label="North Warehouse LLC",
                        doc_number="K-1",
                    ),
                ],
            )
        ],
    )
    assert request.retrieval.signals.margin_top1_top2 == 0.9
    assert request.retrieval.documents[0].doc_number == request.retrieval.documents[1].doc_number
    labels = [
        item.label
        for group in request.retrieval.ambiguity
        for item in group.candidates
    ]
    assert labels == ["North Warehouse LLC", "North Warehouse LLC"]
    assert all(doc.applicable_edition is True for doc in request.retrieval.documents)
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.ambiguous_edition in result.reason_codes
    assert ReasonCode.date_conflict not in result.reason_codes
    assert result.missing_fields == []


def test_19_unreadable_relevance_is_contract_invalid_without_raising():
    valid = _request(
        "What monthly fee applies to North Warehouse LLC?",
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    payload = valid.model_dump(mode="json")
    payload["retrieval"]["signals"]["max_relevance"] = 1.5
    assert payload["retrieval"]["signals"]["max_relevance"] == 1.5
    result = _evaluate(payload)
    dumped = result.model_dump(mode="json")
    for key in _FORBIDDEN:
        assert key not in dumped
    assert result.layer_status == LayerStatus.failed
    assert result.failure_reason == FailureReason.contract_invalid
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert list(result.reason_codes) == []
    assert isinstance(dumped["suggested_decision"], str)


def test_20_dense_used_false_still_allows():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "dense_used": False,
            "rerank_used": True,
            "topic_match": True,
            "score_kind": "semantic",
            "max_relevance": 0.8,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert request.retrieval.signals.dense_used is False
    assert result.suggested_decision == Decision.allow_answer


def test_20_rerank_used_false_still_allows():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        signal_overrides={
            "dense_used": True,
            "rerank_used": False,
            "topic_match": True,
            "score_kind": "semantic",
            "max_relevance": 0.8,
            "low_confidence": False,
            "entity_match": ["North Warehouse LLC"],
        },
        evidence_overrides={"text": "The monthly fee is 100."},
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert request.retrieval.signals.rerank_used is False
    assert result.suggested_decision == Decision.allow_answer
