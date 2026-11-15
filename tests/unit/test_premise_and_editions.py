"""Premise and edition checks through ``decision.evaluate``.

``decision.evaluate`` is called from the test body. These fixtures do not
import a search package, ``origin``, or the live suite.
"""

from __future__ import annotations

import datetime
import re
import sys

import pytest

from contracts import DocumentStatus, Unresolved
from decision import (
    CoverageRequest,
    Decision,
    DecisionConfig,
    DialogueTurn,
    LayerStatus,
    ModelMode,
    ReasonCode,
)
from tests.support import (
    ambiguity,
    candidate,
    document,
    evidence,
    retrieval,
    signals,
)

_FORBIDDEN = ("user_facing_text", "questions", "answer")


def _config() -> DecisionConfig:
    return DecisionConfig(model_mode=ModelMode.off)


def _evaluate(request):
    import decision

    return decision.evaluate(request, config=_config())


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
        "score": 4.0,
        "score_kind": "semantic",
        "relevance": 0.8,
        "source": "bm25",
        "why": ["matched"],
    }
    base.update(overrides)
    return evidence(**base)


def _request(
    question: str,
    *,
    dialogue=None,
    question_kind=None,
    policy=None,
    signal_overrides=None,
    documents=None,
    evidence_items=None,
    unresolved=None,
    ambiguity_items=None,
) -> CoverageRequest:
    overrides = {
        "topic_match": True,
        "score_kind": "semantic",
        "max_relevance": 0.8,
        "low_confidence": False,
        "entity_match": [],
        "margin_top1_top2": 0.2,
        "event_date": "2024-06-01",
        "applicable_editions": ["edition-1"],
        "dense_used": True,
        "rerank_used": True,
        "distinct_docs": 1,
    }
    overrides.update(signal_overrides or {})
    found = retrieval(
        documents=documents if documents is not None else [_plain_document()],
        evidence=evidence_items
        if evidence_items is not None
        else [_plain_evidence()],
        signals=signals(**overrides),
        unresolved=unresolved if unresolved is not None else Unresolved(),
        ambiguity=ambiguity_items if ambiguity_items is not None else [],
        reference_errors=[],
    )
    return CoverageRequest(
        question=question,
        dialogue=list(dialogue or []),
        retrieval=found,
        question_kind=question_kind,
        policy=policy,
    )


def _assert_readable(result, found) -> None:
    dumped = result.model_dump(mode="json")
    decision_value = dumped["suggested_decision"]
    assert isinstance(decision_value, str)
    assert decision_value in {item.value for item in Decision}
    assert not isinstance(result.suggested_decision, (list, tuple, set))
    assert result.suggested_decision == Decision(decision_value)
    codes = dumped["reason_codes"]
    assert codes == list(dict.fromkeys(codes))
    assert set(codes) <= {code.value for code in ReasonCode}
    assert result.event_date == found.signals.event_date
    assert list(result.applicable_editions) == list(found.signals.applicable_editions)
    for key in _FORBIDDEN:
        assert key not in dumped
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == ModelMode.off


def _pairs(result):
    return [(err.expected, err.actual) for err in result.premise_errors]


def _gaps(result, code):
    return [gap for gap in result.gaps if gap.code == code]


def _text_blob(found) -> str:
    parts = []
    for item in found.evidence:
        parts.extend(
            [item.text, item.title, item.section, item.evidence_id, " ".join(item.clause_ids)]
        )
    for doc in found.documents:
        parts.extend(
            [
                doc.doc_id,
                doc.title,
                doc.doc_number,
                doc.doc_type,
                doc.edition_note,
                doc.effective_from or "",
                doc.effective_to or "",
                doc.status.value,
                " ".join(doc.parties),
                " ".join(doc.matched_clauses),
                " ".join(doc.why),
            ]
        )
    return "\n".join(parts)


def _freeze_calendar(monkeypatch, year: int, month: int, day: int) -> list[str]:
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


def test_amount_pairs_share_the_allow_class():
    fee = _request(
        "The fee is 250.",
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
    )
    fine = _request(
        "The fine is 80.",
        evidence_items=[_plain_evidence(text="The fine is 15.")],
    )
    fee_result = _evaluate(fee)
    fine_result = _evaluate(fine)
    _assert_readable(fee_result, fee.retrieval)
    _assert_readable(fine_result, fine.retrieval)
    assert fee_result.suggested_decision == fine_result.suggested_decision
    assert fee_result.suggested_decision == Decision.allow_answer
    assert ReasonCode.premise_error in fee_result.reason_codes
    assert ReasonCode.premise_error in fine_result.reason_codes
    assert _pairs(fee_result) == [("100", "250")]
    assert _pairs(fine_result) == [("15", "80")]
    assert fee_result.premise_errors[0].expected != "250"
    assert fine_result.premise_errors[0].expected != "80"
    assert "250" in fee_result.premise_errors[0].statement
    assert "80" in fine_result.premise_errors[0].statement
    assert fee.retrieval.evidence[0].text == "The monthly fee is 100."
    assert fine.retrieval.evidence[0].text == "The fine is 15."


def test_numbers_match_after_spaces_are_removed():
    request = _request(
        "The fee is 1 250.",
        evidence_items=[_plain_evidence(text="The fee is 1250.")],
    )
    assert "1 250" in request.question
    assert "1250" in request.retrieval.evidence[0].text
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.premise_errors == []
    assert ReasonCode.premise_error not in result.reason_codes


def test_spaced_number_against_a_different_amount_is_a_premise_error():
    request = _request(
        "The fee is 1 250.",
        evidence_items=[_plain_evidence(text="The fee is 100.")],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.premise_error in result.reason_codes
    assert _pairs(result) == [("100", "1 250")]
    assert result.premise_errors[0].expected != "1250"
    assert result.premise_errors[0].expected != "1 250"


@pytest.mark.parametrize(
    ("question_date", "material_date", "is_error"),
    [
        ("01.06.2024", "2024-06-01", False),
        ("2024-06-01", "2024-06-01", False),
        ("02.06.2024", "2024-06-01", True),
        ("2024-06-02", "2024-06-01", True),
    ],
)
def test_dates_use_iso_and_dotted_calendar_forms(
    question_date, material_date, is_error
):
    request = _request(
        f"The schedule date is {question_date}.",
        evidence_items=[
            _plain_evidence(text=f"The schedule date is {material_date}.")
        ],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    if is_error:
        assert _pairs(result) == [(material_date, question_date)]
        assert ReasonCode.premise_error in result.reason_codes
        assert all(err.expected != question_date for err in result.premise_errors)
    else:
        assert result.premise_errors == []
        assert ReasonCode.premise_error not in result.reason_codes


@pytest.mark.parametrize(
    "question",
    [
        "Does North Warehouse LLC owe the monthly fee?",
        "Does «North Warehouse LLC» owe the monthly fee?",
    ],
)
def test_party_on_the_document_matches_plain_text_and_guillemets(question):
    request = _request(
        question,
        documents=[_plain_document(parties=["North Warehouse LLC"])],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.premise_errors == []
    assert ReasonCode.premise_error not in result.reason_codes


def test_party_inside_guillemets_uses_the_materials_name():
    request = _request(
        "The counterparty is «Other Party LLC».",
        documents=[_plain_document(parties=["North Warehouse LLC"])],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.named_not_found not in result.reason_codes
    assert _pairs(result) == [("North Warehouse LLC", "Other Party LLC")]
    assert result.premise_errors[0].expected != "Other Party LLC"
    assert request.retrieval.documents[0].parties == ["North Warehouse LLC"]


@pytest.mark.parametrize(
    "question",
    [
        "Does Harbor Gate LLC owe the monthly fee?",
        "Does «Harbor Gate LLC» owe the monthly fee?",
    ],
)
def test_candidate_label_is_a_materials_party(question):
    request = _request(
        question,
        documents=[_plain_document(parties=[])],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
        ambiguity_items=[
            ambiguity(
                raw="counterparty",
                candidates=[
                    candidate(
                        doc_id="doc",
                        title="Storage rule",
                        doc_number="KW",
                        label="Harbor Gate LLC",
                    )
                ],
            )
        ],
    )
    assert len(request.retrieval.ambiguity[0].candidates) == 1
    assert request.retrieval.signals.margin_top1_top2 == 0.2
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert result.premise_errors == []
    assert ReasonCode.premise_error not in result.reason_codes


def test_candidate_label_is_expected_when_the_lawyer_names_another_party():
    request = _request(
        "The counterparty is «Other Party LLC».",
        documents=[_plain_document(parties=[])],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
        ambiguity_items=[
            ambiguity(
                raw="counterparty",
                candidates=[
                    candidate(
                        doc_id="doc",
                        title="Storage rule",
                        doc_number="KW",
                        label="Harbor Gate LLC",
                    )
                ],
            )
        ],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert _pairs(result) == [("Harbor Gate LLC", "Other Party LLC")]
    assert all(err.expected != "" for err in result.premise_errors)
    assert result.premise_errors[0].expected != "Other Party LLC"


def test_follow_up_keeps_the_fee_from_the_earlier_user_turn():
    request = _request(
        "Is that fee right?",
        dialogue=[DialogueTurn(role="user", content="The fee is 250.")],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
    )
    assert "250" not in request.question
    assert "250" in request.dialogue[0].content
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.premise_error in result.reason_codes
    assert _pairs(result) == [("100", "250")]
    assert "250" in result.premise_errors[0].statement
    assert result.premise_errors[0].expected != "250"


def test_follow_up_keeps_the_date_from_the_earlier_user_turn():
    request = _request(
        "Is that date right?",
        dialogue=[
            DialogueTurn(role="user", content="The schedule date is 02.06.2024.")
        ],
        evidence_items=[_plain_evidence(text="The schedule date is 2024-06-01.")],
    )
    assert "02.06.2024" not in request.question
    assert "2024-06-01" not in request.question
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert _pairs(result) == [("2024-06-01", "02.06.2024")]


def test_follow_up_keeps_the_party_from_the_earlier_user_turn():
    request = _request(
        "Is that party right?",
        dialogue=[
            DialogueTurn(
                role="user",
                content="The counterparty is «Other Party LLC».",
            )
        ],
        documents=[_plain_document(parties=["North Warehouse LLC"])],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
    )
    assert "Other Party LLC" not in request.question
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert _pairs(result) == [("North Warehouse LLC", "Other Party LLC")]
    assert ReasonCode.named_not_found not in result.reason_codes


def test_assistant_turn_does_not_clear_the_lawyer_fee():
    request = _request(
        "Is that fee right?",
        dialogue=[
            DialogueTurn(role="user", content="The fee is 250."),
            DialogueTurn(role="assistant", content="The fee is 100."),
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
    )
    assert "250" not in request.question
    assert "250" not in request.dialogue[1].content
    assert request.dialogue[1].role == "assistant"
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    assert ("100", "250") in _pairs(result)
    assert all(err.expected != "250" for err in result.premise_errors)
    assert all(err.actual != "100" for err in result.premise_errors)


def test_assistant_turn_does_not_close_a_clause_gap():
    request = _request(
        "Are the required clauses covered?",
        dialogue=[
            DialogueTurn(role="assistant", content="Clause B is confirmed."),
        ],
        question_kind="bundle",
        policy={"bundle": ["A", "B"]},
        documents=[
            _plain_document(
                matched_clauses=["A"],
                effective_from="2024-01-01",
                effective_to="2024-12-31",
            )
        ],
        evidence_items=[
            _plain_evidence(text="Clause A applies.", clause_ids=["A"])
        ],
        signal_overrides={"entity_match": ["North Warehouse LLC"]},
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_partial_coverage
    assert result.suggested_decision != Decision.allow_answer
    assert result.coverage.missing == ["B"]
    assert "B" not in result.coverage.confirmed
    assert ReasonCode.partial_clause_coverage in result.reason_codes
    assert _gaps(result, ReasonCode.partial_clause_coverage)


def test_assistant_turn_does_not_confirm_an_entity():
    request = _request(
        "What is the monthly fee?",
        dialogue=[
            DialogueTurn(
                role="assistant",
                content="North Warehouse LLC is confirmed.",
            )
        ],
        unresolved=Unresolved(party_names=["North Warehouse LLC"]),
        signal_overrides={"entity_match": []},
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert request.retrieval.signals.entity_match == []
    assert request.retrieval.unresolved.party_names == ["North Warehouse LLC"]
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.named_not_found in result.reason_codes
    assert result.missing_fields == []
    assert result.candidates == []


def test_stated_fee_with_no_number_in_the_materials_is_missing_facts():
    request = _request(
        "The fee is 250.",
        documents=[_plain_document()],
        evidence_items=[_plain_evidence(text="The storage rule applies.")],
    )
    blob = _text_blob(request.retrieval)
    assert not re.search(r"\d", blob)
    assert "250" not in blob
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.allow_answer
    missing = _gaps(result, ReasonCode.missing_facts)
    assert missing
    assert all(gap.detail.strip() for gap in missing)
    assert ReasonCode.missing_facts in result.reason_codes
    assert result.premise_errors
    assert all(err.expected == "" for err in result.premise_errors)
    assert any(err.actual == "250" for err in result.premise_errors)
    for err in result.premise_errors:
        assert not re.search(r"\d", err.expected)
    for gap in missing:
        for token in re.findall(r"\d+", gap.detail):
            assert token == "250"
    assert "250" not in result.coverage.confirmed


def test_event_date_outside_the_applicable_term_is_only_date_conflict():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        documents=[
            _plain_document(
                applicable_edition=True,
                effective_from="2024-01-01",
                effective_to="2024-03-31",
                parties=["North Warehouse LLC"],
            )
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["edition-1"],
            "entity_match": ["North Warehouse LLC"],
        },
    )
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.date_conflict in result.reason_codes
    assert ReasonCode.ambiguous_edition not in result.reason_codes


def test_contradictory_term_is_date_conflict_not_ambiguous_edition():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        documents=[
            _plain_document(
                applicable_edition=True,
                effective_from="2024-12-31",
                effective_to="2024-01-01",
            )
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["edition-1"],
            "entity_match": ["North Warehouse LLC"],
        },
    )
    doc = request.retrieval.documents[0]
    assert doc.effective_from > doc.effective_to
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert ReasonCode.date_conflict in result.reason_codes
    assert ReasonCode.ambiguous_edition not in result.reason_codes


def test_two_marked_editions_are_ambiguous_and_not_date_conflict():
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
        documents=[
            document(doc_id="doc-1", **shared),
            document(doc_id="doc-2", **shared),
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.", doc_id="doc-1")],
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["ed-1", "ed-2"],
            "margin_top1_top2": 0.9,
            "distinct_docs": 2,
            "entity_match": ["North Warehouse LLC"],
        },
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
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.ambiguous_edition in result.reason_codes
    assert ReasonCode.date_conflict not in result.reason_codes


def test_silence_about_the_applicable_edition_is_ambiguous_edition():
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        documents=[
            _plain_document(
                applicable_edition=False,
                effective_from="2024-01-01",
                effective_to="2024-12-31",
            )
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": [],
            "entity_match": ["North Warehouse LLC"],
        },
    )
    assert request.retrieval.documents
    assert request.retrieval.signals.event_date == "2024-06-01"
    assert request.retrieval.signals.applicable_editions == []
    assert request.retrieval.documents[0].applicable_edition is False
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.ambiguous_edition in result.reason_codes
    assert ReasonCode.date_conflict not in result.reason_codes


def test_run_date_is_not_read(monkeypatch):
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        documents=[
            _plain_document(
                applicable_edition=True,
                effective_from="2024-01-01",
                effective_to="2024-12-31",
            )
        ],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["edition-1"],
            "entity_match": ["North Warehouse LLC"],
        },
    )
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
    assert first.event_date == second.event_date == "2024-06-01"
    assert first.event_date not in {"2031-05-05", "1999-01-02"}
    assert first.reason_codes == second.reason_codes
    assert ReasonCode.date_conflict not in first.reason_codes


def test_repealed_status_phrase_is_mapped_before_edition_comparison():
    doc = _plain_document(
        status="утратил силу",
        applicable_edition=True,
        effective_from="2024-01-01",
        effective_to="2024-12-31",
    )
    assert doc.status == DocumentStatus.repealed
    request = _request(
        "What monthly fee applies to North Warehouse LLC?",
        documents=[doc],
        evidence_items=[_plain_evidence(text="The monthly fee is 100.")],
        signal_overrides={
            "event_date": "2024-06-01",
            "applicable_editions": ["edition-1"],
            "entity_match": ["North Warehouse LLC"],
        },
    )
    assert request.retrieval.documents[0].status == DocumentStatus.repealed
    result = _evaluate(request)
    _assert_readable(result, request.retrieval)
    assert request.retrieval.documents[0].status == DocumentStatus.repealed
    assert result.suggested_decision == Decision.allow_answer
    assert ReasonCode.date_conflict not in result.reason_codes
    assert ReasonCode.ambiguous_edition not in result.reason_codes
