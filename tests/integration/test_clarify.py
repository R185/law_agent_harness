"""T025: evaluate clarifies only the two document-choice cases."""

from __future__ import annotations

import pytest
from contracts import Unresolved
from decision import (
    CoverageRequest,
    Decision,
    DecisionConfig,
    DialogueTurn,
    LayerStatus,
    MissingField,
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

NEUTRAL = "Какой размер ответственности?"
NORTH = "Север Логистик"
SOUTH = "Южный Порт"
SAME_PERIOD = (
    ("2024-01-01", "2024-12-31"),
    ("2024-01-01", "2024-12-31"),
)
SPLIT_PERIOD = (
    ("2024-01-01", "2024-12-31"),
    ("2021-01-01", "2021-12-31"),
)
FORBIDDEN_TEXT_KEYS = ("question", "questions", "user_facing_text", "answer")


def evaluate(request: CoverageRequest):
    import decision

    return decision.evaluate(request, config=DecisionConfig(model_mode="off"))


def cover(question, found, dialogue=None, **extra) -> CoverageRequest:
    return CoverageRequest(
        question=question,
        dialogue=list(dialogue or []),
        retrieval=found,
        **extra,
    )


def docs_with(
    numbers: tuple[str, ...],
    periods: tuple[tuple[str | None, str | None], ...],
    parties: tuple[tuple[str, ...], ...] | None = None,
    titles: tuple[str, ...] | None = None,
    clauses: tuple[tuple[str, ...], ...] | None = None,
) -> list:
    if parties is None:
        parties = tuple(("Сторона",) for _ in numbers)
    if titles is None:
        titles = tuple(f"Документ {index + 1}" for index in range(len(numbers)))
    if clauses is None:
        clauses = tuple(("A",) for _ in numbers)
    built = []
    for index, number in enumerate(numbers):
        built.append(
            document(
                doc_id=f"doc-{index + 1}",
                title=titles[index],
                doc_number=number,
                parties=list(parties[index]),
                effective_from=periods[index][0],
                effective_to=periods[index][1],
                applicable_edition=index == 0,
                matched_clauses=list(clauses[index]),
            )
        )
    return built


def cands_with(
    labels: tuple[str, ...],
    numbers: tuple[str, ...] | None = None,
) -> list:
    if numbers is None:
        numbers = tuple("" for _ in labels)
    return [
        candidate(
            doc_id=f"doc-{index + 1}",
            title=f"Документ {index + 1}",
            doc_number=numbers[index],
            label=label,
        )
        for index, label in enumerate(labels)
    ]


def found_retrieval(
    documents: list,
    candidates: list,
    *,
    margin: float = 0.01,
    groups: int = 1,
    topic_match: bool = True,
    max_relevance: float = 0.8,
    unresolved: Unresolved | None = None,
    evidence_items: list | None = None,
):
    if evidence_items is None:
        if documents:
            evidence_items = [
                evidence(
                    doc_id=documents[0].doc_id,
                    clause_ids=["A"],
                    relevance=max_relevance,
                )
            ]
        else:
            evidence_items = []
    ambiguity_groups = []
    if candidates and groups > 0:
        ambiguity_groups = [
            ambiguity(f"choice-{index}", list(candidates))
            for index in range(groups)
        ]
    return retrieval(
        documents=documents,
        evidence=evidence_items,
        signals=signals(
            margin_top1_top2=margin,
            distinct_docs=len(documents),
            topic_match=topic_match,
            max_relevance=max_relevance,
            low_confidence=False,
            score_kind="semantic",
            event_date="2024-06-01",
            applicable_editions=["edition-1"],
        ),
        ambiguity=ambiguity_groups,
        unresolved=Unresolved() if unresolved is None else unresolved,
    )


def assert_envelope(result, found) -> None:
    assert result.layer_status == LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == "off"
    dumped = result.model_dump()
    for key in FORBIDDEN_TEXT_KEYS:
        assert key not in dumped
    allowed = {
        item.doc_id
        for group in found.ambiguity
        for item in group.candidates
    }
    for item in result.candidates:
        assert item.doc_id in allowed
    assert len(result.missing_fields) <= 3
    assert len(set(result.missing_fields)) == len(result.missing_fields)
    order = [MissingField.contract, MissingField.counterparty, MissingField.period]
    ranks = [order.index(field) for field in result.missing_fields]
    assert ranks == sorted(ranks)


def test_two_candidates_and_small_margin_clarify():
    documents = docs_with(("А-15", "Б-20"), SAME_PERIOD)
    choices = cands_with((NORTH, NORTH), ("А-15", "Б-20"))
    found = found_retrieval(documents, choices, margin=0.01)
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.contract]
    assert {(item.doc_id, item.label) for item in result.candidates} == {
        ("doc-1", NORTH),
        ("doc-2", NORTH),
    }
    assert set(result.reason_codes) == {
        ReasonCode.ambiguous_entity,
        ReasonCode.missing_identifier,
    }
    assert "question" not in result.model_dump()


def test_shared_title_fragment_with_empty_ambiguity_is_not_clarify():
    documents = [
        document(
            title="Положение Альфа",
            parties=["Альфа Групп"],
            applicable_edition=True,
        )
    ]
    found = found_retrieval(documents, [], margin=0.01, groups=0)
    result = evaluate(cover("Что требует положение Альфа?", found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert result.candidates == []
    assert all(item.label != "Альфа Групп" for item in result.candidates)


def test_unnamed_identifier_clarifies_with_that_field():
    documents = docs_with(("К-9", "К-9"), SPLIT_PERIOD)
    found = found_retrieval(documents, [], margin=0.2, groups=0)
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.period]
    assert set(result.reason_codes) == {ReasonCode.missing_identifier}


def test_empty_retrieval_rejects_without_clarify():
    choices = cands_with((NORTH, SOUTH), ("А-15", "Б-20"))
    found = found_retrieval(
        [],
        choices,
        margin=0.01,
        evidence_items=[],
        topic_match=True,
    )
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.clarify
    assert result.missing_fields == []
    assert result.candidates == []
    assert any(gap.code == ReasonCode.missing_facts for gap in result.gaps)
    assert ReasonCode.topic_mismatch not in result.reason_codes


@pytest.mark.parametrize(
    "unresolved",
    [
        Unresolved(contract_numbers=["Я-99"]),
        Unresolved(party_names=["Чужая сторона"]),
        Unresolved(law_articles=["статья 44"]),
        Unresolved(clauses=["пункт 8"]),
    ],
)
def test_named_not_found_is_not_a_missing_field(unresolved):
    found = retrieval(
        ambiguity=[],
        signals=signals(margin_top1_top2=0.2, distinct_docs=1, topic_match=True),
        unresolved=unresolved,
    )
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.named_not_found in result.reason_codes
    assert result.missing_fields == []
    assert result.candidates == []


def test_absent_topic_does_not_ask_for_a_regulation():
    documents = docs_with(("А-15", "Б-20"), SAME_PERIOD)
    choices = cands_with((NORTH, SOUTH), ("А-15", "Б-20"))
    found = found_retrieval(
        documents,
        choices,
        margin=0.01,
        topic_match=False,
        max_relevance=0.95,
    )
    result = evaluate(
        cover("Добавьте отсутствующий регламент по этой теме", found)
    )

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.reject_insufficient_kb
    assert result.suggested_decision != Decision.clarify
    assert ReasonCode.topic_mismatch in result.reason_codes
    assert ReasonCode.ambiguous_entity in result.reason_codes
    assert result.missing_fields == [
        MissingField.contract,
        MissingField.counterparty,
    ]
    assert result.candidates
    assert "question" not in result.model_dump()


def test_assistant_named_party_does_not_clear_ambiguity():
    documents = docs_with(("К-9",), (("2024-01-01", "2024-12-31"),))
    choices = cands_with((NORTH, SOUTH), ("К-9", "К-9"))
    found = found_retrieval(documents, choices, margin=0.01)
    dialogue = [
        DialogueTurn(role="assistant", content="Сторона — Север Логистик")
    ]
    result = evaluate(cover(NEUTRAL, found, dialogue))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.counterparty]
    assert {item.label for item in result.candidates} == {NORTH, SOUTH}
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_user_named_identifier_is_not_requested_again():
    documents = docs_with(
        ("А-15", "Б-20"),
        SAME_PERIOD,
        parties=((NORTH,), (SOUTH,)),
    )
    choices = cands_with((NORTH, SOUTH), ("А-15", "Б-20"))
    found = found_retrieval(documents, choices, margin=0.01)
    dialogue = [DialogueTurn(role="user", content="смотрим договор А-15")]
    result = evaluate(cover(NEUTRAL, found, dialogue))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.counterparty]
    assert MissingField.contract not in result.missing_fields
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_same_choice_is_not_requested_twice():
    documents = docs_with(("К-9",), (("2024-01-01", "2024-12-31"),))
    choices = cands_with((NORTH, SOUTH), ("К-9", "К-9"))
    found = found_retrieval(documents, choices, margin=0.01, groups=2)
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.counterparty]
    assert result.missing_fields.count(MissingField.counterparty) == 1


@pytest.mark.parametrize(
    "question",
    [
        "договор 2024, статья 12",
        "договор 2024, ст. 12",
    ],
)
def test_year_and_statute_article_do_not_count_as_contract_number(question):
    documents = docs_with(("А-15", "Б-20"), SPLIT_PERIOD)
    choices = cands_with((NORTH, NORTH), ("А-15", "Б-20"))
    found = found_retrieval(documents, choices, margin=0.01)
    result = evaluate(cover(question, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.contract]
    assert MissingField.period not in result.missing_fields
    assert {(item.doc_id, item.label) for item in result.candidates} == {
        ("doc-1", NORTH),
        ("doc-2", NORTH),
    }


def test_previous_user_topic_gap_is_not_copied():
    found = retrieval()
    dialogue = [DialogueTurn(role="user", content="Какая сегодня погода в Сочи?")]
    result = evaluate(
        cover("Какой порядок согласования поставки?", found, dialogue)
    )

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.allow_answer
    assert result.missing_fields == []
    assert ReasonCode.topic_mismatch not in result.reason_codes
    assert all(gap.code != ReasonCode.topic_mismatch for gap in result.gaps)


def test_unconfirmed_policy_clause_remains_while_decision_is_clarify():
    documents = docs_with(
        ("А-15", "Б-20"),
        SAME_PERIOD,
        clauses=(("A",), ("A",)),
    )
    choices = cands_with((NORTH, NORTH), ("А-15", "Б-20"))
    found = found_retrieval(documents, choices, margin=0.01)
    result = evaluate(
        cover(
            NEUTRAL,
            found,
            question_kind="supply",
            policy={"supply": ["A", "B"]},
        )
    )

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.suggested_decision != Decision.reject_partial_coverage
    assert result.coverage.missing == ["B"]
    assert "A" not in result.coverage.missing
    assert result.missing_fields == [MissingField.contract]


def test_two_documents_without_selecting_identifier_clarify():
    documents = docs_with(("А-15", "Б-20"), SAME_PERIOD)
    found = found_retrieval(documents, [], margin=0.2, groups=0)
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    assert result.missing_fields == [MissingField.contract]
    assert result.candidates == []
    assert set(result.reason_codes) == {ReasonCode.missing_identifier}


def test_one_remaining_document_does_not_open_case_two():
    documents = docs_with(("К-9",), (("2024-01-01", "2024-12-31"),))
    choices = cands_with((NORTH, NORTH), ("К-9", "К-9"))
    found = found_retrieval(documents, choices, margin=0.01)
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.allow_answer
    assert result.suggested_decision != Decision.clarify
    assert result.missing_fields == []


def test_candidate_doc_ids_belong_to_ambiguity():
    documents = docs_with(
        ("А-15", "Б-20"),
        SAME_PERIOD,
        parties=((NORTH,), (SOUTH,)),
    )
    choices = cands_with((NORTH, SOUTH), ("А-15", "Б-20"))
    found = found_retrieval(documents, choices, margin=0.01)
    result = evaluate(cover("сравните условия для Гамма Корп", found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    allowed = {item.doc_id for item in choices}
    assert result.candidates
    for item in result.candidates:
        assert item.doc_id in allowed
    assert all(item.label != "Гамма Корп" for item in result.candidates)


def test_serialized_recommendation_has_no_question_text():
    documents = docs_with(("А-15", "Б-20"), SAME_PERIOD)
    choices = cands_with((NORTH, NORTH), ("А-15", "Б-20"))
    found = found_retrieval(documents, choices, margin=0.01)
    result = evaluate(cover(NEUTRAL, found))

    assert_envelope(result, found)
    assert result.suggested_decision == Decision.clarify
    dumped = result.model_dump()
    for key in FORBIDDEN_TEXT_KEYS:
        assert key not in dumped
