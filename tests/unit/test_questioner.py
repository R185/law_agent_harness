"""T024: collect returns clarification fields and candidates, not a question."""

from __future__ import annotations

import pytest
from contracts import Unresolved
from decision import DecisionConfig, DialogueTurn, MissingField, ReasonCode
from decision.questioner import collect
from tests.support import ambiguity, candidate, document, evidence, retrieval, signals

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
BOTH_CASES = {
    ReasonCode.ambiguous_entity,
    ReasonCode.missing_identifier,
}


def docs_with(
    numbers: tuple[str, ...],
    periods: tuple[tuple[str | None, str | None], ...],
    parties: tuple[tuple[str, ...], ...] | None = None,
    titles: tuple[str, ...] | None = None,
) -> list:
    if parties is None:
        parties = tuple(("Сторона",) for _ in numbers)
    if titles is None:
        titles = tuple(f"Документ {index + 1}" for index in range(len(numbers)))
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
                matched_clauses=["A"],
            )
        )
    return built


def cands_with(
    labels: tuple[str, ...],
    numbers: tuple[str, ...] | None = None,
    titles: tuple[str, ...] | None = None,
) -> list:
    if numbers is None:
        numbers = tuple("" for _ in labels)
    if titles is None:
        titles = tuple(f"Документ {index + 1}" for index in range(len(labels)))
    return [
        candidate(
            doc_id=f"doc-{index + 1}",
            title=titles[index],
            doc_number=numbers[index],
            label=label,
        )
        for index, label in enumerate(labels)
    ]


def one_doc(number: str = "А-15"):
    return docs_with((number,), (("2024-01-01", "2024-12-31"),))


def retrieval_for(
    documents: list,
    candidates: list,
    *,
    margin: float = 0.01,
    groups: int = 1,
    evidence_items: list | None = None,
    unresolved: Unresolved | None = None,
):
    if evidence_items is None:
        if documents:
            evidence_items = [evidence(doc_id=documents[0].doc_id)]
        else:
            evidence_items = []
    ambiguity_groups = []
    if candidates and groups > 0:
        ambiguity_groups = [
            ambiguity(f"choice-{index}", list(candidates)) for index in range(groups)
        ]
    return retrieval(
        documents=documents,
        evidence=evidence_items,
        signals=signals(
            margin_top1_top2=margin,
            distinct_docs=len(documents),
            topic_match=True,
            max_relevance=0.8,
            low_confidence=False,
            score_kind="semantic",
            event_date="2024-06-01",
            applicable_editions=["edition-1"],
        ),
        ambiguity=ambiguity_groups,
        unresolved=Unresolved() if unresolved is None else unresolved,
    )


def assert_shape(result, found) -> None:
    dumped = result.model_dump()
    assert set(dumped) == {
        "applies",
        "missing_fields",
        "candidates",
        "reason_codes",
    }
    assert "question" not in dumped
    assert len(result.missing_fields) <= 3
    assert len(set(result.missing_fields)) == len(result.missing_fields)
    assert len(set(result.reason_codes)) == len(result.reason_codes)
    order = [MissingField.contract, MissingField.counterparty, MissingField.period]
    ranks = [order.index(field) for field in result.missing_fields]
    assert ranks == sorted(ranks)
    allowed = {
        (item.doc_id, item.label)
        for group in found.ambiguity
        for item in group.candidates
    }
    got = {(item.doc_id, item.label) for item in result.candidates}
    assert got <= allowed
    if result.applies:
        assert set(result.reason_codes) <= BOTH_CASES
        assert result.reason_codes
    else:
        assert result.missing_fields == []
        assert result.candidates == []
        assert result.reason_codes == []


def ask(question, found, dialogue=None, **config_overrides):
    result = collect(
        question,
        list(dialogue or []),
        found,
        DecisionConfig(model_mode="off", **config_overrides),
    )
    assert_shape(result, found)
    return result


def test_three_fields_keep_order_and_both_reason_codes():
    found = retrieval_for(
        docs_with(
            ("А-15", "Б-20"),
            SPLIT_PERIOD,
            parties=((NORTH,), (SOUTH,)),
        ),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.contract,
        MissingField.counterparty,
        MissingField.period,
    ]
    assert set(result.reason_codes) == BOTH_CASES
    assert {(item.doc_id, item.label) for item in result.candidates} == {
        ("doc-1", NORTH),
        ("doc-2", SOUTH),
    }


def test_contract_field_when_labels_and_periods_do_not_differ():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD),
        cands_with((NORTH, NORTH), ("А-15", "Б-20")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert set(result.reason_codes) == BOTH_CASES


def test_counterparty_field_on_one_document_opens_case_one_only():
    found = retrieval_for(
        one_doc(),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_period_field_when_contract_numbers_and_labels_do_not_differ():
    found = retrieval_for(
        docs_with(("К-9", "К-9"), SPLIT_PERIOD),
        cands_with((NORTH, NORTH), ("К-9", "К-9")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.period]
    assert set(result.reason_codes) == BOTH_CASES


def test_margin_equal_to_default_does_not_open_case_one():
    found = retrieval_for(
        one_doc(),
        cands_with((NORTH, SOUTH)),
        margin=0.05,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is False


def test_margin_above_default_does_not_open_case_one():
    found = retrieval_for(
        one_doc(),
        cands_with((NORTH, SOUTH)),
        margin=0.2,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is False


def test_margin_of_one_tenth_stays_closed_at_the_default_boundary():
    found = retrieval_for(
        one_doc(),
        cands_with((NORTH, SOUTH)),
        margin=0.1,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is False


def test_custom_margin_opens_only_when_strictly_below():
    documents = one_doc()
    candidates = cands_with((NORTH, SOUTH))
    closed = ask(
        NEUTRAL,
        retrieval_for(documents, candidates, margin=0.2),
        max_clarify_margin=0.2,
    )
    opened = ask(
        NEUTRAL,
        retrieval_for(documents, candidates, margin=0.1),
        max_clarify_margin=0.2,
    )

    assert closed.applies is False
    assert opened.applies is True
    assert opened.missing_fields == [MissingField.counterparty]
    assert set(opened.reason_codes) == {ReasonCode.ambiguous_entity}


def test_one_ambiguity_candidate_is_case_two_without_ambiguous_entity():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD),
        cands_with((NORTH,), ("А-15",)),
        margin=0.01,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert set(result.reason_codes) == {ReasonCode.missing_identifier}


def test_case_one_stays_closed_when_no_field_predicate_matches():
    found = retrieval_for(
        one_doc(),
        cands_with((NORTH, NORTH)),
        margin=0.01,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is False


@pytest.mark.parametrize(
    "question",
    [
        "договор А-15/2019",
        "контракт № 7-19",
        "договор ZX-100",
        "контракт 15/8",
        "условия по договору А-15",
    ],
)
def test_structural_contract_number_clears_only_the_contract_field(question):
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask(question, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]


def test_named_contract_that_leaves_one_document_closes_case_two():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask("договор А-15", found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_selecting_contract_number_does_not_clarify_without_another_field():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD),
        [],
        margin=0.2,
        groups=0,
    )
    result = ask("договор А-15", found)

    assert result.applies is False


@pytest.mark.parametrize(
    "question",
    [
        "договор 2024",
        "договор 2024, статья 12",
        "статья 12",
        "ст. 12",
        "ст.12",
        "договор № 152",
        "договор 15",
        "см. № 7-19",
    ],
)
def test_year_article_and_bare_number_are_not_contract_numbers(question):
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD),
        cands_with((NORTH, NORTH), ("А-15", "Б-20")),
    )
    result = ask(question, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert set(result.reason_codes) == BOTH_CASES


def test_four_digit_year_names_a_period_and_can_leave_one_document():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD),
        cands_with((NORTH, NORTH), ("А-15", "Б-20")),
    )
    result = ask("лимит в 2024 году", found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert MissingField.period not in result.missing_fields
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_user_turn_names_a_contract_so_it_is_not_requested_again():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    dialogue = [
        DialogueTurn(role="user", content="добрый день"),
        DialogueTurn(role="user", content="имею в виду договор А-15"),
    ]
    result = ask(NEUTRAL, found, dialogue)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.counterparty,
        MissingField.period,
    ]
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_user_turn_names_a_label_so_counterparty_is_not_requested_again():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    dialogue = [DialogueTurn(role="user", content="сторона Южный Порт")]
    result = ask(NEUTRAL, found, dialogue)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract, MissingField.period]
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_user_turn_names_a_year_so_period_is_not_requested_again():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    dialogue = [DialogueTurn(role="user", content="нужен 2024 год")]
    result = ask(NEUTRAL, found, dialogue)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.contract,
        MissingField.counterparty,
    ]
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_assistant_turn_does_not_name_an_identifier():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    dialogue = [
        DialogueTurn(
            role="assistant",
            content="договор А-15, сторона Север Логистик, 2024 год",
        )
    ]
    result = ask(NEUTRAL, found, dialogue)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.contract,
        MissingField.counterparty,
        MissingField.period,
    ]
    assert set(result.reason_codes) == BOTH_CASES


def test_case_two_when_two_documents_have_no_selecting_identifier():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD),
        [],
        margin=0.2,
        groups=0,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert result.candidates == []
    assert set(result.reason_codes) == {ReasonCode.missing_identifier}


def test_one_document_does_not_open_case_two():
    found = retrieval_for(one_doc(), [], margin=0.0, groups=0)
    result = ask(NEUTRAL, found)

    assert result.applies is False


def test_empty_documents_and_evidence_never_apply():
    found = retrieval_for(
        [],
        cands_with((NORTH, SOUTH)),
        margin=0.01,
        evidence_items=[],
    )
    result = ask(NEUTRAL, found)

    assert result.applies is False
    assert result.missing_fields == []
    assert result.candidates == []


def test_empty_evidence_alone_does_not_block_case_two():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD),
        [],
        margin=0.2,
        groups=0,
        evidence_items=[],
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert set(result.reason_codes) == {ReasonCode.missing_identifier}


def test_shared_title_fragment_does_not_add_a_candidate():
    found = retrieval_for(
        docs_with(
            ("А-15", "Б-20"),
            SAME_PERIOD,
            parties=(("Альфа Групп",), ("Альфа Плюс",)),
            titles=("Регламент Альфа", "Политика Альфа"),
        ),
        [],
        groups=0,
    )
    result = ask("сравните Альфа", found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert result.candidates == []
    assert all(item.label != "Альфа Групп" for item in result.candidates)
    assert all(item.label != "Альфа Плюс" for item in result.candidates)


def test_unknown_party_in_the_question_is_not_a_new_candidate():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask("что известно про Гамма Корп", found)

    assert result.applies is True
    assert {(item.doc_id, item.label) for item in result.candidates} == {
        ("doc-1", NORTH),
        ("doc-2", SOUTH),
    }
    assert all(item.label != "Гамма Корп" for item in result.candidates)


def test_open_clarification_ignores_unresolved_references():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
        unresolved=Unresolved(
            contract_numbers=["Я-99"],
            party_names=["Чужая сторона"],
            law_articles=["статья 44"],
            clauses=["пункт 8"],
        ),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.contract,
        MissingField.counterparty,
        MissingField.period,
    ]
    assert {(item.doc_id, item.label) for item in result.candidates} == {
        ("doc-1", NORTH),
        ("doc-2", SOUTH),
    }
    assert all(item.label != "Чужая сторона" for item in result.candidates)


@pytest.mark.parametrize(
    "unresolved",
    [
        Unresolved(contract_numbers=["Я-99"]),
        Unresolved(party_names=["Чужая сторона"]),
        Unresolved(law_articles=["статья 44"]),
        Unresolved(clauses=["пункт 8"]),
    ],
)
def test_unresolved_reference_alone_is_not_a_field_or_candidate(unresolved):
    found = retrieval_for(one_doc(), [], margin=0.2, groups=0, unresolved=unresolved)
    result = ask(NEUTRAL, found)

    assert result.applies is False
    assert result.missing_fields == []
    assert result.candidates == []


def test_duplicate_ambiguity_groups_do_not_duplicate_the_field():
    found = retrieval_for(
        one_doc(),
        cands_with((NORTH, SOUTH)),
        groups=2,
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]
    assert result.missing_fields.count(MissingField.counterparty) == 1


def test_norm_request_adds_no_question_and_no_extra_field():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask("Допишите отсутствующую норму", found)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.contract,
        MissingField.counterparty,
        MissingField.period,
    ]
    assert "question" not in result.model_dump()


def test_partial_label_does_not_clear_counterparty():
    found = retrieval_for(one_doc(), cands_with((NORTH, SOUTH)))
    result = ask("условия для Север", found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]


def test_one_full_label_clears_counterparty_and_leaves_contract():
    found = retrieval_for(
        docs_with(("А-15", "Б-20"), SAME_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "Б-20")),
    )
    result = ask("сторона Южный Порт", found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert set(result.reason_codes) == {ReasonCode.ambiguous_entity}


def test_blank_doc_number_does_not_count_as_a_second_contract():
    found = retrieval_for(
        docs_with(("А-15", ""), SPLIT_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [
        MissingField.counterparty,
        MissingField.period,
    ]


def test_repeated_doc_number_is_not_a_missing_contract():
    found = retrieval_for(
        docs_with(("А-15", "А-15"), SAME_PERIOD, parties=((NORTH,), (SOUTH,))),
        cands_with((NORTH, SOUTH), ("А-15", "А-15")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]
    assert set(result.reason_codes) == BOTH_CASES


def test_case_two_with_evidence_only_and_two_ambiguity_candidates():
    """Zero documents but evidence present: case 2 uses ambiguity candidates."""

    found = retrieval_for(
        [],
        cands_with((NORTH, SOUTH), ("", "")),
        margin=0.2,
        groups=1,
        evidence_items=[evidence(doc_id="doc-x", text="Fee terms appear in the fragment.")],
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.counterparty]
    assert set(result.reason_codes) == {ReasonCode.missing_identifier}
    assert {(item.doc_id, item.label) for item in result.candidates} == {
        ("doc-1", NORTH),
        ("doc-2", SOUTH),
    }


def test_counterparty_uses_candidate_labels_not_document_parties():
    found = retrieval_for(
        docs_with(
            ("А-15", "Б-20"),
            SAME_PERIOD,
            parties=(("Альфа Групп",), ("Бета Холдинг",)),
        ),
        cands_with(("Одна сторона", "Одна сторона"), ("А-15", "Б-20")),
    )
    result = ask(NEUTRAL, found)

    assert result.applies is True
    assert result.missing_fields == [MissingField.contract]
    assert MissingField.counterparty not in result.missing_fields
