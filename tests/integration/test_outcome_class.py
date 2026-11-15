"""Class-not-literal outcomes for ``decision.evaluate``.

Twins differ in party names, document numbers, and amounts and share a
decision class. Question wording does not choose the class. Retrievals are
built here from contract fields. ``evaluate`` is intentionally not
implemented yet: these tests call it and keep the failure.
"""

from __future__ import annotations

import csv
import io
import tokenize
from functools import lru_cache
from pathlib import Path

import pytest

from tests.support import document, evidence, retrieval, signals

_ROOT = Path(__file__).resolve().parents[2]
_DEMO_CSV = _ROOT / "demo_questions.csv"
_SOURCE_ROOTS = ("decision", "contracts")

_RETIRED_CODES = frozenset(
    {
        "no_chunks",
        "low_retrieval_score",
        "topic_not_in_kb",
        "ambiguous_counterparty",
        "missing_contract_reference",
        "partial_legal_basis",
        "sufficient_context",
    }
)

# Demo-set literals. Structural words such as договор, контракт, статья, and
# ст. are not in this list. These strings may appear only in this test file
# and in the repository-root CSV the fixtures read.
_DEMO_LITERALS = (
    "Северный Склад",
    "Альфа-Трейд",
    "Альфа-Трейдинг",
    "Меридиан",
    "П-17",
    "А-08",
    "ПТ-05",
    "500 000",
    "12 млн",
)

_SEMANTIC_THRESHOLD = 0.35
_LEXICAL_THRESHOLD = 0.20

# Twenty questions outside the legal base. The retrieval mode is a column on
# the fixture. It is not chosen by reading the question.
_OFF_BASE = (
    ("weather", "Какая погода в Москве завтра?", "topic_mismatch"),
    ("recipe", "Сколько минут варить спагетти?", "topic_mismatch"),
    ("sport", "Кто выиграл чемпионат мира по футболу в 2018 году?", "topic_mismatch"),
    ("movie", "Посоветуй художественный фильм на вечер.", "topic_mismatch"),
    ("translate", "Переведи слово «спасибо» на французский.", "topic_mismatch"),
    ("code", "Напиши функцию сортировки списка на Python.", "topic_mismatch"),
    ("capital", "Какой город является столицей Австралии?", "topic_mismatch"),
    ("joke", "Расскажи короткий анекдот про лифт.", "topic_mismatch"),
    ("calories", "Сколько калорий в одном банане?", "topic_mismatch"),
    ("route", "Как дойти пешком от вокзала до реки?", "topic_mismatch"),
    ("poem", "Напиши стихотворение про море и чаек.", "empty"),
    ("playlist", "Составь список песен для утренней пробежки.", "empty"),
    ("souvenir", "Какой сувенир привезти из поездки на море?", "empty"),
    ("geometry", "Объясни теорему Пифагора пятикласснику.", "empty"),
    ("pet", "Придумай кличку для рыжего кота.", "empty"),
    ("garden", "Какие цветы посадить в ящике на балконе?", "empty"),
    ("flight", "Сколько часов длится перелёт до Токио?", "empty"),
    ("greeting", "Напиши поздравление с днём рождения коллеге.", "empty"),
    ("dream", "Что обычно значит сон про полёт?", "empty"),
    ("rhyme", "Подбери рифму к слову «облако».", "empty"),
)


@lru_cache(maxsize=1)
def _demo_questions() -> tuple[str, ...]:
    """Question cells from the repository-root CSV, unchanged."""

    with _DEMO_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == [str(number) for number in range(1, 19)]
    questions = tuple(row["question"] for row in rows)
    assert len(questions) == 18
    assert all(isinstance(item, str) and item for item in questions)
    return questions


def _document_retrieval(
    *,
    party: str,
    doc_number: str,
    amount: str,
    topic_match: bool = True,
    score_kind: str = "semantic",
    relevance: float = 0.8,
    entity_match: list[str] | None = None,
    matched_clauses: list[str] | None = None,
    mean_relevance: float = 0.5,
    with_order_scale: bool = False,
    low_confidence: bool = False,
):
    """One document. Clarification stays closed: margin 0.2, no ambiguity.

    ``amount`` is stored in the fragment text. ``with_order_scale`` adds a
    second fragment whose scale is only an order. Signals keep the sufficiency
    scale and ``relevance``; the ordering fragment must not replace them.
    """

    confirmed = [party] if entity_match is None else list(entity_match)
    clauses = ["A"] if matched_clauses is None else list(matched_clauses)
    text = f"The fee is {amount}."
    doc_id = f"doc-{doc_number}"
    title = f"Schedule {doc_number}"
    doc = document(
        doc_id=doc_id,
        title=title,
        doc_number=doc_number,
        parties=[party],
        status="active",
        effective_from="2024-01-01",
        effective_to="2024-12-31",
        applicable_edition=True,
        matched_clauses=clauses,
    )
    primary = evidence(
        evidence_id=f"ev-{doc_number}",
        doc_id=doc_id,
        title=title,
        clause_ids=clauses,
        text=text,
        start=0,
        end=len(text),
        score=60.0,
        score_kind=score_kind,
        relevance=relevance,
        source="dense" if score_kind == "semantic" else "bm25",
    )
    chunks = [primary]
    if with_order_scale:
        order_text = "Rank only."
        chunks.append(
            evidence(
                evidence_id=f"ev-order-{doc_number}",
                doc_id=doc_id,
                title=title,
                section="0",
                clause_ids=[],
                text=order_text,
                start=0,
                end=len(order_text),
                score=0.016,
                score_kind="order_only",
                relevance=0.99,
                source="bm25",
                why=["order"],
            )
        )
    found = retrieval(
        documents=[doc],
        evidence=chunks,
        signals=signals(
            low_confidence=low_confidence,
            max_relevance=relevance,
            mean_relevance=mean_relevance,
            mean_top3_relevance=mean_relevance,
            margin_top1_top2=0.2,
            distinct_docs=1,
            entity_match=confirmed,
            topic_match=topic_match,
            score_kind=score_kind,
            event_date="2024-06-01",
            applicable_editions=["edition-1"],
        ),
        ambiguity=[],
        reference_errors=[],
    )
    return found


def _empty_retrieval(trace: str):
    """No documents and no evidence. Topic matches; the scale is semantic."""

    return retrieval(
        documents=[],
        evidence=[],
        signals=signals(
            low_confidence=False,
            max_relevance=0.8,
            mean_relevance=0.5,
            mean_top3_relevance=0.5,
            margin_top1_top2=0.2,
            distinct_docs=0,
            entity_match=[],
            topic_match=True,
            score_kind="semantic",
            event_date=None,
            applicable_editions=[],
        ),
        ambiguity=[],
        reference_errors=[],
        trace=[trace],
    )


def _assert_closed(found, *, empty: bool) -> None:
    assert found.ambiguity == []
    assert found.signals.margin_top1_top2 == 0.2
    assert found.reference_errors == []
    assert found.unresolved.contract_numbers == []
    assert found.unresolved.party_names == []
    assert found.unresolved.law_articles == []
    assert found.unresolved.clauses == []
    if empty:
        assert found.documents == []
        assert found.evidence == []
        return
    assert len(found.documents) == 1


def _assert_edition_open(found) -> None:
    doc = found.documents[0]
    assert found.signals.event_date is not None
    assert doc.effective_from <= found.signals.event_date <= doc.effective_to
    assert found.signals.applicable_editions == ["edition-1"]
    assert doc.applicable_edition is True
    assert found.signals.low_confidence is False


def _assert_parties_differ(left, right, left_amount: str, right_amount: str) -> None:
    assert left.documents[0].parties != right.documents[0].parties
    assert left.documents[0].doc_number != right.documents[0].doc_number
    assert left_amount != right_amount
    assert left_amount in left.evidence[0].text
    assert right_amount in right.evidence[0].text
    assert left_amount not in right.evidence[0].text
    assert right_amount not in left.evidence[0].text


def _assert_layer_ok(result) -> None:
    import decision

    assert result.layer_status == decision.LayerStatus.ok
    assert result.failure_reason is None
    assert result.model_mode == decision.ModelMode.off


def test_allow_answer_twins_share_decision() -> None:
    import decision

    left_amount = "4100"
    right_amount = "77"
    left = _document_retrieval(
        party="Harbor Mills",
        doc_number="HM-14",
        amount=left_amount,
        topic_match=True,
        score_kind="semantic",
        relevance=0.8,
    )
    right = _document_retrieval(
        party="Cedar Transit",
        doc_number="CT-88",
        amount=right_amount,
        topic_match=True,
        score_kind="semantic",
        relevance=0.8,
    )
    _assert_closed(left, empty=False)
    _assert_closed(right, empty=False)
    _assert_edition_open(left)
    _assert_edition_open(right)
    _assert_parties_differ(left, right, left_amount, right_amount)
    assert left.signals.topic_match is True
    assert right.signals.topic_match is True
    assert left.signals.score_kind == "semantic"
    assert right.signals.score_kind == "semantic"
    assert left.signals.max_relevance == 0.8
    assert right.evidence[0].relevance == 0.8
    assert left.signals.low_confidence is False
    assert left.signals.entity_match == ["Harbor Mills"]
    assert right.signals.entity_match == ["Cedar Transit"]

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    left_result = decision.evaluate(
        decision.CoverageRequest(
            question="Каков размер платежа?",
            retrieval=left,
            policy={},
        ),
        config,
    )
    right_result = decision.evaluate(
        decision.CoverageRequest(
            question="Как считается ежемесячный платёж?",
            retrieval=right,
            policy={},
        ),
        config,
    )
    assert left_result.suggested_decision == right_result.suggested_decision
    assert left_result.suggested_decision == decision.Decision.allow_answer
    assert decision.ReasonCode.topic_mismatch not in left_result.reason_codes
    assert decision.ReasonCode.insufficient_relevance not in left_result.reason_codes
    assert decision.ReasonCode.sufficiency_scale_unavailable not in left_result.reason_codes
    _assert_layer_ok(left_result)
    _assert_layer_ok(right_result)


def test_topic_mismatch_twins_share_decision() -> None:
    import decision

    left_amount = "1280"
    right_amount = "903"
    left = _document_retrieval(
        party="Oak Ferry",
        doc_number="OF-3",
        amount=left_amount,
        topic_match=False,
        score_kind="semantic",
        relevance=0.99,
    )
    right = _document_retrieval(
        party="Linden Post",
        doc_number="LP-60",
        amount=right_amount,
        topic_match=False,
        score_kind="semantic",
        relevance=0.8,
    )
    _assert_closed(left, empty=False)
    _assert_closed(right, empty=False)
    _assert_edition_open(left)
    _assert_edition_open(right)
    _assert_parties_differ(left, right, left_amount, right_amount)
    assert left.signals.topic_match is False
    assert right.signals.topic_match is False
    assert left.signals.max_relevance == 0.99
    assert right.signals.max_relevance == 0.8

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    left_result = decision.evaluate(
        decision.CoverageRequest(
            question="Каков размер платежа?",
            retrieval=left,
            policy={},
        ),
        config,
    )
    right_result = decision.evaluate(
        decision.CoverageRequest(
            question="Как считается ежемесячный платёж?",
            retrieval=right,
            policy={},
        ),
        config,
    )
    assert left_result.suggested_decision == right_result.suggested_decision
    assert left_result.suggested_decision == decision.Decision.reject_insufficient_kb
    assert left_result.reason_codes == [decision.ReasonCode.topic_mismatch]
    assert right_result.reason_codes == [decision.ReasonCode.topic_mismatch]
    _assert_layer_ok(left_result)
    _assert_layer_ok(right_result)


def test_order_only_twins_share_decision() -> None:
    import decision

    left_amount = "560"
    right_amount = "24000"
    left = _document_retrieval(
        party="Maple Yard",
        doc_number="MY-21",
        amount=left_amount,
        topic_match=True,
        score_kind="order_only",
        relevance=0.99,
    )
    right = _document_retrieval(
        party="Birch Gate",
        doc_number="BG-5",
        amount=right_amount,
        topic_match=True,
        score_kind="order_only",
        relevance=0.91,
    )
    _assert_closed(left, empty=False)
    _assert_closed(right, empty=False)
    _assert_edition_open(left)
    _assert_edition_open(right)
    _assert_parties_differ(left, right, left_amount, right_amount)
    assert left.signals.score_kind == "order_only"
    assert right.signals.score_kind == "order_only"
    assert left.signals.max_relevance == 0.99
    assert right.signals.max_relevance == 0.91
    assert left.signals.topic_match is True
    assert right.signals.topic_match is True

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    left_result = decision.evaluate(
        decision.CoverageRequest(
            question="Каков размер платежа?",
            retrieval=left,
            policy={},
        ),
        config,
    )
    right_result = decision.evaluate(
        decision.CoverageRequest(
            question="Как считается ежемесячный платёж?",
            retrieval=right,
            policy={},
        ),
        config,
    )
    assert left_result.suggested_decision == right_result.suggested_decision
    assert left_result.suggested_decision == decision.Decision.reject_insufficient_kb
    assert left_result.reason_codes == [decision.ReasonCode.sufficiency_scale_unavailable]
    assert right_result.reason_codes == [decision.ReasonCode.sufficiency_scale_unavailable]
    _assert_layer_ok(left_result)
    _assert_layer_ok(right_result)


def test_partial_coverage_twins_share_decision() -> None:
    import decision

    left_amount = "300"
    right_amount = "815"
    left = _document_retrieval(
        party="Quarry Line",
        doc_number="QL-12",
        amount=left_amount,
        matched_clauses=["fee"],
    )
    right = _document_retrieval(
        party="Nimbus Hall",
        doc_number="NH-70",
        amount=right_amount,
        matched_clauses=["pack"],
    )
    _assert_closed(left, empty=False)
    _assert_closed(right, empty=False)
    _assert_edition_open(left)
    _assert_edition_open(right)
    _assert_parties_differ(left, right, left_amount, right_amount)
    assert left.documents[0].matched_clauses == ["fee"]
    assert right.documents[0].matched_clauses == ["pack"]
    assert "notice" not in left.documents[0].matched_clauses
    assert "claim" not in right.documents[0].matched_clauses

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    left_result = decision.evaluate(
        decision.CoverageRequest(
            question="Какие условия платежа действуют?",
            retrieval=left,
            question_kind="lease",
            policy={"lease": ["fee", "notice"]},
        ),
        config,
    )
    right_result = decision.evaluate(
        decision.CoverageRequest(
            question="Какие условия поставки действуют?",
            retrieval=right,
            question_kind="supply",
            policy={"supply": ["pack", "claim"]},
        ),
        config,
    )
    assert left_result.suggested_decision == right_result.suggested_decision
    assert left_result.suggested_decision == decision.Decision.reject_partial_coverage
    assert left_result.reason_codes == [decision.ReasonCode.partial_clause_coverage]
    assert right_result.reason_codes == [decision.ReasonCode.partial_clause_coverage]
    assert left_result.coverage.missing == ["notice"]
    assert right_result.coverage.missing == ["claim"]
    assert left_result.suggested_decision != decision.Decision.clarify
    _assert_layer_ok(left_result)
    _assert_layer_ok(right_result)


def test_empty_retrieval_twins_share_decision() -> None:
    import decision

    left_question = "Сколько платит ООО «Причал Восток» по документу Г-11, если сумма 8100?"
    right_question = "Сколько платит ООО «Кедровый Берег» по документу Ю-44, если сумма 25?"
    assert "Причал Восток" in left_question
    assert "Г-11" in left_question
    assert "8100" in left_question
    assert "Кедровый Берег" in right_question
    assert "Ю-44" in right_question
    assert "25" in right_question
    assert left_question != right_question

    left = _empty_retrieval("empty-east")
    right = _empty_retrieval("empty-west")
    _assert_closed(left, empty=True)
    _assert_closed(right, empty=True)
    assert left.signals.topic_match is True
    assert right.signals.topic_match is True
    assert left.signals.score_kind == "semantic"
    assert left.signals.max_relevance == 0.8
    assert right.signals.max_relevance == 0.8

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    left_result = decision.evaluate(
        decision.CoverageRequest(question=left_question, retrieval=left, policy={}),
        config,
    )
    right_result = decision.evaluate(
        decision.CoverageRequest(question=right_question, retrieval=right, policy={}),
        config,
    )
    assert left_result.suggested_decision == right_result.suggested_decision
    assert left_result.suggested_decision == decision.Decision.reject_insufficient_kb
    assert left_result.suggested_decision != decision.Decision.clarify
    assert right_result.suggested_decision != decision.Decision.clarify
    assert left_result.reason_codes == [decision.ReasonCode.missing_facts]
    assert right_result.reason_codes == [decision.ReasonCode.missing_facts]
    assert any(
        gap.code == decision.ReasonCode.missing_facts and gap.detail.strip()
        for gap in left_result.gaps
    )
    assert any(
        gap.code == decision.ReasonCode.missing_facts and gap.detail.strip()
        for gap in right_result.gaps
    )
    assert decision.ReasonCode.topic_mismatch not in left_result.reason_codes
    assert decision.ReasonCode.sufficiency_scale_unavailable not in left_result.reason_codes
    _assert_layer_ok(left_result)
    _assert_layer_ok(right_result)


@pytest.mark.parametrize("index", range(18), ids=[f"q{number:02d}" for number in range(1, 19)])
def test_demo_question_follows_retrieval_not_wording(index: int) -> None:
    """Even indexes allow. Odd indexes are topic mismatches. The split is by index."""

    import decision

    questions = _demo_questions()
    assert len(questions) == 18
    assert sum(item.count("\n") for item in questions) >= 1
    assert sum(item % 2 == 0 for item in range(18)) == 9
    assert sum(item % 2 == 1 for item in range(18)) == 9
    question = questions[index]
    expects_allow = index % 2 == 0
    if expects_allow:
        found = _document_retrieval(
            party="Blue Harbor LLC",
            doc_number="BH-2",
            amount="64",
            topic_match=True,
            score_kind="semantic",
            relevance=0.8,
        )
    else:
        found = _document_retrieval(
            party="Red Lantern LLC",
            doc_number="RL-9",
            amount="13",
            topic_match=False,
            score_kind="semantic",
            relevance=0.99,
        )
    _assert_closed(found, empty=False)
    _assert_edition_open(found)
    # Constants above are not read from the question text.
    if expects_allow:
        assert found.documents[0].parties == ["Blue Harbor LLC"]
        assert found.documents[0].doc_number == "BH-2"
        assert "64" in found.evidence[0].text
        assert found.signals.topic_match is True
    else:
        assert found.documents[0].parties == ["Red Lantern LLC"]
        assert found.documents[0].doc_number == "RL-9"
        assert "13" in found.evidence[0].text
        assert found.signals.topic_match is False
        assert found.signals.max_relevance == 0.99

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    request = decision.CoverageRequest(question=question, retrieval=found, policy={})
    assert request.question == question
    result = decision.evaluate(request, config)
    _assert_layer_ok(result)
    if expects_allow:
        assert result.suggested_decision == decision.Decision.allow_answer
    else:
        assert result.suggested_decision == decision.Decision.reject_insufficient_kb
        assert decision.ReasonCode.topic_mismatch in result.reason_codes


def test_same_demo_question_changes_only_with_retrieval() -> None:
    import decision

    question = _demo_questions()[0]
    allow = _document_retrieval(
        party="Blue Harbor LLC",
        doc_number="BH-2",
        amount="64",
        topic_match=True,
        score_kind="semantic",
        relevance=0.8,
    )
    mismatch = _document_retrieval(
        party="Blue Harbor LLC",
        doc_number="BH-2",
        amount="64",
        topic_match=False,
        score_kind="semantic",
        relevance=0.8,
    )
    assert allow.signals.topic_match is True
    assert mismatch.signals.topic_match is False
    assert allow.signals.max_relevance == mismatch.signals.max_relevance == 0.8
    assert allow.documents[0].parties == mismatch.documents[0].parties
    assert allow.documents[0].doc_number == mismatch.documents[0].doc_number

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    allow_request = decision.CoverageRequest(question=question, retrieval=allow, policy={})
    mismatch_request = decision.CoverageRequest(
        question=question,
        retrieval=mismatch,
        policy={},
    )
    assert allow_request.question == mismatch_request.question == question
    allow_result = decision.evaluate(allow_request, config)
    mismatch_result = decision.evaluate(mismatch_request, config)
    assert allow_result.suggested_decision == decision.Decision.allow_answer
    assert mismatch_result.suggested_decision == decision.Decision.reject_insufficient_kb
    assert decision.ReasonCode.topic_mismatch in mismatch_result.reason_codes
    _assert_layer_ok(allow_result)
    _assert_layer_ok(mismatch_result)


@pytest.mark.parametrize(
    ("case_id", "question", "retrieval_mode"),
    _OFF_BASE,
    ids=[item[0] for item in _OFF_BASE],
)
def test_off_base_question_is_not_allow_answer(
    case_id: str,
    question: str,
    retrieval_mode: str,
) -> None:
    import decision

    assert len(_OFF_BASE) >= 20
    texts = [item[1] for item in _OFF_BASE]
    assert any("погод" in text.casefold() for text in texts)
    assert any("стихотворен" in text.casefold() for text in texts)
    assert case_id
    if retrieval_mode == "topic_mismatch":
        found = _document_retrieval(
            party="Sample Press",
            doc_number="SP-1",
            amount="4",
            topic_match=False,
            score_kind="semantic",
            relevance=0.99,
        )
        _assert_closed(found, empty=False)
        assert found.signals.topic_match is False
        assert found.signals.max_relevance == 0.99
        assert found.documents[0].parties == ["Sample Press"]
        assert found.documents[0].doc_number == "SP-1"
    elif retrieval_mode == "empty":
        found = _empty_retrieval(f"off-{case_id}")
        _assert_closed(found, empty=True)
        assert found.signals.topic_match is True
    else:
        raise AssertionError(retrieval_mode)

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    request = decision.CoverageRequest(question=question, retrieval=found, policy={})
    assert request.question == question
    result = decision.evaluate(request, config)
    _assert_layer_ok(result)
    assert result.suggested_decision != decision.Decision.allow_answer
    assert result.suggested_decision == decision.Decision.reject_insufficient_kb


@pytest.mark.parametrize(
    ("score_kind", "relevance", "party", "doc_number", "amount", "confirm_entity", "with_order_scale"),
    [
        pytest.param(
            "semantic",
            _SEMANTIC_THRESHOLD,
            "Harbor Mills",
            "HM-35",
            "4100",
            True,
            True,
            id="semantic-entity-order-scale",
        ),
        pytest.param(
            "lexical",
            _LEXICAL_THRESHOLD,
            "Cedar Transit",
            "CT-20",
            "77",
            False,
            True,
            id="lexical-no-entity-order-scale",
        ),
        pytest.param(
            "semantic",
            _SEMANTIC_THRESHOLD,
            "Oak Ferry",
            "OF-35",
            "1280",
            False,
            False,
            id="semantic-no-entity",
        ),
        pytest.param(
            "lexical",
            _LEXICAL_THRESHOLD,
            "Linden Post",
            "LP-20",
            "903",
            True,
            False,
            id="lexical-entity",
        ),
    ],
)
def test_relevance_at_threshold_is_allow_answer(
    score_kind: str,
    relevance: float,
    party: str,
    doc_number: str,
    amount: str,
    confirm_entity: bool,
    with_order_scale: bool,
) -> None:
    import decision

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    assert config.min_relevance_semantic == 0.35
    assert config.min_relevance_lexical == 0.20
    assert _SEMANTIC_THRESHOLD == 0.35
    assert _LEXICAL_THRESHOLD == 0.20
    entity_match = [party] if confirm_entity else []
    found = _document_retrieval(
        party=party,
        doc_number=doc_number,
        amount=amount,
        topic_match=True,
        score_kind=score_kind,
        relevance=relevance,
        entity_match=entity_match,
        mean_relevance=0.05,
        with_order_scale=with_order_scale,
    )
    _assert_closed(found, empty=False)
    _assert_edition_open(found)
    assert found.signals.topic_match is True
    assert found.signals.score_kind == score_kind
    assert found.signals.score_kind != "order_only"
    assert found.signals.max_relevance == relevance
    assert found.evidence[0].relevance == relevance
    assert found.evidence[0].score_kind == score_kind
    assert found.signals.mean_relevance == 0.05
    assert found.signals.mean_top3_relevance == 0.05
    assert found.signals.mean_relevance < relevance
    assert found.signals.entity_match == entity_match
    if confirm_entity:
        assert found.signals.entity_match == [party]
    else:
        assert found.signals.entity_match == []
    if with_order_scale:
        assert any(item.score_kind == "order_only" for item in found.evidence)
        assert found.signals.max_relevance != 0.99
    else:
        assert all(item.score_kind != "order_only" for item in found.evidence)

    result = decision.evaluate(
        decision.CoverageRequest(
            question="Каков размер платежа?",
            retrieval=found,
            policy={},
        ),
        config,
    )
    assert result.suggested_decision == decision.Decision.allow_answer
    assert decision.ReasonCode.sufficiency_scale_unavailable not in result.reason_codes
    assert decision.ReasonCode.insufficient_relevance not in result.reason_codes
    assert decision.ReasonCode.topic_mismatch not in result.reason_codes
    _assert_layer_ok(result)


def test_order_only_max_relevance_099_is_insufficient_kb() -> None:
    import decision

    found = _document_retrieval(
        party="Maple Yard",
        doc_number="MY-99",
        amount="560",
        topic_match=True,
        score_kind="order_only",
        relevance=0.99,
    )
    _assert_closed(found, empty=False)
    _assert_edition_open(found)
    assert found.signals.score_kind == "order_only"
    assert found.evidence[0].score_kind == "order_only"
    assert found.signals.max_relevance == 0.99
    assert found.evidence[0].relevance == 0.99
    assert found.signals.topic_match is True
    assert found.documents
    assert found.evidence

    config = decision.DecisionConfig(model_mode=decision.ModelMode.off)
    result = decision.evaluate(
        decision.CoverageRequest(
            question="Каков размер платежа?",
            retrieval=found,
            policy={},
        ),
        config,
    )
    assert result.suggested_decision == decision.Decision.reject_insufficient_kb
    assert result.reason_codes == [decision.ReasonCode.sufficiency_scale_unavailable]
    assert result.suggested_decision != decision.Decision.allow_answer
    _assert_layer_ok(result)


def test_sources_exclude_retired_codes_and_demo_literals() -> None:
    import decision

    member_names = {code.name for code in decision.ReasonCode}
    member_values = {code.value for code in decision.ReasonCode}
    assert _RETIRED_CODES.isdisjoint(member_names)
    assert _RETIRED_CODES.isdisjoint(member_values)

    problems: list[str] = []
    scanned = 0
    for folder in _SOURCE_ROOTS:
        root = _ROOT / folder
        assert root.is_dir()
        files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
        assert files
        for path in files:
            scanned += 1
            relative = path.relative_to(_ROOT).as_posix()
            with tokenize.open(path) as handle:
                source = handle.read()
            for literal in _DEMO_LITERALS:
                if literal in source:
                    problems.append(f"{relative} contains {literal!r}")
            try:
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                for token in tokens:
                    if token.type == tokenize.NAME and token.string in _RETIRED_CODES:
                        line = token.start[0]
                        problems.append(
                            f"{relative}:{line} identifier {token.string}"
                        )
            except tokenize.TokenError as exc:
                problems.append(f"{relative} does not tokenize: {exc}")
    assert scanned >= 2
    assert not problems, "retired codes or demo literals:\n" + "\n".join(problems)
