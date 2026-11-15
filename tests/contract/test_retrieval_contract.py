"""Parsing rules for the retrieval contract.

These tests build models directly. They do not call ``decision.evaluate``.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from contracts import DocumentStatus, RetrievalResult, ScoreKind, Signals
from tests.support import document, evidence, retrieval, signals

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


def test_retrieval_fixture_parses():
    found = retrieval()
    assert isinstance(found, RetrievalResult)
    again = RetrievalResult.model_validate(found.model_dump(mode="json"))
    assert again.evidence[0].score == found.evidence[0].score
    assert again.evidence[0].score > 1
    assert again.evidence[0].relevance == 0.8
    assert again.evidence[0].score_kind == ScoreKind.semantic
    assert again.signals.score_kind == ScoreKind.semantic
    assert again.documents[0].status == DocumentStatus.active


@pytest.mark.parametrize("score", [1.5, 2, 12.5, 100, 1_000_000])
def test_finite_score_above_one_parses(score):
    parsed = evidence(score=score, relevance=0.4, score_kind="semantic")
    assert parsed.score == score
    assert parsed.score > 1
    assert parsed.relevance == 0.4
    assert parsed.score_kind == ScoreKind.semantic
    assert parsed.score != parsed.relevance


@pytest.mark.parametrize("score", [0, 1])
def test_finite_score_on_unit_boundary_parses(score):
    assert evidence(score=score).score == score


@pytest.mark.parametrize(
    "value",
    [-0.01, -1, 1.01, 2, math.nan, math.inf, -math.inf],
)
def test_relevance_outside_zero_one_fails(value):
    with pytest.raises(ValidationError):
        evidence(relevance=value)


@pytest.mark.parametrize("value", [0, 1])
def test_relevance_bounds_parse(value):
    assert evidence(relevance=value).relevance == value


@pytest.mark.parametrize(
    "field",
    ["max_relevance", "mean_relevance", "mean_top3_relevance"],
)
@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
def test_signal_relevance_outside_zero_one_fails(field, value):
    with pytest.raises(ValidationError):
        signals(**{field: value})


@pytest.mark.parametrize(
    "field",
    ["max_relevance", "mean_relevance", "mean_top3_relevance"],
)
@pytest.mark.parametrize("value", [0, 1])
def test_signal_relevance_bounds_parse(field, value):
    parsed = signals(**{field: value})
    assert getattr(parsed, field) == value


@pytest.mark.parametrize("score", ["high", None, "", "twelve", [], {}])
def test_non_numeric_score_fails(score):
    with pytest.raises(ValidationError):
        evidence(score=score)


@pytest.mark.parametrize(
    "score",
    [math.nan, math.inf, -math.inf, "nan", "NaN", "inf", "-inf", "Infinity"],
)
def test_non_finite_score_fails(score):
    with pytest.raises(ValidationError):
        evidence(score=score)


@pytest.mark.parametrize("score", [True, False])
def test_bool_score_fails(score):
    with pytest.raises(ValidationError):
        evidence(score=score)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("semantic", ScoreKind.semantic),
        ("rerank", ScoreKind.semantic),
        ("lexical", ScoreKind.lexical),
        ("bm25", ScoreKind.lexical),
        ("order_only", ScoreKind.order_only),
        ("rrf", ScoreKind.order_only),
        ("order", ScoreKind.order_only),
    ],
)
def test_score_kind_maps_known_aliases(raw, expected):
    assert evidence(score_kind=raw).score_kind == expected
    assert signals(score_kind=raw).score_kind == expected


@pytest.mark.parametrize("raw", ["mystery", "cosine", "", None])
def test_unknown_score_kind_fails(raw):
    with pytest.raises(ValidationError):
        evidence(score_kind=raw)
    with pytest.raises(ValidationError):
        signals(score_kind=raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("active", DocumentStatus.active),
        ("Active", DocumentStatus.active),
        ("ACTIVE", DocumentStatus.active),
        ("действующая", DocumentStatus.active),
        ("действует", DocumentStatus.active),
        ("Действует", DocumentStatus.active),
        ("repealed", DocumentStatus.repealed),
        ("Repealed", DocumentStatus.repealed),
        ("утратил силу", DocumentStatus.repealed),
        ("", DocumentStatus.unknown),
        ("pending", DocumentStatus.unknown),
    ],
)
def test_document_status_maps(raw, expected):
    assert document(status=raw).status == expected


def test_event_date_accepts_real_calendar_date_or_null():
    assert signals(event_date="2024-06-01").event_date == "2024-06-01"
    assert signals(event_date="2020-02-29").event_date == "2020-02-29"
    assert signals(event_date=None).event_date is None


@pytest.mark.parametrize(
    "value",
    [
        "",
        "2024/06/01",
        "01.06.2024",
        "2024-6-01",
        "20240601",
        "June 1, 2024",
        "2024-02-30",
        "2021-02-29",
        "2024-13-01",
        "2024-00-10",
        "2024-06-31",
        " 2024-06-01",
        "2024-06-01 ",
        "2024-06-01T00:00:00",
        "null",
        "yesterday",
    ],
)
def test_event_date_rejects_other_strings(value):
    with pytest.raises(ValidationError):
        signals(event_date=value)


@pytest.mark.parametrize("field", _SIGNAL_FIELDS)
def test_missing_required_signal_fails(field):
    payload = signals().model_dump()
    del payload[field]
    with pytest.raises(ValidationError):
        Signals.model_validate(payload)


def test_unknown_signal_field_is_ignored():
    payload = signals().model_dump()
    payload["bundle_note"] = "ignored"
    parsed = Signals.model_validate(payload)
    assert isinstance(parsed, Signals)
    assert "bundle_note" not in parsed.model_dump()
    assert "bundle_note" not in Signals.model_fields
    assert parsed.topic_match is True
    assert parsed.max_relevance == 0.8


def test_signals_are_attributes_not_a_dict():
    parsed = retrieval().signals
    assert isinstance(parsed, Signals)
    assert not isinstance(parsed, dict)
    assert parsed.low_confidence is False
    assert parsed.max_relevance == 0.8
    assert parsed.mean_relevance == 0.5
    assert parsed.mean_top3_relevance == 0.5
    assert parsed.margin_top1_top2 == 0.2
    assert parsed.distinct_docs == 1
    assert parsed.entity_match == ["entity-a"]
    assert parsed.topic_match is True
    assert parsed.score_kind == ScoreKind.semantic
    assert parsed.dense_used is True
    assert parsed.rerank_used is True
    assert parsed.event_date == "2024-06-01"
    assert parsed.applicable_editions == ["edition-1"]
    with pytest.raises(TypeError):
        parsed["topic_match"]


@pytest.mark.parametrize(
    "payload",
    [
        {"score": 4.2},
        [{"score": 4.2}],
        [{"score": 4.2}, {"score": 1.1}],
    ],
)
def test_chunk_with_only_score_is_not_a_retrieval_result(payload):
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate(payload)


def test_end_before_start_fails():
    with pytest.raises(ValidationError):
        evidence(start=8, end=3)


def test_end_equal_to_start_parses():
    parsed = evidence(start=4, end=4)
    assert parsed.end == parsed.start


@pytest.mark.parametrize("value", [-1, -10])
def test_distinct_docs_below_zero_fails(value):
    with pytest.raises(ValidationError):
        signals(distinct_docs=value)


def test_distinct_docs_zero_parses():
    assert signals(distinct_docs=0).distinct_docs == 0
