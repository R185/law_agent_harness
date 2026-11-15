"""Synthetic retrieval builders for the decision-layer tests.

Fixtures live next to the tests that use them. This module only fills the
required contract fields so each test can override the one fact it cares about.
"""

from __future__ import annotations

from contracts import (
    Ambiguity,
    Candidate,
    Citation,
    CitationCheck,
    DocumentMatch,
    Evidence,
    ReferenceError,
    RetrievalResult,
    Signals,
    Unresolved,
)


def signals(**overrides: object) -> Signals:
    base: dict[str, object] = {
        "low_confidence": False,
        "max_relevance": 0.8,
        "mean_relevance": 0.5,
        "mean_top3_relevance": 0.5,
        "margin_top1_top2": 0.2,
        "distinct_docs": 1,
        "entity_match": ["entity-a"],
        "topic_match": True,
        "score_kind": "semantic",
        "dense_used": True,
        "rerank_used": True,
        "event_date": "2024-06-01",
        "applicable_editions": ["edition-1"],
    }
    base.update(overrides)
    return Signals(**base)


def document(**overrides: object) -> DocumentMatch:
    base: dict[str, object] = {
        "doc_id": "doc-1",
        "title": "Internal rule",
        "doc_number": "K-1",
        "doc_type": "policy",
        "parties": ["North Warehouse LLC"],
        "status": "active",
        "effective_from": "2024-01-01",
        "effective_to": "2024-12-31",
        "applicable_edition": True,
        "edition_note": "",
        "matched_clauses": ["A"],
        "is_direct": True,
        "why": ["selected"],
    }
    base.update(overrides)
    return DocumentMatch(**base)


def evidence(**overrides: object) -> Evidence:
    base: dict[str, object] = {
        "evidence_id": "ev-1",
        "doc_id": "doc-1",
        "title": "Internal rule",
        "section": "1",
        "clause_ids": ["A"],
        "text": "The monthly fee is 100.",
        "start": 0,
        "end": 24,
        "score": 12.5,
        "score_kind": "semantic",
        "relevance": 0.8,
        "source": "bm25",
        "why": ["matched"],
    }
    base.update(overrides)
    return Evidence(**base)


def retrieval(**overrides: object) -> RetrievalResult:
    """One in-range document. Clarification stays closed: one document, margin 0.2."""

    base: dict[str, object] = {
        "documents": [document()],
        "evidence": [evidence()],
        "signals": signals(),
        "unresolved": Unresolved(),
        "ambiguity": [],
        "reference_errors": [],
        "trace": ["trace-1"],
    }
    base.update(overrides)
    return RetrievalResult(**base)


def candidate(**overrides: object) -> Candidate:
    base: dict[str, object] = {
        "doc_id": "doc-1",
        "title": "Internal rule",
        "doc_number": "K-1",
        "label": "North Warehouse LLC",
    }
    base.update(overrides)
    return Candidate(**base)


def ambiguity(raw: str = "counterparty", candidates: list[Candidate] | None = None) -> Ambiguity:
    return Ambiguity(raw=raw, candidates=candidates or [candidate()])


def reference_error(**overrides: object) -> ReferenceError:
    base: dict[str, object] = {
        "kind": "broken_internal_ref",
        "where": "kb_document",
        "raw": "clause 9",
        "doc_id": "doc-1",
        "clause": "9",
        "candidates": [],
        "severity": "high",
    }
    base.update(overrides)
    return ReferenceError(**base)


def citation_check(**overrides: object) -> CitationCheck:
    base: dict[str, object] = {
        "valid": True,
        "citations": [
            Citation(raw="Internal rule, clause A", doc_id="doc-1", clause="A", evidence_ids=["ev-1"])
        ],
        "unknown_docs": [],
        "unknown_clauses": [],
        "unsupported": [],
    }
    base.update(overrides)
    return CitationCheck(**base)
