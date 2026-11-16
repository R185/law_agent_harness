"""Adapters from the local RAG dataclasses to the decision/contracts layer.

The decision package intentionally does not import ``rag``.  This module is
the single translation point: it normalises the few field types that differ
(score scale, dates, nullable document numbers, citation shape) and keeps the
decision layer unaware of the local retrieval implementation.

It does not make coverage decisions.  Use ``decision.evaluate`` for that.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from contracts import (
    Ambiguity as ContractAmbiguity,
    Candidate as ContractCandidate,
    Citation as ContractCitation,
    CitationCheck as ContractCitationCheck,
    DocumentMatch as ContractDocumentMatch,
    DocumentStatus,
    Evidence as ContractEvidence,
    EvidenceSource,
    ReferenceError as ContractReferenceError,
    ReferenceKind,
    ReferenceWhere,
    RetrievalResult as ContractRetrievalResult,
    ScoreKind,
    Signals as ContractSignals,
    Unresolved as ContractUnresolved,
)
from rag.schema import RetrievalResult as RagRetrievalResult

_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_PARTIAL_DATE = re.compile(r"^(?P<year>[0-9]{4})(?:-(?P<month>[0-9]{1,2}))?$")
_REFERENCE_KINDS = {
    "unresolved_doc",
    "unresolved_clause",
    "ambiguous",
    "edition_missing",
    "broken_internal_ref",
    "orphan_amendment",
}
_EVIDENCE_SOURCES = {"bm25", "dense", "graph", "explicit"}


def _finite(value: Any, default: float = 0.0) -> float:
    """Convert a numeric value to a finite float without leaking NaN/inf."""
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _unit(value: Any) -> float:
    return max(0.0, min(1.0, _finite(value)))


def _iso_date(value: Any) -> str | None:
    """Normalise a full or partial date to YYYY-MM-DD; invalid -> None.

    RAG may emit a year or a year-month from query analysis.  The contract
    only accepts a calendar date, so a partial date is anchored to the first
    day of its period.  An unknown date stays None instead of inventing one.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if _ISO_DATE.fullmatch(text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    match = _PARTIAL_DATE.fullmatch(text)
    if match:
        year = int(match.group("year"))
        month = int(match.group("month") or 1)
        try:
            return date(year, month, 1).isoformat()
        except ValueError:
            return None
    return None


def _status(value: Any) -> DocumentStatus:
    text = str(value or "").strip().casefold()
    if text in {"active", "действующая", "действует"}:
        return DocumentStatus.active
    if text in {"repealed", "утратил силу"}:
        return DocumentStatus.repealed
    return DocumentStatus.unknown


def _score_kind(signals: Any) -> ScoreKind:
    """Map the RAG fusion scale onto the decision layer's closed scale.

    RRF is an ordering-only scale when nothing semantic was used.  If dense
    retrieval or a reranker participated, the ordering is still not a
    calibrated relevance score, but the decision layer has a semantic
    threshold for that case; using ``lexical`` for a pure BM25 run and
    ``semantic`` for dense/rerank keeps the gate meaningful.
    """
    raw = str(getattr(signals, "score_kind", "") or "").lower()
    if raw in {"semantic", "rerank", "dense"}:
        return ScoreKind.semantic
    if getattr(signals, "dense_used", False) or getattr(signals, "rerank_used", False):
        return ScoreKind.semantic
    return ScoreKind.lexical


def _candidate(value: Any) -> ContractCandidate | None:
    if isinstance(value, dict):
        doc_id = str(value.get("doc_id") or "")
        if not doc_id:
            return None
        return ContractCandidate(
            doc_id=doc_id,
            title=str(value.get("title") or ""),
            doc_number=str(value.get("doc_number") or ""),
            label=str(value.get("label") or value.get("title") or doc_id),
        )
    doc_id = str(getattr(value, "doc_id", "") or "")
    if not doc_id:
        return None
    return ContractCandidate(
        doc_id=doc_id,
        title=str(getattr(value, "title", "") or ""),
        doc_number=str(getattr(value, "doc_number", "") or ""),
        label=str(getattr(value, "label", "") or getattr(value, "title", "") or doc_id),
    )


def _document(doc: Any) -> ContractDocumentMatch:
    return ContractDocumentMatch(
        doc_id=str(doc.doc_id),
        title=str(doc.title or ""),
        # RAG permits None for an unknown contract number; the contract uses "".
        doc_number=str(doc.doc_number or ""),
        doc_type=str(doc.doc_type or "unknown"),
        parties=[str(p) for p in (doc.parties or [])],
        status=_status(doc.status),
        effective_from=_iso_date(doc.effective_from),
        effective_to=_iso_date(doc.effective_to),
        applicable_edition=bool(doc.applicable_edition),
        edition_note=str(doc.edition_note or ""),
        matched_clauses=[str(c) for c in (doc.matched_clauses or [])],
        is_direct=bool(doc.is_direct),
        why=[str(w) for w in (doc.why or [])],
    )


def _evidence(item: Any, index: int) -> ContractEvidence:
    source = str(item.source or "bm25").lower()
    if source not in _EVIDENCE_SOURCES:
        source = "bm25"
    start = int(_finite(item.start))
    end = int(_finite(item.end))
    if end < start:
        end = start
    doc_id = str(item.doc_id)
    return ContractEvidence(
        evidence_id=f"{doc_id}:{start}:{end}:{index}",
        doc_id=doc_id,
        title=str(item.title or ""),
        section=str(item.section or ""),
        clause_ids=[str(c) for c in (item.clause_ids or [])],
        text=str(item.text or ""),
        start=start,
        end=end,
        score=_finite(item.score),
        score_kind=_score_kind(item),
        relevance=_unit(item.relevance),
        source=EvidenceSource(source),
        why=[str(w) for w in (item.why or [])],
    )


def _reference_error(item: Any) -> ContractReferenceError | None:
    kind = str(item.kind or "")
    if kind not in _REFERENCE_KINDS:
        return None
    where = str(item.where or "question")
    if where not in {"question", "kb_document"}:
        where = "question"
    candidates: list[str] = []
    for candidate in item.candidates or []:
        if isinstance(candidate, dict):
            text = candidate.get("label") or candidate.get("doc_number") or candidate.get("title")
        else:
            text = getattr(candidate, "label", None) or getattr(candidate, "doc_number", None)
        if text:
            candidates.append(str(text))
    return ContractReferenceError(
        kind=ReferenceKind(kind),
        where=ReferenceWhere(where),
        raw=str(item.raw or ""),
        doc_id=str(item.doc_id) if item.doc_id else None,
        clause=str(item.clause) if item.clause else None,
        candidates=candidates,
        severity=str(item.severity or "warning"),
    )


def to_contract(
    result: RagRetrievalResult,
    *,
    select_top_document: bool = True,
) -> ContractRetrievalResult:
    """Translate one ``rag.schema.RetrievalResult`` into the judge contract.

    The local retriever returns a ranked candidate list (top-k), while the
    decision contract expects the documents selected for one answer.  With
    ``select_top_document`` the best document is passed as the selected one;
    the remaining candidates stay out of the coverage gates instead of
    looking like several competing editions.  Evidence is kept in full for
    fact checking.

    Broken internal references in matched source documents are kept out of
    ``reference_errors``: they are a property of the KB document, not an
    unresolved reference in the lawyer's question, and the decision gate
    rejects on any reference error.  Ambiguous entity mentions that were
    actually matched in the question are also removed; they are not missing
    references.
    """
    signals = result.signals
    documents = [_document(doc) for doc in (result.documents or [])]
    if select_top_document and documents:
        documents = documents[:1]
    applicable_editions = (
        []
        if select_top_document
        else [str(x) for x in (signals.applicable_editions or [])]
    )
    contract_signals = ContractSignals(
        low_confidence=bool(signals.low_confidence),
        max_relevance=_unit(signals.max_relevance),
        mean_relevance=_unit(signals.mean_relevance),
        mean_top3_relevance=_unit(signals.mean_top3_relevance),
        margin_top1_top2=_finite(signals.margin_top1_top2),
        distinct_docs=max(0, int(_finite(signals.distinct_docs))),
        entity_match=[str(x) for x in (signals.entity_match or [])],
        topic_match=bool(signals.topic_match),
        score_kind=_score_kind(signals),
        dense_used=bool(signals.dense_used),
        rerank_used=bool(signals.rerank_used),
        event_date=_iso_date(signals.event_date),
        applicable_editions=applicable_editions,
    )

    ambiguities: list[ContractAmbiguity] = []
    if not select_top_document:
        for item in result.ambiguity or []:
            candidates = [c for c in (_candidate(x) for x in (item.candidates or [])) if c]
            ambiguities.append(
                ContractAmbiguity(raw=str(item.raw or ""), candidates=candidates)
            )

    entity_names = {str(x).casefold() for x in (signals.entity_match or [])}
    references = [
        ref
        for ref in (_reference_error(item) for item in (result.reference_errors or []))
        if ref is not None
        and not (ref.where == "kb_document" and ref.kind == "broken_internal_ref")
        and not (select_top_document and ref.kind == "ambiguous")
        and not (ref.kind == "ambiguous" and ref.raw.casefold() in entity_names)
    ]

    unresolved = result.unresolved
    return ContractRetrievalResult(
        documents=documents,
        evidence=[_evidence(ev, i) for i, ev in enumerate(result.evidence or [])],
        signals=contract_signals,
        unresolved=ContractUnresolved(
            contract_numbers=[str(x) for x in (unresolved.contract_numbers or [])],
            party_names=[str(x) for x in (unresolved.party_names or [])],
            law_articles=[str(x) for x in (unresolved.law_articles or [])],
            clauses=[str(x) for x in (unresolved.clauses or [])],
        ),
        ambiguity=ambiguities,
        reference_errors=references,
        trace=[str(x) for x in (result.trace or [])],
    )


def to_citation_check(
    check: Any,
    retrieval: RagRetrievalResult | None = None,
) -> ContractCitationCheck | None:
    """Best-effort translation of the local citation check.

    The current RAG check stores raw citation strings, not resolved
    ``(doc_id, clause, evidence_ids)`` objects.  Strings are matched against
    retrieved evidence when possible; unmatched strings stay in the existing
    unknown/unsupported buckets.  If the input already has structured
    citations, it is validated and returned as-is.
    """
    if check is None:
        return None

    # Already structured (for a future RAG version): let pydantic validate it.
    if check.citations and all(hasattr(c, "doc_id") for c in check.citations):
        return ContractCitationCheck(
            valid=bool(check.valid),
            citations=[
                ContractCitation(
                    raw=str(c.raw),
                    doc_id=str(c.doc_id),
                    clause=str(c.clause),
                    evidence_ids=[str(x) for x in (c.evidence_ids or [])],
                )
                for c in check.citations
            ],
            unknown_docs=[str(x) for x in (check.unknown_docs or [])],
            unknown_clauses=[str(x) for x in (check.unknown_clauses or [])],
            unsupported=[str(x) for x in (check.unsupported or [])],
        )

    citations: list[ContractCitation] = []
    if retrieval is not None:
        for raw in check.citations or []:
            raw_text = str(raw).casefold()
            matched = None
            matched_index = 0
            for ev_index, ev in enumerate(retrieval.evidence or []):
                title = str(ev.title or "").casefold()
                doc_id = str(ev.doc_id or "").casefold()
                if (title and title in raw_text) or (doc_id and doc_id in raw_text):
                    matched = ev
                    matched_index = ev_index
                    break
            if matched is None:
                continue
            citations.append(
                ContractCitation(
                    raw=str(raw),
                    doc_id=str(matched.doc_id),
                    clause=str((matched.clause_ids or [""])[0]),
                    evidence_ids=[
                        f"{matched.doc_id}:{int(matched.start)}:{int(matched.end)}:{matched_index}"
                    ],
                )
            )

    return ContractCitationCheck(
        valid=bool(check.valid),
        citations=citations,
        unknown_docs=[str(x) for x in (check.unknown_docs or [])],
        unknown_clauses=[str(x) for x in (check.unknown_clauses or [])],
        unsupported=[str(x) for x in (check.unsupported or [])],
    )


__all__ = ["to_contract", "to_citation_check"]
