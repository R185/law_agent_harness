"""Retrieval result and citation-check models.

Dates are stored as ``YYYY-MM-DD`` strings or null. The run timestamp is not
stored. Callers import these types from ``contracts``.
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import Enum
from typing import Any, Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+$")

_ACTIVE_STATUS = frozenset(
    phrase.casefold()
    for phrase in ("active", "действующая", "действует")
)
_REPEALED_STATUS = frozenset(
    phrase.casefold() for phrase in ("repealed", "утратил силу")
)

_SCORE_KIND_INPUT = {
    "semantic": "semantic",
    "rerank": "semantic",
    "lexical": "lexical",
    "bm25": "lexical",
    "order_only": "order_only",
    "rrf": "order_only",
    "order": "order_only",
}


class DocumentStatus(str, Enum):
    active = "active"
    repealed = "repealed"
    unknown = "unknown"


class ScoreKind(str, Enum):
    semantic = "semantic"
    lexical = "lexical"
    order_only = "order_only"


class EvidenceSource(str, Enum):
    bm25 = "bm25"
    dense = "dense"
    graph = "graph"
    explicit = "explicit"


class ReferenceKind(str, Enum):
    unresolved_doc = "unresolved_doc"
    unresolved_clause = "unresolved_clause"
    ambiguous = "ambiguous"
    edition_missing = "edition_missing"
    broken_internal_ref = "broken_internal_ref"
    orphan_amendment = "orphan_amendment"


class ReferenceWhere(str, Enum):
    question = "question"
    kb_document = "kb_document"


def _map_status(value: Any) -> DocumentStatus:
    """Map a status phrase onto the closed enum before the field is stored.

    Latin tokens are matched case-insensitively. The Russian phrases match
    exactly after casefold. An empty value or any other input becomes
    ``unknown``.
    """

    if isinstance(value, DocumentStatus):
        return value
    if isinstance(value, str):
        folded = value.casefold()
        if folded in _ACTIVE_STATUS:
            return DocumentStatus.active
        if folded in _REPEALED_STATUS:
            return DocumentStatus.repealed
    return DocumentStatus.unknown


def _map_score_kind(value: Any) -> ScoreKind:
    """Map a scale alias onto semantic, lexical, or order_only."""

    if isinstance(value, ScoreKind):
        return value
    if isinstance(value, str):
        mapped = _SCORE_KIND_INPUT.get(value)
        if mapped is not None:
            return ScoreKind(mapped)
    raise ValueError(
        "score_kind must be semantic, lexical, or order_only"
    )


def _finite_float(value: Any) -> float:
    """Coerce a real number to float. Booleans and non-finite values fail."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    return number


def _unit_interval(value: Any) -> float:
    number = _finite_float(value)
    if number < 0 or number > 1:
        raise ValueError("must be from 0 to 1 inclusive")
    return number


def _calendar_date(value: Any) -> str | None:
    """Accept null or a real zero-padded ``YYYY-MM-DD`` calendar date."""

    if value is None:
        return None
    if not isinstance(value, str) or _ISO_DATE.fullmatch(value) is None:
        raise ValueError("must be a real calendar date YYYY-MM-DD or null")
    year_s, month_s, day_s = value.split("-")
    try:
        date(int(year_s), int(month_s), int(day_s))
    except ValueError as exc:
        raise ValueError("must be a real calendar date YYYY-MM-DD or null") from exc
    return value


def _distinct_docs(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("distinct_docs must be an integer >= 0")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("distinct_docs must be an integer >= 0")
        number = int(value)
    elif isinstance(value, str) and _INTEGER_TEXT.fullmatch(value.strip()) is not None:
        number = int(value)
    else:
        raise ValueError("distinct_docs must be an integer >= 0")
    if number < 0:
        raise ValueError("distinct_docs must be an integer >= 0")
    return number


_Status = Annotated[DocumentStatus, BeforeValidator(_map_status)]
_ScoreKindInput = Annotated[ScoreKind, BeforeValidator(_map_score_kind)]
_FiniteFloat = Annotated[float, BeforeValidator(_finite_float)]
_UnitInterval = Annotated[float, BeforeValidator(_unit_interval)]
_IsoDate = Annotated[str | None, BeforeValidator(_calendar_date)]
_DistinctDocs = Annotated[int, BeforeValidator(_distinct_docs)]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class DocumentMatch(_ContractModel):
    doc_id: str
    title: str
    doc_number: str = ""
    doc_type: str
    parties: list[str]
    status: _Status
    effective_from: _IsoDate = None
    effective_to: _IsoDate = None
    applicable_edition: bool
    edition_note: str = ""
    matched_clauses: list[str]
    is_direct: bool
    why: list[str]


class Evidence(_ContractModel):
    evidence_id: str
    doc_id: str
    title: str
    section: str
    clause_ids: list[str]
    text: str
    start: int
    end: int
    score: _FiniteFloat
    score_kind: _ScoreKindInput
    relevance: _UnitInterval
    source: EvidenceSource
    why: list[str]

    @model_validator(mode="after")
    def _end_not_before_start(self) -> Evidence:
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class Signals(_ContractModel):
    """Typed signals. An unknown key is ignored and is not stored."""

    low_confidence: bool
    max_relevance: _UnitInterval
    mean_relevance: _UnitInterval
    mean_top3_relevance: _UnitInterval
    margin_top1_top2: _FiniteFloat
    distinct_docs: _DistinctDocs
    entity_match: list[str]
    topic_match: bool
    score_kind: _ScoreKindInput
    dense_used: bool
    rerank_used: bool
    event_date: _IsoDate
    applicable_editions: list[str]


class Unresolved(_ContractModel):
    contract_numbers: list[str] = Field(default_factory=list)
    party_names: list[str] = Field(default_factory=list)
    law_articles: list[str] = Field(default_factory=list)
    clauses: list[str] = Field(default_factory=list)


class Candidate(_ContractModel):
    doc_id: str
    title: str
    doc_number: str
    label: str


class Ambiguity(_ContractModel):
    raw: str
    candidates: list[Candidate]


class ReferenceError(_ContractModel):
    kind: ReferenceKind
    where: ReferenceWhere
    raw: str
    doc_id: str | None = None
    clause: str | None = None
    candidates: list[str] = Field(default_factory=list)
    severity: str


class RetrievalResult(_ContractModel):
    documents: list[DocumentMatch] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    signals: Signals
    unresolved: Unresolved = Field(default_factory=Unresolved)
    ambiguity: list[Ambiguity] = Field(default_factory=list)
    reference_errors: list[ReferenceError] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class Citation(_ContractModel):
    raw: str
    doc_id: str
    clause: str
    evidence_ids: list[str]


class CitationCheck(_ContractModel):
    """A finished citation check. A missing check is None, not this model."""

    valid: bool
    citations: list[Citation]
    unknown_docs: list[str]
    unknown_clauses: list[str]
    unsupported: list[str]


__all__ = [
    "DocumentStatus",
    "ScoreKind",
    "EvidenceSource",
    "ReferenceKind",
    "ReferenceWhere",
    "DocumentMatch",
    "Evidence",
    "Signals",
    "Unresolved",
    "Candidate",
    "Ambiguity",
    "ReferenceError",
    "RetrievalResult",
    "Citation",
    "CitationCheck",
]
