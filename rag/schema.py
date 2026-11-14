from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMatch:
    doc_id: str
    title: str
    doc_number: str | None
    doc_type: str
    parties: list[str]
    status: str
    effective_from: str | None
    effective_to: str | None
    applicable_edition: bool
    edition_note: str
    score: float
    score_kind: str
    relevance: float  # Normalized 0.0 .. 1.0
    matched_clauses: list[str] = field(default_factory=list)
    is_direct: bool = True
    why: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    doc_id: str
    title: str
    section: str
    clause_ids: list[str]
    text: str
    start: int
    end: int
    score: float
    score_kind: str
    relevance: float  # Normalized 0.0 .. 1.0
    source: str  # bm25 | dense | graph | explicit
    why: list[str] = field(default_factory=list)


@dataclass
class Signals:
    low_confidence: bool
    max_relevance: float
    mean_relevance: float
    mean_top3_relevance: float
    margin_top1_top2: float
    distinct_docs: int
    entity_match: list[str]
    topic_match: bool
    score_kind: str
    dense_used: bool
    rerank_used: bool
    event_date: str | None
    applicable_editions: list[str] = field(default_factory=list)


@dataclass
class Unresolved:
    contract_numbers: list[str] = field(default_factory=list)
    party_names: list[str] = field(default_factory=list)
    law_articles: list[str] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)


@dataclass
class AmbiguityCandidate:
    doc_id: str
    title: str
    doc_number: str | None
    label: str


@dataclass
class Ambiguity:
    raw: str
    candidates: list[AmbiguityCandidate] = field(default_factory=list)


@dataclass
class Resolution:
    status: str  # resolved | unresolved | ambiguous
    doc_id: str | None = None
    title: str | None = None
    doc_number: str | None = None
    clause: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


@dataclass
class ReferenceError:
    kind: str  # unresolved_doc | unresolved_clause | ambiguous | edition_missing | broken_internal_ref | orphan_amendment
    where: str  # question | kb_document
    raw: str
    doc_id: str | None = None
    clause: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "warning"  # warning | error


@dataclass
class CitationCheck:
    valid: bool
    citations: list[str] = field(default_factory=list)
    unknown_docs: list[str] = field(default_factory=list)
    unknown_clauses: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def citations_found(self) -> list[str]:
        return self.citations

    @property
    def missing_citations(self) -> list[str]:
        return self.unknown_docs + self.unknown_clauses


@dataclass
class ContextResult:
    text: str
    sources: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def prompt_text(self) -> str:
        return self.text

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 3)


@dataclass
class RetrievalResult:
    documents: list[DocumentMatch]
    evidence: list[Evidence]
    signals: Signals
    unresolved: Unresolved
    ambiguity: list[Ambiguity]
    reference_errors: list[ReferenceError]
    trace: list[str] = field(default_factory=list)

    # Cached context text for orchestrator / LLM generator
    _context_cache: str | None = None

    @property
    def context(self) -> str:
        if self._context_cache is not None:
            return self._context_cache
        # Lazy import to avoid circular dependency
        from rag.context import build_context_cards
        return ""

    @property
    def low_confidence(self) -> bool:
        return self.signals.low_confidence

    @property
    def is_ambiguous(self) -> bool:
        return len(self.ambiguity) > 0

    @property
    def target_docs(self) -> list[str]:
        return [d.title for d in self.documents]

    @property
    def ranked_chunks(self) -> list[tuple[Any, float]]:
        # Helper for legacy solution.py consumers
        return [(ev, ev.score) for ev in self.evidence]
