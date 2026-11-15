"""Public retrieval and citation contracts.

Import these types from ``contracts``. The decision layer does not import
them from a search package.
"""

from contracts.retrieval import (
    Ambiguity,
    Candidate,
    Citation,
    CitationCheck,
    DocumentMatch,
    DocumentStatus,
    Evidence,
    EvidenceSource,
    ReferenceError,
    ReferenceKind,
    ReferenceWhere,
    RetrievalResult,
    ScoreKind,
    Signals,
    Unresolved,
)

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
