"""RAG subsystem for Meridian Law Assistant."""

# ---------------------------------------------------------------------------
# 1. Публичный API-контракт RAG (Core RAG Contract & Entrypoints)
# ---------------------------------------------------------------------------
from rag.pipeline import (
    RAGPipeline,
    build_context,
    get_clause,
    get_document,
    get_pipeline,
    resolve_reference,
    retrieve,
    verify_citations,
)

# ---------------------------------------------------------------------------
# 2. Типизированные схемы данных (Data Schemas & Contracts)
# ---------------------------------------------------------------------------
from rag.schema import (
    Ambiguity,
    AmbiguityCandidate,
    CitationCheck,
    ContextResult,
    DocumentMatch,
    Evidence,
    ReferenceError,
    RetrievalResult,
    RetrievedChunk,
    Signals,
    Unresolved,
)

# ---------------------------------------------------------------------------
# 3. Индексация и поиск (Search & Indexing Components)
# ---------------------------------------------------------------------------
from rag.api import ApiClient
from rag.bm25 import BM25Index
from rag.chunker import Chunk, chunk_document
from rag.context import build_context_cards
from rag.index import KBIndex
from rag.query import ParsedDialog, QueryAnalysis, analyze_query, parse_dialog
from rag.retrieve import EntityResolver, retrieve_candidates

# ---------------------------------------------------------------------------
# 4. Метаданные, загрузчики и нормализация (Metadata, Loaders & Text Utilities)
# ---------------------------------------------------------------------------
from rag.loader import find_file, load_questions, load_raw_documents
from rag.metadata import (
    DocGraph,
    Document,
    build_doc_graph,
    build_document_metadata,
    resolve_applicable_edition,
)
from rag.text import normalize_whitespace

__all__ = [
    # Core RAG Contract
    "retrieve",
    "build_context",
    "resolve_reference",
    "verify_citations",
    "get_document",
    "get_clause",
    "get_pipeline",
    "RAGPipeline",
    "RetrievalResult",
    "RetrievedChunk",
    "DocumentMatch",
    "Evidence",
    "Signals",
    "Unresolved",
    "Ambiguity",
    "AmbiguityCandidate",
    "ReferenceError",
    "CitationCheck",
    "ContextResult",
    # Underlying components
    "ApiClient",
    "BM25Index",
    "Chunk",
    "DocGraph",
    "Document",
    "EntityResolver",
    "KBIndex",
    "ParsedDialog",
    "QueryAnalysis",
    "analyze_query",
    "build_context_cards",
    "build_doc_graph",
    "build_document_metadata",
    "chunk_document",
    "find_file",
    "load_questions",
    "load_raw_documents",
    "normalize_whitespace",
    "parse_dialog",
    "resolve_applicable_edition",
    "retrieve_candidates",
]
