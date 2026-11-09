"""RAG subsystem for Meridian Law Assistant."""

from rag.answer import generate_answer
from rag.api import ApiClient
from rag.bm25 import BM25Index
from rag.chunker import Chunk, chunk_document
from rag.context import build_context_cards
from rag.index import KBIndex
from rag.loader import load_questions, load_raw_documents
from rag.metadata import DocGraph, Document, build_doc_graph, build_document_metadata
from rag.query import ParsedDialog, QueryAnalysis, analyze_query, parse_dialog
from rag.retrieve import EntityResolver, retrieve_candidates
from rag.sanitize import sanitize_text

__all__ = [
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
    "generate_answer",
    "load_questions",
    "load_raw_documents",
    "parse_dialog",
    "retrieve_candidates",
    "sanitize_text",
]
