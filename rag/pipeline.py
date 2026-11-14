from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rag import config
from rag.answer import generate_answer
from rag.api import ApiClient
from rag.context import build_context_cards
from rag.index import KBIndex
from rag.loader import load_raw_documents
from rag.metadata import DocGraph, Document, build_doc_graph, build_document_metadata
from rag.query import ParsedDialog, QueryAnalysis, analyze_query, parse_dialog
from rag.retrieve import EntityResolver, retrieve_candidates
from rag.sanitize import sanitize_text
from rag.schema import (
    CitationCheck,
    ContextResult,
    DocumentMatch,
    Evidence,
    RetrievalResult,
)
from rag.tools import (
    build_context as tool_build_context,
    get_clause as tool_get_clause,
    get_document as tool_get_document,
    resolve_reference as tool_resolve_reference,
    verify_citations as tool_verify_citations,
)


class RAGPipeline:
    """Unified knowledge-only RAG subsystem conforming to the project contract."""

    def __init__(self, source_path: Path | None = None, index_dir: Path | None = None):
        raw_docs = load_raw_documents(source_path)
        self.docs: list[Document] = []
        for rd in raw_docs:
            clean_text, removed = sanitize_text(rd.text, rd.path)
            doc = build_document_metadata(rd.path, clean_text, removed)
            self.docs.append(doc)

        self.graph: DocGraph = build_doc_graph(self.docs)
        self.resolver: EntityResolver = EntityResolver(self.docs, self.graph)

        self.index: KBIndex = KBIndex()
        self.index.build_chunks_and_bm25(self.docs)

        idx_dir = index_dir or (Path(__file__).resolve().parent.parent / "index")
        if idx_dir.is_dir():
            self.index.load_precomputed_embeddings(idx_dir)

    def retrieve(
        self,
        question: str,
        history: str | None = None,
        top_k: int = 10,
        client: ApiClient | None = None,
    ) -> RetrievalResult:
        """Core RAG retrieval function.

        Takes question and optional history, runs BM25 + dense + graph + rerank,
        and returns structured RetrievalResult (documents, evidence, signals, unresolved, ambiguity).
        When client is None, operates deterministically on BM25 and graph without network.
        """
        # Compose dialog representation
        if history:
            full_input = f"{history.strip()}\nПользователь: {question.strip()}"
        else:
            full_input = question.strip()

        # Extract entities using KB known parties to avoid hardcoding
        known_parties = self.resolver.get_known_parties()
        dialog, analysis = analyze_query(
            full_input, client=client, known_parties=known_parties
        )

        dense_fn = (
            (lambda q, k: self.index.dense_search(q, client, top_k=k))
            if self.index.embeddings is not None and config.EMBED_KEY
            else None
        )

        result: RetrievalResult = retrieve_candidates(
            analysis=analysis,
            chunks=self.index.chunks,
            docs=self.docs,
            graph=self.graph,
            bm25=self.index.bm25,
            resolver=self.resolver,
            dense_scores_fn=dense_fn,
            client=client,
            top_k=top_k,
        )

        # Precompute context string for quick access via result.context
        context_res = tool_build_context(result, max_chars=config.CONTEXT_MAX_CHARS)
        result._context_cache = context_res.text

        return result

    def build_context(
        self,
        result: RetrievalResult,
        max_chars: int = 12000,
        token_budget: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ContextResult:
        """Format retrieval result into text cards, sources, evidence and warnings."""
        return tool_build_context(
            result, max_chars=max_chars, token_budget=token_budget, max_tokens=max_tokens
        )

    def resolve_reference(self, raw: str) -> Any:
        """Resolve a raw reference string like 'п. 3.2 договора № П-17/2024' or 'ст. 70 ТК РФ'."""
        return tool_resolve_reference(raw, docs=self.docs)

    def verify_citations(self, answer_text: str, context: str | None = None) -> CitationCheck:
        """Verify legal citations in the answer against indexed KB and context."""
        return tool_verify_citations(answer_text, context=context, docs=self.docs)

    def get_document(self, doc_id: str) -> DocumentMatch | None:
        """Retrieve DocumentMatch by doc_id."""
        return tool_get_document(doc_id, docs=self.docs)

    def get_clause(self, doc_id: str, clause: str) -> Evidence | None:
        """Retrieve specific clause Evidence by doc_id and clause number."""
        return tool_get_clause(doc_id, clause, docs=self.docs)

    def ask(self, question: str, history: str | None = None, client: ApiClient | None = None) -> str:
        """End-to-end convenience method: retrieve + generate answer (for standalone CLI / tests)."""
        res = self.retrieve(question, history=history, client=client)
        dialog = parse_dialog(f"{history}\n{question}" if history else question)
        if not client:
            with ApiClient(timeout=90.0) as cli:
                return generate_answer(
                    client=cli,
                    dialog=dialog,
                    analysis=res.signals,  # type: ignore
                    context=res.context,
                    low_confidence=res.signals.low_confidence,
                )
        return generate_answer(
            client=client,
            dialog=dialog,
            analysis=res.signals,  # type: ignore
            context=res.context,
            low_confidence=res.signals.low_confidence,
        )


# Global default instance for top-level retrieve() calls
_default_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def retrieve(
    question: str,
    history: str | None = None,
    top_k: int = 10,
    client: ApiClient | None = None,
) -> RetrievalResult:
    """Module-level retrieve entry point matching the exact PR #1 contract."""
    return get_pipeline().retrieve(question, history=history, top_k=top_k, client=client)


def build_context(
    result: RetrievalResult,
    max_chars: int = 12000,
    token_budget: int | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> ContextResult:
    return tool_build_context(
        result, max_chars=max_chars, token_budget=token_budget, max_tokens=max_tokens
    )


def resolve_reference(raw: str) -> Any:
    return get_pipeline().resolve_reference(raw)


def verify_citations(answer_text: str, context: str | None = None) -> CitationCheck:
    return get_pipeline().verify_citations(answer_text, context=context)


def get_document(doc_id: str) -> DocumentMatch | None:
    return get_pipeline().get_document(doc_id)


def get_clause(doc_id: str, clause: str) -> Evidence | None:
    return get_pipeline().get_clause(doc_id, clause)
