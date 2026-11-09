from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rag import config
from rag.answer import generate_answer
from rag.api import ApiClient
from rag.context import build_context_cards
from rag.index import KBIndex
from rag.loader import load_raw_documents
from rag.metadata import DocGraph, Document, build_doc_graph, build_document_metadata
from rag.query import ParsedDialog, QueryAnalysis, analyze_query
from rag.retrieve import EntityResolver, retrieve_candidates
from rag.sanitize import sanitize_text


@dataclass
class RAGResult:
    question: str
    dialog: ParsedDialog
    analysis: QueryAnalysis
    context: str
    ranked_chunks: list
    low_confidence: bool
    is_ambiguous: bool
    target_docs: list[str]
    answer: str | None = None


class RAGPipeline:
    """High-level RAG harness interface for the team."""

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

    def retrieve(self, question: str, client: ApiClient | None = None) -> RAGResult:
        """Execute query parsing, entity resolution, and hybrid retrieval.

        Provides signals for decision making:
          - low_confidence: True when no relevant information was found in KB
          - is_ambiguous: True when organization name requires user clarification
        """
        dialog, analysis = analyze_query(question, client=client)

        dense_fn = (
            (lambda q, top_k: self.index.dense_search(q, client, top_k=top_k))
            if self.index.embeddings is not None and config.EMBED_KEY
            else None
        )

        ranked_chunks, low_conf = retrieve_candidates(
            analysis=analysis,
            chunks=self.index.chunks,
            docs=self.docs,
            graph=self.graph,
            bm25=self.index.bm25,
            resolver=self.resolver,
            dense_scores_fn=dense_fn,
            client=client,
        )

        is_amb = bool(analysis.ambiguous_entities)
        context = build_context_cards(
            ranked_chunks=ranked_chunks,
            docs=self.docs,
            graph=self.graph,
            analysis=analysis,
            is_ambiguous=is_amb,
        )

        target_names = [self.resolver.doc_map[c.doc_id].title for c, _ in ranked_chunks[:5]]

        return RAGResult(
            question=question,
            dialog=dialog,
            analysis=analysis,
            context=context,
            ranked_chunks=ranked_chunks,
            low_confidence=low_conf,
            is_ambiguous=is_amb,
            target_docs=target_names,
        )

    def ask(self, question: str, client: ApiClient | None = None) -> str:
        """Full pipeline: retrieve + build context + generate validated answer."""
        res = self.retrieve(question, client=client)
        if not client:
            with ApiClient(timeout=90.0) as cli:
                return generate_answer(
                    client=cli,
                    dialog=res.dialog,
                    analysis=res.analysis,
                    context=res.context,
                    low_confidence=res.low_confidence,
                )
        return generate_answer(
            client=client,
            dialog=res.dialog,
            analysis=res.analysis,
            context=res.context,
            low_confidence=res.low_confidence,
        )
