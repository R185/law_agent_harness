from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from rag.chunker import Chunk
from rag.text import tokenize


class BM25Index:
    """Fast, dependency-free BM25Okapi implementation for chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lens: list[int] = []
        self.doc_freqs: dict[str, int] = {}  # term -> number of documents containing term
        self.term_freqs: list[dict[str, int]] = []  # chunk_idx -> {term: count}
        self.idf: dict[str, float] = {}

    def fit(self, chunks: Sequence[Chunk]) -> BM25Index:
        """Index a sequence of Chunk objects using chunk.full text."""
        self.corpus_size = len(chunks)
        self.doc_lens = []
        self.doc_freqs = {}
        self.term_freqs = []

        total_tokens = 0
        for chunk in chunks:
            tokens = tokenize(chunk.full)
            t_len = len(tokens)
            self.doc_lens.append(t_len)
            total_tokens += t_len

            counts = Counter(tokens)
            self.term_freqs.append(counts)
            for term in counts.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avgdl = (total_tokens / self.corpus_size) if self.corpus_size > 0 else 1.0

        # Precompute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        return self

    def score(self, query_tokens: Sequence[str]) -> list[float]:
        """Compute BM25 scores for all chunks against given query tokens."""
        scores = [0.0] * self.corpus_size
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl

        # Score only terms that exist in corpus
        q_terms = [t for t in query_tokens if t in self.idf]
        if not q_terms:
            return scores

        for idx in range(self.corpus_size):
            doc_len = self.doc_lens[idx]
            tf_map = self.term_freqs[idx]
            len_norm = k1 * (1.0 - b + b * (doc_len / avgdl))

            s = 0.0
            for term in q_terms:
                tf = tf_map.get(term, 0)
                if tf > 0:
                    idf_val = self.idf[term]
                    s += idf_val * (tf * (k1 + 1.0)) / (tf + len_norm)
            scores[idx] = s

        return scores

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        """Search top-k chunk indices and BM25 scores for query string."""
        tokens = tokenize(query)
        scores = self.score(tokens)
        # Find indices of non-zero scores sorted descending
        indexed_scores = [(i, scores[i]) for i in range(self.corpus_size) if scores[i] > 0.0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]
