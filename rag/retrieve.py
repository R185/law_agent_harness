from __future__ import annotations

import re
import sys
from typing import Sequence

from rag import config
from rag.api import ApiClient
from rag.bm25 import BM25Index
from rag.chunker import Chunk
from rag.metadata import DocGraph, Document
from rag.query import QueryAnalysis
from rag.text import _stem_fn, levenshtein_distance, norm_str, tokenize


class EntityResolver:
    """Resolve extracted organization names and document numbers into target and confusable doc IDs."""

    def __init__(self, docs: Sequence[Document], graph: DocGraph):
        self.docs = docs
        self.graph = graph
        self.doc_map: dict[str, Document] = {d.doc_id: d for d in docs}
        self.num_to_doc: dict[str, str] = {}
        self.org_to_docs: dict[str, set[str]] = {}

        for d in docs:
            for num in d.meta.get("doc_numbers", []):
                self.num_to_doc[num] = d.doc_id

            for party in d.meta.get("parties", []):
                p_name = norm_str(party["name"]).strip()
                if p_name:
                    self.org_to_docs.setdefault(p_name, set()).add(d.doc_id)

    def resolve(
        self, analysis: QueryAnalysis
    ) -> tuple[set[str], set[str], bool]:
        """Returns (target_doc_ids, confusable_doc_ids, is_ambiguous)."""
        target_doc_ids: set[str] = set()
        confusable_doc_ids: set[str] = set()
        is_ambiguous = bool(analysis.ambiguous_entities)

        # 1. Match document numbers directly
        for num in analysis.entities.get("doc_numbers", []):
            canon_num = num.lower().strip()
            if canon_num in self.num_to_doc:
                target_doc_ids.add(self.num_to_doc[canon_num])

        # 2. Match organizations
        queried_orgs = analysis.entities.get("organizations", [])
        for q_org in queried_orgs:
            norm_q = norm_str(q_org).strip()
            if not norm_q:
                continue

            matched_this_org: set[str] = set()

            # Exact match
            if norm_q in self.org_to_docs:
                matched_this_org.update(self.org_to_docs[norm_q])
            else:
                q_words = [w for w in tokenize(norm_q) if w not in {"ооо", "ао", "пао", "зао", "ип"}]
                q_stems = {_stem_fn(w) for w in q_words}
                # Check token stem exact set match and prefix matching
                for known_org, doc_ids in self.org_to_docs.items():
                    k_words = [w for w in tokenize(known_org) if w not in {"ооо", "ао", "пао", "зао", "ип"}]
                    k_stems = {_stem_fn(w) for w in k_words}
                    if q_stems and q_stems == k_stems:
                        matched_this_org.update(doc_ids)
                    elif known_org.startswith(norm_q) or norm_q.startswith(known_org):
                        matched_this_org.update(doc_ids)
                    elif len(norm_q) >= 8 and levenshtein_distance(norm_q, known_org) <= 1:
                        matched_this_org.update(doc_ids)

            # If user queried generic prefix like "альфа", mark ambiguous
            if len(matched_this_org) > 1 and any(norm_q in amb.lower() for amb in analysis.ambiguous_entities + ["альфа"]):
                is_ambiguous = True

            target_doc_ids.update(matched_this_org)

            # Find confusables (similar organizations that were NOT targeted)
            for known_org, doc_ids in self.org_to_docs.items():
                if doc_ids.issubset(target_doc_ids):
                    continue
                # Common prefix >= 5 (e.g. "альфа-") or Levenshtein <= 3
                common_prefix_len = 0
                for c1, c2 in zip(norm_q, known_org):
                    if c1 == c2:
                        common_prefix_len += 1
                    else:
                        break
                if common_prefix_len >= 5 or levenshtein_distance(norm_q, known_org) <= 3:
                    confusable_doc_ids.update(doc_ids)

        # 3. Expand target docs with amendments and supplements via graph
        expanded_targets = set(target_doc_ids)
        for doc_id in target_doc_ids:
            for neighbor_id, rel in self.graph.get_neighbors(doc_id):
                if rel in ("amended_by", "supplemented_by"):
                    expanded_targets.add(neighbor_id)
        target_doc_ids = expanded_targets

        return target_doc_ids, confusable_doc_ids, is_ambiguous


def score_date_version(doc: Document, event_date_str: str | None) -> float:
    """Determine multiplier for document based on event date and effective interval."""
    if not event_date_str:
        return 1.0

    eff_from = doc.meta.get("effective_from") or "0000-00-00"
    eff_to = doc.meta.get("effective_to") or "9999-99-99"

    # Compare lexicographically (YYYY-MM or YYYY-MM-DD)
    e_date = event_date_str[:7]
    f_date = eff_from[:7]
    t_date = eff_to[:7]

    if f_date <= e_date <= t_date:
        return config.BOOST_EVENT_VERSION
    return config.PENALTY_REPEALED


def rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]], rrf_k: int = 60
) -> dict[int, float]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    rrf_scores: dict[int, float] = {}
    for r_list in ranked_lists:
        for rank, (chunk_idx, _) in enumerate(r_list):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + (1.0 / (rrf_k + rank + 1))
    return rrf_scores


def retrieve_candidates(
    analysis: QueryAnalysis,
    chunks: Sequence[Chunk],
    docs: Sequence[Document],
    graph: DocGraph,
    bm25: BM25Index,
    resolver: EntityResolver,
    dense_scores_fn: Any = None,  # optional (query_str) -> list[tuple[int, float]]
    client: ApiClient | None = None,
) -> tuple[list[tuple[Chunk, float]], bool]:
    """Hybrid retrieval, entity boosts/penalties, graph expansion, and reranking."""
    target_doc_ids, confusable_doc_ids, is_ambiguous = resolver.resolve(analysis)
    doc_map = {d.doc_id: d for d in docs}
    chunk_map = {c.chunk_id: c for c in chunks}
    doc_to_chunks: dict[str, list[Chunk]] = {}
    for c in chunks:
        doc_to_chunks.setdefault(c.doc_id, []).append(c)

    # 1. Multi-query search with BM25 (and Dense if available)
    ranked_lists: list[list[tuple[int, float]]] = []
    queries = analysis.search_queries or [analysis.standalone_question]
    best_raw_bm25 = 0.0

    for q in queries:
        bm_hits = bm25.search(q, top_k=config.BM25_TOP_K)
        if bm_hits:
            ranked_lists.append(bm_hits)
            if bm_hits[0][1] > best_raw_bm25:
                best_raw_bm25 = bm_hits[0][1]
        if dense_scores_fn is not None:
            dense_hits = dense_scores_fn(q, top_k=config.DENSE_TOP_K)
            if dense_hits:
                ranked_lists.append(dense_hits)

    if not ranked_lists:
        # Fallback to standalone question
        fallback_hits = bm25.search(analysis.standalone_question, top_k=config.BM25_TOP_K)
        ranked_lists.append(fallback_hits)
        if fallback_hits:
            best_raw_bm25 = max(best_raw_bm25, fallback_hits[0][1])

    # Initial low confidence check based on lexical match strength
    low_confidence = (best_raw_bm25 < 7.0)

    # 2. RRF Fusion
    fused_scores = rrf_fuse(ranked_lists, rrf_k=config.RRF_K)

    # 3. Apply deterministic boosts and penalties
    adjusted_scores: list[tuple[int, float]] = []
    for idx, score in fused_scores.items():
        chunk = chunks[idx]
        doc = doc_map.get(chunk.doc_id)
        multiplier = 1.0

        if doc:
            # Target entity boost
            if chunk.doc_id in target_doc_ids:
                multiplier *= config.BOOST_TARGET
            elif chunk.doc_id in confusable_doc_ids and not is_ambiguous:
                multiplier *= config.PENALTY_CONFUSABLE

            # Validity date / status adjustments
            if doc.meta.get("status") == "repealed":
                if analysis.event_date:
                    multiplier *= score_date_version(doc, analysis.event_date)
                else:
                    multiplier *= config.PENALTY_REPEALED

            # Explicit law match boost
            law_code = doc.meta.get("law_code", "").lower()
            art = doc.meta.get("article", "")
            if law_code and (law_code in analysis.standalone_question.lower() or f"ст. {art}" in analysis.standalone_question.lower()):
                multiplier *= 1.5

        adjusted_scores.append((idx, score * multiplier))

    adjusted_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = adjusted_scores[:config.FUSED_TOP_K]

    # 4. Graph & neighbor expansion
    candidate_chunks: dict[str, tuple[Chunk, float]] = {
        chunks[idx].chunk_id: (chunks[idx], score) for idx, score in top_candidates
    }

    # Identify top docs with their best scores
    top_docs_in_cand: dict[str, float] = {}
    for idx, score in top_candidates[:10]:
        d_id = chunks[idx].doc_id
        if d_id not in top_docs_in_cand or score > top_docs_in_cand[d_id]:
            top_docs_in_cand[d_id] = score

    # (a) Add related documents (amended_by, supplemented_by, superseded_by/supersedes)
    for doc_id, parent_score in top_docs_in_cand.items():
        for neighbor_id, rel in graph.get_neighbors(doc_id):
            n_chunks = doc_to_chunks.get(neighbor_id, [])
            rel_boost = parent_score * 0.75
            if rel in ("amended_by", "supplemented_by", "superseded_by", "supersedes"):
                for nc in n_chunks:
                    if nc.chunk_id not in candidate_chunks:
                        candidate_chunks[nc.chunk_id] = (nc, rel_boost)
                    else:
                        curr_c, curr_s = candidate_chunks[nc.chunk_id]
                        candidate_chunks[nc.chunk_id] = (curr_c, max(curr_s, rel_boost))

    # (b) Add neighboring chunks (seq +- 1) if chunk is short or incomplete
    for idx, parent_score in top_candidates[:10]:
        c = chunks[idx]
        doc_chunks = doc_to_chunks.get(c.doc_id, [])
        if len(c.text) < 500 or not c.text.rstrip().endswith("."):
            for offset in (-1, 1):
                target_seq = c.seq + offset
                for nc in doc_chunks:
                    if nc.seq == target_seq and nc.chunk_id not in candidate_chunks:
                        if len(candidate_chunks) < config.MAX_CANDIDATES:
                            candidate_chunks[nc.chunk_id] = (nc, parent_score * 0.5)

    # (c) Ensure target doc is present if identified
    if target_doc_ids:
        has_target = any(c.doc_id in target_doc_ids for c, _ in candidate_chunks.values())
        if not has_target:
            best_cand_score = max((s for _, s in candidate_chunks.values()), default=0.01)
            for t_id in target_doc_ids:
                t_chunks = doc_to_chunks.get(t_id, [])
                for tc in t_chunks[:3]:
                    candidate_chunks[tc.chunk_id] = (tc, best_cand_score * 0.8)

    # List of candidate chunks sorted by fused/boosted score up to MAX_CANDIDATES
    cand_list = sorted(candidate_chunks.values(), key=lambda x: x[1], reverse=True)[:config.MAX_CANDIDATES]

    # 5. Reranking step
    rerank_scores: list[tuple[Chunk, float]] = []

    if client and config.RERANK_KEY and len(cand_list) > 1:
        query_text = analysis.standalone_question
        if config.RERANK_WITH_INSTRUCTION:
            query_text = (
                "Найди фрагмент внутреннего документа или закона, который отвечает на юридический вопрос. Вопрос: "
                + analysis.standalone_question
            )
        doc_texts = [c.full[:2000] for c, _ in cand_list]

        try:
            raw_ranks = client.rerank(query_text, doc_texts, top_n=len(doc_texts))
            score_map = {idx: s for idx, s in raw_ranks}
            max_r_score = max(score_map.values()) if score_map else 0.0

            if max_r_score < config.RERANK_EMPTY_THRESHOLD:
                low_confidence = True

            for pos, (c, _) in enumerate(cand_list):
                base_score = score_map.get(pos, 0.0)
                # Soft boost adjustment
                mod = 1.0
                if c.doc_id in target_doc_ids:
                    mod *= 1.25
                elif c.doc_id in confusable_doc_ids and not is_ambiguous:
                    mod *= 0.5
                rerank_scores.append((c, base_score * mod))

            rerank_scores.sort(key=lambda x: x[1], reverse=True)
        except Exception as exc:
            print(f"[RERANK] Warning: reranker failed ({exc}); falling back to fused scores", file=sys.stderr)
            rerank_scores = cand_list
    else:
        rerank_scores = cand_list

    # Ensure related graph documents (amendments/supplements/versions) from top-5 are preserved in context
    top_context = rerank_scores[:config.CONTEXT_TOP_N]
    top_doc_ids = {c.doc_id for c, _ in top_context[:5]}

    for c, score in rerank_scores[config.CONTEXT_TOP_N:]:
        for top_id in top_doc_ids:
            if any(tid == c.doc_id and rel in ("amended_by", "supplemented_by", "superseded_by", "supersedes") for tid, rel in graph.get_neighbors(top_id)):
                top_context.append((c, score))
                break

    return top_context, low_confidence
