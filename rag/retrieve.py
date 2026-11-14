from __future__ import annotations

import math
import re
import sys
from typing import Any, Sequence

from rag import config
from rag.api import ApiClient
from rag.bm25 import BM25Index
from rag.chunker import Chunk
from rag.metadata import DocGraph, Document, resolve_applicable_edition
from rag.query import ParsedDialog, QueryAnalysis, analyze_query
from rag.schema import (
    Ambiguity,
    AmbiguityCandidate,
    DocumentMatch,
    Evidence,
    ReferenceError,
    RetrievalResult,
    Signals,
    Unresolved,
)
from rag.text import _stem_fn, levenshtein_distance, norm_str, tokenize


class EntityResolver:
    """Resolve extracted organization names, document numbers, and law articles against the KB."""

    def __init__(self, docs: Sequence[Document], graph: DocGraph):
        self.docs = docs
        self.graph = graph
        self.doc_map: dict[str, Document] = {d.doc_id: d for d in docs}
        self.num_to_doc: dict[str, str] = {}
        self.org_to_docs: dict[str, set[str]] = {}
        self.law_articles: set[str] = set()

        for d in docs:
            for num in d.meta.get("doc_numbers", []):
                self.num_to_doc[num] = d.doc_id

            for party in d.meta.get("parties", []):
                p_name = norm_str(party["name"]).strip()
                if p_name:
                    self.org_to_docs.setdefault(p_name, set()).add(d.doc_id)

            if d.category == "law":
                code = d.meta.get("law_code", "")
                art = d.meta.get("article", "")
                if code and art:
                    self.law_articles.add(f"{code} ст. {art}".lower())
                    self.law_articles.add(f"ст. {art}".lower())

    def get_known_parties(self) -> list[str]:
        """Return list of all clean party names from indexed documents."""
        parties = []
        for d in self.docs:
            for p in d.meta.get("parties", []):
                name = p.get("name", "").strip()
                if name and name not in parties:
                    parties.append(name)
        return parties

    def resolve(
        self, analysis: QueryAnalysis
    ) -> tuple[set[str], set[str], list[Ambiguity], Unresolved, list[ReferenceError], list[str]]:
        """Resolve query entities, returning targets, confusables, ambiguity, unresolved, and errors."""
        target_doc_ids: set[str] = set()
        confusable_doc_ids: set[str] = set()
        ambiguities: list[Ambiguity] = []
        unresolved = Unresolved()
        ref_errors: list[ReferenceError] = []
        trace: list[str] = []

        # 1. Match document numbers directly
        for num in analysis.entities.get("doc_numbers", []):
            canon_num = num.lower().strip()
            if canon_num in self.num_to_doc:
                matched_id = self.num_to_doc[canon_num]
                target_doc_ids.add(matched_id)
                trace.append(f"Entity: doc_number '{num}' matched {self.doc_map[matched_id].title}")
            else:
                unresolved.contract_numbers.append(num)
                ref_errors.append(
                    ReferenceError(
                        kind="unresolved_doc",
                        where="question",
                        raw=num,
                        candidates=[],
                        severity="error",
                    )
                )
                trace.append(f"Entity: doc_number '{num}' UNRESOLVED")

        # 2. Match organizations
        queried_orgs = analysis.entities.get("organizations", [])
        for q_org in queried_orgs:
            norm_q = norm_str(q_org).strip()
            if not norm_q or norm_q in {"ооо", "ао", "пао", "зао", "ип"}:
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

            if not matched_this_org:
                unresolved.party_names.append(q_org)
                ref_errors.append(
                    ReferenceError(
                        kind="unresolved_doc",
                        where="question",
                        raw=q_org,
                        candidates=[],
                        severity="warning",
                    )
                )
                trace.append(f"Entity: organization '{q_org}' UNRESOLVED")
                continue

            # Check ambiguity (multiple candidates matching single user query term)
            if len(matched_this_org) > 1:
                cand_items = []
                for d_id in matched_this_org:
                    d = self.doc_map[d_id]
                    label = d.title
                    for p in d.meta.get("parties", []):
                        if norm_q in norm_str(p["name"]) or norm_str(p["name"]) in norm_q:
                            role = p.get("role") or p.get("form") or ""
                            label = f"{role}: {p['name']}".strip(": ")
                            break
                    d_num = d.meta.get("doc_numbers", [None])[0] if d.meta.get("doc_numbers") else None
                    cand_items.append(
                        AmbiguityCandidate(
                            doc_id=d.doc_id,
                            title=d.title,
                            doc_number=d_num,
                            label=label,
                        )
                    )
                amb = Ambiguity(raw=q_org, candidates=cand_items)
                ambiguities.append(amb)
                ref_errors.append(
                    ReferenceError(
                        kind="ambiguous",
                        where="question",
                        raw=q_org,
                        candidates=[{"doc_id": c.doc_id, "title": c.title, "label": c.label} for c in cand_items],
                        severity="warning",
                    )
                )
                trace.append(f"Entity: organization '{q_org}' is AMBIGUOUS between {len(cand_items)} documents")

            target_doc_ids.update(matched_this_org)

            # Find confusables (similar organizations that were NOT targeted)
            for known_org, doc_ids in self.org_to_docs.items():
                if doc_ids.issubset(target_doc_ids):
                    continue
                common_prefix_len = 0
                for c1, c2 in zip(norm_q, known_org):
                    if c1 == c2:
                        common_prefix_len += 1
                    else:
                        break
                if common_prefix_len >= 5 or levenshtein_distance(norm_q, known_org) <= 3:
                    confusable_doc_ids.update(doc_ids)

        # 3. Match law articles
        for law in analysis.entities.get("laws", []):
            clean_l = norm_str(law).strip()
            if clean_l and not any(clean_l in k or k in clean_l for k in self.law_articles):
                unresolved.law_articles.append(law)
                ref_errors.append(
                    ReferenceError(
                        kind="unresolved_doc",
                        where="question",
                        raw=law,
                        candidates=[],
                        severity="warning",
                    )
                )
                trace.append(f"Entity: law article '{law}' UNRESOLVED in database")

        # 4. Expand target docs with amendments and supplements via graph
        expanded_targets = set(target_doc_ids)
        for doc_id in target_doc_ids:
            for neighbor_id, rel in self.graph.get_neighbors(doc_id):
                if rel in ("amended_by", "supplemented_by"):
                    expanded_targets.add(neighbor_id)
                    trace.append(f"Graph expansion: added target {self.doc_map[neighbor_id].title} via {rel}")
        target_doc_ids = expanded_targets

        return target_doc_ids, confusable_doc_ids, ambiguities, unresolved, ref_errors, trace


def score_date_version(doc: Document, event_date_str: str | None) -> float:
    """Determine multiplier for document based on event date and effective interval."""
    if not event_date_str:
        return 1.0

    eff_from = doc.meta.get("effective_from") or "0000-00-00"
    eff_to = doc.meta.get("effective_to") or "9999-99-99"

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


def normalize_relevance(raw_score: float, score_kind: str) -> float:
    """Map raw scores onto unified 0.0 .. 1.0 relevance scale."""
    if score_kind == "rerank":
        return max(0.0, min(1.0, float(raw_score)))
    elif score_kind == "dense":
        return max(0.0, min(1.0, float(raw_score)))
    elif score_kind == "bm25":
        if raw_score <= 0.0:
            return 0.0
        return float(raw_score / (raw_score + 20.0))
    else:
        if raw_score <= 0.0:
            return 0.0
        return min(1.0, float(raw_score * 35.0))


def retrieve_candidates(
    analysis: QueryAnalysis,
    chunks: Sequence[Chunk],
    docs: Sequence[Document],
    graph: DocGraph,
    bm25: BM25Index,
    resolver: EntityResolver,
    dense_scores_fn: Any = None,
    client: ApiClient | None = None,
    top_k: int = 10,
) -> RetrievalResult:
    """Execute full knowledge retrieval fulfilling the unified RAG contract."""
    trace: list[str] = []
    (
        target_doc_ids,
        confusable_doc_ids,
        ambiguities,
        unresolved,
        ref_errors,
        resolver_trace,
    ) = resolver.resolve(analysis)
    trace.extend(resolver_trace)

    doc_map = {d.doc_id: d for d in docs}
    doc_to_chunks: dict[str, list[Chunk]] = {}
    for c in chunks:
        doc_to_chunks.setdefault(c.doc_id, []).append(c)

    is_ambiguous = len(ambiguities) > 0

    # 1. Multi-query search with BM25 (and Dense if available)
    ranked_lists: list[list[tuple[int, float]]] = []
    queries = analysis.search_queries or [analysis.standalone_question]
    best_raw_bm25 = 0.0
    dense_used = False

    for q in queries:
        bm_hits = bm25.search(q, top_k=config.BM25_TOP_K)
        if bm_hits:
            ranked_lists.append(bm_hits)
            if bm_hits[0][1] > best_raw_bm25:
                best_raw_bm25 = bm_hits[0][1]
            trace.append(f"BM25 query '{q[:40]}': top score {bm_hits[0][1]:.2f}")

        if dense_scores_fn is not None:
            try:
                dense_hits = dense_scores_fn(q, top_k=config.DENSE_TOP_K)
                if dense_hits:
                    ranked_lists.append(dense_hits)
                    dense_used = True
                    trace.append(f"Dense query '{q[:40]}': top score {dense_hits[0][1]:.2f}")
            except Exception as exc:
                trace.append(f"Dense query error: {exc}")

    if not ranked_lists:
        fallback_hits = bm25.search(analysis.standalone_question, top_k=config.BM25_TOP_K)
        ranked_lists.append(fallback_hits)
        if fallback_hits:
            best_raw_bm25 = max(best_raw_bm25, fallback_hits[0][1])

    score_kind = "rrf" if dense_used else "bm25"
    fused_scores = rrf_fuse(ranked_lists, rrf_k=config.RRF_K)

    # 2. Apply deterministic boosts and penalties
    adjusted_scores: list[tuple[int, float, list[str]]] = []
    for idx, score in fused_scores.items():
        chunk = chunks[idx]
        doc = doc_map.get(chunk.doc_id)
        multiplier = 1.0
        why = ["bm25_rrf"]

        if doc:
            # Target entity boost
            if chunk.doc_id in target_doc_ids:
                multiplier *= config.BOOST_TARGET
                why.append(f"boost_target_entity (x{config.BOOST_TARGET})")
            elif chunk.doc_id in confusable_doc_ids and not is_ambiguous:
                multiplier *= config.PENALTY_CONFUSABLE
                why.append(f"penalty_confusable (x{config.PENALTY_CONFUSABLE})")

            # Validity date / status adjustments
            if doc.meta.get("status") == "repealed":
                if analysis.event_date:
                    date_mult = score_date_version(doc, analysis.event_date)
                    multiplier *= date_mult
                    why.append(f"event_date_version (x{date_mult})")
                else:
                    multiplier *= config.PENALTY_REPEALED
                    why.append(f"penalty_repealed (x{config.PENALTY_REPEALED})")

            # Explicit law match boost
            law_code = doc.meta.get("law_code", "").lower()
            art = doc.meta.get("article", "")
            if law_code and (
                law_code in analysis.standalone_question.lower()
                or f"ст. {art}" in analysis.standalone_question.lower()
            ):
                multiplier *= 1.5
                why.append("explicit_law_match (x1.5)")

        adjusted_scores.append((idx, score * multiplier, why))

    adjusted_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = adjusted_scores[:config.FUSED_TOP_K]

    # 3. Graph & neighbor expansion
    candidate_chunks: dict[str, tuple[Chunk, float, list[str], bool]] = {
        chunks[idx].chunk_id: (chunks[idx], score, why, True) for idx, score, why in top_candidates
    }

    top_docs_in_cand: dict[str, float] = {}
    for idx, score, _ in top_candidates[:10]:
        d_id = chunks[idx].doc_id
        if d_id not in top_docs_in_cand or score > top_docs_in_cand[d_id]:
            top_docs_in_cand[d_id] = score

    # (a) Add related documents via graph
    for doc_id, parent_score in top_docs_in_cand.items():
        for neighbor_id, rel in graph.get_neighbors(doc_id):
            n_chunks = doc_to_chunks.get(neighbor_id, [])
            rel_boost = parent_score * 0.75
            if rel in ("amended_by", "supplemented_by", "superseded_by", "supersedes"):
                for nc in n_chunks:
                    why_msg = f"graph_{rel}_from_{doc_map[doc_id].title[:20]}"
                    if nc.chunk_id not in candidate_chunks:
                        candidate_chunks[nc.chunk_id] = (nc, rel_boost, [why_msg], False)
                    else:
                        curr_c, curr_s, curr_why, curr_dir = candidate_chunks[nc.chunk_id]
                        candidate_chunks[nc.chunk_id] = (
                            curr_c,
                            max(curr_s, rel_boost),
                            curr_why + [why_msg],
                            curr_dir,
                        )

    # (b) Add neighboring chunks (seq +- 1) if chunk is short
    for idx, parent_score, _ in top_candidates[:10]:
        c = chunks[idx]
        doc_chunks = doc_to_chunks.get(c.doc_id, [])
        if len(c.text) < 500 or not c.text.rstrip().endswith("."):
            for offset in (-1, 1):
                target_seq = c.seq + offset
                for nc in doc_chunks:
                    if nc.seq == target_seq and nc.chunk_id not in candidate_chunks:
                        if len(candidate_chunks) < config.MAX_CANDIDATES:
                            candidate_chunks[nc.chunk_id] = (
                                nc,
                                parent_score * 0.5,
                                ["neighbor_seq_offset"],
                                False,
                            )

    # (c) Ensure target doc is present if identified
    if target_doc_ids:
        has_target = any(c.doc_id in target_doc_ids for c, _, _, _ in candidate_chunks.values())
        if not has_target:
            best_cand_score = max((s for _, s, _, _ in candidate_chunks.values()), default=0.01)
            for t_id in target_doc_ids:
                t_chunks = doc_to_chunks.get(t_id, [])
                for tc in t_chunks[:3]:
                    candidate_chunks[tc.chunk_id] = (
                        tc,
                        best_cand_score * 0.8,
                        ["target_entity_forced"],
                        True,
                    )

    cand_list = sorted(candidate_chunks.values(), key=lambda x: x[1], reverse=True)[:config.MAX_CANDIDATES]

    # 4. Reranking step
    rerank_used = False
    rerank_scores: list[tuple[Chunk, float, list[str], bool]] = []

    if client and config.RERANK_KEY and len(cand_list) > 1:
        query_text = analysis.standalone_question
        if config.RERANK_WITH_INSTRUCTION:
            query_text = (
                "Найди фрагмент внутреннего документа или закона, который отвечает на юридический вопрос. Вопрос: "
                + analysis.standalone_question
            )
        doc_texts = [c.full[:2000] for c, _, _, _ in cand_list]

        try:
            raw_ranks = client.rerank(query_text, doc_texts, top_n=len(doc_texts))
            score_map = {idx: s for idx, s in raw_ranks}
            score_kind = "rerank"
            rerank_used = True

            for pos, (c, _, prev_why, is_dir) in enumerate(cand_list):
                base_score = score_map.get(pos, 0.0)
                mod = 1.0
                why = list(prev_why) + [f"rerank_score={base_score:.3f}"]
                if c.doc_id in target_doc_ids:
                    mod *= 1.25
                elif c.doc_id in confusable_doc_ids and not is_ambiguous:
                    mod *= 0.5
                rerank_scores.append((c, base_score * mod, why, is_dir))

            rerank_scores.sort(key=lambda x: x[1], reverse=True)
            trace.append(f"Reranking completed for {len(cand_list)} items")
        except Exception as exc:
            print(f"[RERANK] Warning: reranker failed ({exc}); falling back to fused scores", file=sys.stderr)
            trace.append(f"Reranker failed: {exc}")
            rerank_scores = cand_list
    else:
        rerank_scores = cand_list

    # Ensure related graph documents from top-5 are preserved in context
    top_context = rerank_scores[:top_k]
    top_doc_ids = {c.doc_id for c, _, _, _ in top_context[:5]}

    for c, score, why, is_dir in rerank_scores[top_k:]:
        for top_id in top_doc_ids:
            if any(
                tid == c.doc_id and rel in ("amended_by", "supplemented_by", "superseded_by", "supersedes")
                for tid, rel in graph.get_neighbors(top_id)
            ):
                top_context.append((c, score, why, is_dir))
                break

    # 5. Build structured Evidence and DocumentMatch objects
    evidence_list: list[Evidence] = []
    doc_matches_map: dict[str, DocumentMatch] = {}
    doc_evidence_scores: dict[str, list[float]] = {}

    for c, s, why, is_dir in top_context:
        doc = doc_map[c.doc_id]
        rel_val = normalize_relevance(s, score_kind)
        start_idx = doc.text.find(c.text)
        if start_idx == -1:
            start_idx = 0
        end_idx = start_idx + len(c.text)

        source_kind = "graph" if not is_dir else ("dense" if dense_used else "bm25")

        ev = Evidence(
            doc_id=c.doc_id,
            title=doc.title,
            section=c.section,
            clause_ids=list(c.clause_ids),
            text=c.text,
            start=start_idx,
            end=end_idx,
            score=float(s),
            score_kind=score_kind,
            relevance=rel_val,
            source=source_kind,
            why=why,
        )
        evidence_list.append(ev)
        doc_evidence_scores.setdefault(c.doc_id, []).append(rel_val)

        if c.doc_id not in doc_matches_map:
            app_ed, ed_note = resolve_applicable_edition(
                doc, event_date_str=analysis.event_date
            )
            doc_num = doc.meta.get("doc_numbers", [None])[0] if doc.meta.get("doc_numbers") else None
            party_names = [p["name"] for p in doc.meta.get("parties", [])]

            doc_matches_map[c.doc_id] = DocumentMatch(
                doc_id=doc.doc_id,
                title=doc.title,
                doc_number=doc_num,
                doc_type=doc.doc_type,
                parties=party_names,
                status=doc.meta.get("status", "active"),
                effective_from=doc.meta.get("effective_from"),
                effective_to=doc.meta.get("effective_to"),
                applicable_edition=app_ed,
                edition_note=ed_note,
                score=float(s),
                score_kind=score_kind,
                relevance=rel_val,
                matched_clauses=list(c.clause_ids),
                is_direct=is_dir,
                why=why,
            )
        else:
            dm = doc_matches_map[c.doc_id]
            for cl in c.clause_ids:
                if cl not in dm.matched_clauses:
                    dm.matched_clauses.append(cl)
            if s > dm.score:
                dm.score = float(s)
                dm.relevance = rel_val

    # Sort documents by relevance descending
    documents_list = sorted(doc_matches_map.values(), key=lambda d: d.relevance, reverse=True)

    # 6. Check for broken internal refs in matched documents (e.g. 152-ФЗ ст. 6, 10, 11)
    for dm in documents_list:
        doc = doc_map[dm.doc_id]
        broken = doc.meta.get("broken_refs", [])
        for b_ref in broken:
            if b_ref not in unresolved.law_articles:
                unresolved.law_articles.append(b_ref)
                ref_errors.append(
                    ReferenceError(
                        kind="broken_internal_ref",
                        where="kb_document",
                        raw=b_ref,
                        doc_id=doc.doc_id,
                        clause=None,
                        candidates=[],
                        severity="warning",
                    )
                )

    # 7. Check edition_missing error (if question specified an event date not covered by KB)
    if analysis.event_date:
        for dm in documents_list:
            if not dm.applicable_edition and "Не действовала на дату события" in dm.edition_note:
                any_applicable = any(d.applicable_edition for d in documents_list if d.doc_type == dm.doc_type)
                if not any_applicable:
                    ref_errors.append(
                        ReferenceError(
                            kind="edition_missing",
                            where="question",
                            raw=analysis.event_date,
                            doc_id=dm.doc_id,
                            clause=None,
                            candidates=[{"doc_id": dm.doc_id, "title": dm.title}],
                            severity="warning",
                        )
                    )

    # 8. Compute Signals
    relevances = [ev.relevance for ev in evidence_list]
    max_rel = max(relevances) if relevances else 0.0
    mean_rel = (sum(relevances) / len(relevances)) if relevances else 0.0
    top3_rel = relevances[:3]
    mean_top3_rel = (sum(top3_rel) / len(top3_rel)) if top3_rel else 0.0

    margin_top = 0.0
    if len(documents_list) >= 2:
        margin_top = max(0.0, documents_list[0].relevance - documents_list[1].relevance)
    elif len(documents_list) == 1:
        margin_top = documents_list[0].relevance

    distinct_doc_count = len(doc_matches_map)

    entity_matches = []
    for num in analysis.entities.get("doc_numbers", []):
        if num.lower() in resolver.num_to_doc:
            entity_matches.append(num)
    for org in analysis.entities.get("organizations", []):
        if any(norm_str(org) in norm_str(d.title) or any(norm_str(org) in norm_str(p) for p in d.parties) for d in documents_list):
            entity_matches.append(org)

    low_confidence = (best_raw_bm25 < 7.0 and max_rel < 0.25)
    if rerank_used and max_rel < config.RERANK_EMPTY_THRESHOLD:
        low_confidence = True

    # If user explicitly asked about an entity/contract number not found in KB
    if (unresolved.party_names or unresolved.contract_numbers) and not entity_matches:
        low_confidence = True

    topic_match = len(documents_list) > 0
    applicable_ed_ids = [d.doc_id for d in documents_list if d.applicable_edition]

    signals = Signals(
        low_confidence=low_confidence,
        max_relevance=round(max_rel, 4),
        mean_relevance=round(mean_rel, 4),
        mean_top3_relevance=round(mean_top3_rel, 4),
        margin_top1_top2=round(margin_top, 4),
        distinct_docs=distinct_doc_count,
        entity_match=entity_matches,
        topic_match=topic_match,
        score_kind=score_kind,
        dense_used=dense_used,
        rerank_used=rerank_used,
        event_date=analysis.event_date,
        applicable_editions=applicable_ed_ids,
    )

    result = RetrievalResult(
        documents=documents_list,
        evidence=evidence_list,
        signals=signals,
        unresolved=unresolved,
        ambiguity=ambiguities,
        reference_errors=ref_errors,
        trace=trace,
    )

    return result
