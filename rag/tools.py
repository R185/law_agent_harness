from __future__ import annotations

import re
from typing import Sequence

from rag import config
from rag.chunker import chunk_document
from rag.metadata import Document, resolve_applicable_edition
from rag.schema import (
    CitationCheck,
    ContextResult,
    DocumentMatch,
    Evidence,
    Resolution,
    RetrievalResult,
)
from rag.text import extract_doc_numbers, norm_str


def build_context(
    result: RetrievalResult,
    max_chars: int = 12000,
    token_budget: int | None = None,
    max_tokens: int | None = None,
) -> ContextResult:
    """Build formatted context string, evidence list, sources and warnings for prompt and UI."""
    if max_tokens is not None:
        max_chars = min(max_chars, max_tokens * 4)
    elif token_budget is not None:
        max_chars = min(max_chars, token_budget * 4)
    if not result.evidence and not result.documents:
        return ContextResult(
            text="В базе знаний не найдено фрагментов по данному вопросу.",
            sources=[],
            evidence=[],
            warnings=["no_matches_found"],
        )

    # Group evidence by doc_id
    doc_evidences: dict[str, list[Evidence]] = {}
    for ev in result.evidence:
        doc_evidences.setdefault(ev.doc_id, []).append(ev)

    lines: list[str] = []
    sources: list[str] = []
    included_ev: list[Evidence] = []
    warnings: list[str] = []

    # 1. Ambiguity warning banner
    if result.ambiguity:
        for amb in result.ambiguity:
            cand_strs = [f"{c.title} ({c.label})" for c in amb.candidates]
            warn = (
                f"Внимание: сущность '{amb.raw}' указана неоднозначно; "
                f"возможные совпадения в базе: {'; '.join(cand_strs)}."
            )
            lines.append(warn + "\n")
            warnings.append(warn)

    # 2. Reference errors warnings
    for err in result.reference_errors:
        if err.kind == "broken_internal_ref":
            w = f"Внимание: документ ссылается на норму, отсутствующую в предоставленной базе: '{err.raw}'."
            if w not in warnings:
                warnings.append(w)
                lines.append(w + "\n")
        elif err.kind == "edition_missing":
            w = f"Внимание: редакция документа на дату '{err.raw}' в базе отсутствует."
            if w not in warnings:
                warnings.append(w)
                lines.append(w + "\n")

    # 3. Document cards
    current_chars = sum(len(line) for line in lines)
    for dm in result.documents:
        doc_header_lines = [f"=== [{dm.doc_id[:8]}] {dm.title}"]
        doc_header_lines.append(f"Статус: {dm.status}. {dm.edition_note}.")
        if dm.parties:
            doc_header_lines.append(f"Стороны: {' — '.join(dm.parties)}")
        if dm.why:
            doc_header_lines.append(f"Обоснование релевантности: {'; '.join(dm.why[:2])}")

        card_str = "\n".join(doc_header_lines) + "\n"
        ev_list = doc_evidences.get(dm.doc_id, [])
        frag_lines = []

        for ev in ev_list:
            sec_part = f" ({ev.section})" if ev.section else ""
            cl_part = f" [пп. {', '.join(ev.clause_ids)}]" if ev.clause_ids else ""
            frag = f"--- фрагмент{sec_part}{cl_part}\n{ev.text}\n"
            frag_lines.append(frag)

        doc_block = card_str + "\n".join(frag_lines) + "\n"
        if current_chars + len(doc_block) > max_chars and lines:
            warnings.append(f"Контекст обрезан по лимиту max_chars={max_chars}")
            break

        lines.append(doc_block)
        current_chars += len(doc_block)
        sources.append(f"{dm.title} ({dm.doc_number or dm.status})")
        included_ev.extend(ev_list)

    final_text = "\n".join(lines).strip()
    return ContextResult(
        text=final_text,
        sources=sources,
        evidence=included_ev,
        warnings=warnings,
    )


def resolve_reference(raw: str, docs: Sequence[Document] | None = None) -> dict:
    """Resolve a raw reference string like 'п. 3.2 договора № П-17/2024' or 'ст. 70 ТК РФ'."""
    from rag.loader import load_raw_documents
    from rag.metadata import build_document_metadata
    from rag.sanitize import sanitize_text

    if docs is None:
        raw_docs = load_raw_documents()
        docs = [build_document_metadata(rd.path, *sanitize_text(rd.text, rd.path)) for rd in raw_docs]

    clean_raw = raw.strip()
    norm_raw = norm_str(clean_raw)

    # 1. Check document numbers
    doc_nums = extract_doc_numbers(clean_raw)
    target_doc = None
    target_clause = None

    # Check clause mention (e.g. п. 3.2, статья 70)
    m_cl = re.search(r"(?:п\.|пункт|ст\.|стать[яеи])\s*([\d.]+)", clean_raw, re.I)
    if m_cl:
        target_clause = m_cl.group(1).rstrip(".")

    for d in docs:
        d_nums = [n.lower() for n in d.meta.get("doc_numbers", [])]
        if any(dn in d_nums for dn in doc_nums):
            target_doc = d
            break
        if d.category == "law":
            art = d.meta.get("article", "")
            code = d.meta.get("law_code", "").lower()
            if art and target_clause and art == target_clause:
                if not code or code in norm_raw:
                    target_doc = d
                    break

    if target_doc:
        doc_nums = target_doc.meta.get("doc_numbers") or []
        doc_num = doc_nums[0] if doc_nums else None
        return Resolution(
            status="resolved",
            doc_id=target_doc.doc_id,
            title=target_doc.title,
            doc_number=doc_num,
            clause=target_clause,
            candidates=[],
        )

    return Resolution(
        status="unresolved",
        doc_id=None,
        title=None,
        doc_number=None,
        clause=target_clause,
        candidates=[],
    )


def verify_citations(
    answer_text: str, context: str | None = None, docs: Sequence[Document] | None = None
) -> CitationCheck:
    """Verify legal citations in the answer text against the KB documents and context."""
    from rag.loader import load_raw_documents
    from rag.metadata import build_document_metadata
    from rag.sanitize import sanitize_text

    if docs is None:
        raw_docs = load_raw_documents()
        docs = [build_document_metadata(rd.path, *sanitize_text(rd.text, rd.path)) for rd in raw_docs]

    known_doc_nums = set()
    known_laws = set()
    doc_clauses: dict[str, set[str]] = {}

    for d in docs:
        for num in d.meta.get("doc_numbers", []):
            known_doc_nums.add(num.lower())
            doc_clauses.setdefault(num.lower(), set())

        if d.category == "law":
            art = d.meta.get("article")
            code = d.meta.get("law_code")
            if art and code:
                known_laws.add(f"{code.lower()} ст. {art}")
                known_laws.add(f"ст. {art}")
                doc_clauses.setdefault(f"ст. {art}", set())

        # Collect clause IDs from chunks
        for c in chunk_document(d):
            for cl in c.clause_ids:
                for num in d.meta.get("doc_numbers", []):
                    doc_clauses.setdefault(num.lower(), set()).add(cl)
                if d.category == "law" and d.meta.get("article"):
                    doc_clauses.setdefault(f"ст. {d.meta.get('article')}", set()).add(cl)

    # Find cited contract numbers (e.g. № А-08/2024, П-17/2024)
    cited_numbers = extract_doc_numbers(answer_text)
    # Find cited law articles (e.g. ст. 70, статья 59, 152-ФЗ)
    cited_articles = re.findall(r"(?:ст\.|стать[яеи])\s*(\d+(?:\.\d+)?)", answer_text, re.I)
    # Find cited clauses (e.g. п. 3.1, пп. 2.4)
    cited_clauses = re.findall(r"(?:п\.|пункт|пп\.)\s*(\d+(?:\.\d+)+)", answer_text, re.I)

    all_citations = []
    unknown_docs = []
    unknown_clauses = []
    unsupported = []

    # Check contract numbers
    for num in cited_numbers:
        all_citations.append(f"договор {num}")
        if num.lower() not in known_doc_nums:
            unknown_docs.append(num)

    # Check law articles
    for art in cited_articles:
        ref_str = f"ст. {art}"
        all_citations.append(ref_str)
        if not any(art == d.meta.get("article") for d in docs if d.category == "law"):
            unknown_docs.append(ref_str)

    # Check clauses
    for cl in cited_clauses:
        all_citations.append(f"п. {cl}")
        clause_found_anywhere = False
        for cl_set in doc_clauses.values():
            if cl in cl_set or any(c_id.startswith(cl) or cl.startswith(c_id) for c_id in cl_set):
                clause_found_anywhere = True
                break
        if not clause_found_anywhere and context and f"{cl}" not in context:
            unknown_clauses.append(f"п. {cl}")

    # Check context support
    if context:
        for cit in all_citations:
            clean_cit = cit.replace("договор ", "").replace("ст. ", "")
            if clean_cit not in context:
                unsupported.append(cit)

    valid = (len(unknown_docs) == 0 and len(unknown_clauses) == 0)

    return CitationCheck(
        valid=valid,
        citations=all_citations,
        unknown_docs=unknown_docs,
        unknown_clauses=unknown_clauses,
        unsupported=unsupported,
    )


def get_document(doc_id: str, docs: Sequence[Document] | None = None) -> DocumentMatch | None:
    """Retrieve DocumentMatch by doc_id."""
    from rag.loader import load_raw_documents
    from rag.metadata import build_document_metadata
    from rag.sanitize import sanitize_text

    if docs is None:
        raw_docs = load_raw_documents()
        docs = [build_document_metadata(rd.path, *sanitize_text(rd.text, rd.path)) for rd in raw_docs]

    for d in docs:
        if d.doc_id == doc_id or d.doc_id.startswith(doc_id):
            app_ed, ed_note = resolve_applicable_edition(d)
            num = d.meta.get("doc_numbers", [None])[0] if d.meta.get("doc_numbers") else None
            parties = [p["name"] for p in d.meta.get("parties", [])]
            return DocumentMatch(
                doc_id=d.doc_id,
                title=d.title,
                doc_number=num,
                doc_type=d.doc_type,
                parties=parties,
                status=d.meta.get("status", "active"),
                effective_from=d.meta.get("effective_from"),
                effective_to=d.meta.get("effective_to"),
                applicable_edition=app_ed,
                edition_note=ed_note,
                score=1.0,
                score_kind="lookup",
                relevance=1.0,
                matched_clauses=[],
                is_direct=True,
                why=["direct_lookup"],
            )
    return None


def get_clause(doc_id: str, clause: str, docs: Sequence[Document] | None = None) -> Evidence | None:
    """Retrieve specific clause Evidence by doc_id and clause number."""
    from rag.loader import load_raw_documents
    from rag.metadata import build_document_metadata
    from rag.sanitize import sanitize_text

    if docs is None:
        raw_docs = load_raw_documents()
        docs = [build_document_metadata(rd.path, *sanitize_text(rd.text, rd.path)) for rd in raw_docs]

    clean_cl = clause.strip().rstrip(".")

    for d in docs:
        if d.doc_id == doc_id or d.doc_id.startswith(doc_id):
            for c in chunk_document(d):
                if clean_cl in c.clause_ids or clean_cl == c.section:
                    start_idx = d.text.find(c.text)
                    if start_idx == -1:
                        start_idx = 0
                    return Evidence(
                        doc_id=d.doc_id,
                        title=d.title,
                        section=c.section,
                        clause_ids=list(c.clause_ids),
                        text=c.text,
                        start=start_idx,
                        end=start_idx + len(c.text),
                        score=1.0,
                        score_kind="lookup",
                        relevance=1.0,
                        source="explicit",
                        why=["clause_lookup"],
                    )
    return None
