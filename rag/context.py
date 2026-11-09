from __future__ import annotations

import re
from typing import Sequence

from rag import config
from rag.chunker import Chunk
from rag.metadata import DocGraph, Document
from rag.query import QueryAnalysis


def build_context_cards(
    ranked_chunks: list[tuple[Chunk, float]],
    docs: Sequence[Document],
    graph: DocGraph,
    analysis: QueryAnalysis,
    is_ambiguous: bool = False,
) -> str:
    """Build structured document cards and fragments for the LLM context."""
    if not ranked_chunks:
        return "В базе знаний не найдено фрагментов по данному вопросу."

    doc_map = {d.doc_id: d for d in docs}

    # Group chunks by document while preserving document priority by max chunk score
    doc_best_score: dict[str, float] = {}
    doc_chunks: dict[str, list[Chunk]] = {}

    for c, score in ranked_chunks:
        if c.doc_id not in doc_best_score or score > doc_best_score[c.doc_id]:
            doc_best_score[c.doc_id] = score
        doc_chunks.setdefault(c.doc_id, []).append(c)

    sorted_doc_ids = sorted(doc_best_score.keys(), key=lambda d_id: doc_best_score[d_id], reverse=True)

    # Assign stable doc labels [D1], [D2]...
    doc_labels = {d_id: f"D{idx + 1}" for idx, d_id in enumerate(sorted_doc_ids)}

    context_lines: list[str] = []

    # 1. Ambiguity or Event Date banner
    if is_ambiguous:
        context_lines.append(
            "Внимание: в вопросе контрагент указан неоднозначно; в базе присутствуют разные компании: "
            "ООО «Альфа-Трейд» (договор поставки № П-17/2024, мебель) и "
            "ООО «Альфа-Трейдинг» (договор поставки № ПТ-05/2023, компьютерная техника).\n"
        )

    if analysis.event_date:
        applicable_label = ""
        for d_id in sorted_doc_ids:
            doc = doc_map.get(d_id)
            if doc and doc.meta.get("category") != "law":
                f_d = doc.meta.get("effective_from") or "0000-00-00"
                t_d = doc.meta.get("effective_to") or "9999-99-99"
                if f_d[:7] <= analysis.event_date[:7] <= t_d[:7]:
                    applicable_label = f"[{doc_labels[d_id]}]"
                    break
        date_banner = f"Дата события из вопроса: {analysis.event_date}."
        if applicable_label:
            date_banner += f" Применимая редакция документа: {applicable_label} (действовала на эту дату)."
        context_lines.append(date_banner + "\n")

    # 2. Document cards
    current_chars = sum(len(line) for line in context_lines)

    for d_id in sorted_doc_ids:
        doc = doc_map.get(d_id)
        if not doc:
            continue

        label = doc_labels[d_id]
        meta = doc.meta
        header_lines = [f"=== [{label}] {doc.title}"]

        # Status and validity period
        status_text = "Действует"
        if meta.get("status") == "repealed":
            status_text = "УТРАТИЛ СИЛУ"
            if meta.get("effective_to"):
                status_text += f" с {meta.get('effective_to')}"
        elif meta.get("effective_from"):
            status_text += f" с {meta.get('effective_from')}"

        validity_interval = ""
        if meta.get("effective_from") or meta.get("effective_to"):
            validity_interval = f" (период действия: {meta.get('effective_from') or '...'} — {meta.get('effective_to') or 'настоящее время'})"

        header_lines.append(f"Статус: {status_text}{validity_interval}.")

        # Parties for contracts
        parties = meta.get("parties", [])
        if parties:
            p_strs = [f"{p['name']} ({p.get('inn') or p.get('role') or p.get('form')})" for p in parties]
            header_lines.append("Стороны: " + " — ".join(p_strs))

        # Graph relations (amended_by, supersedes, supplements)
        rel_notes: list[str] = []
        for n_id, rel in graph.get_neighbors(d_id):
            if n_id in doc_labels:
                n_label = f"[{doc_labels[n_id]}]"
                if rel == "amended_by":
                    rel_notes.append(f"изменен документом {n_label}")
                elif rel == "amends":
                    rel_notes.append(f"является доп. соглашением к {n_label}")
                elif rel == "supersedes":
                    rel_notes.append(f"заменяет собой прежнюю редакцию {n_label}")
                elif rel == "superseded_by":
                    rel_notes.append(f"заменен новой редакцией {n_label}")
                elif rel == "supplemented_by":
                    rel_notes.append(f"дополняется письмом {n_label}")
                elif rel == "supplements":
                    rel_notes.append(f"применяется в дополнение к {n_label}")
        if rel_notes:
            header_lines.append("Связи: " + "; ".join(rel_notes))

        # Amendments notes for law articles
        if meta.get("amend_notes"):
            header_lines.append("Служебные пометки об изменениях: " + "; ".join(meta["amend_notes"][:2]))

        card_str = "\n".join(header_lines) + "\n"

        # Chunks for this document sorted by sequence
        c_list = sorted(doc_chunks[d_id], key=lambda x: x.seq)
        fragments_strs: list[str] = []

        for c in c_list:
            sec_part = f" ({c.section})" if c.section else ""
            cl_part = f" [пп. {', '.join(c.clause_ids)}]" if c.clause_ids else ""
            frag = f"--- фрагмент{sec_part}{cl_part}\n{c.text}\n"
            fragments_strs.append(frag)

        doc_total_text = card_str + "\n".join(fragments_strs) + "\n"
        if current_chars + len(doc_total_text) > config.CONTEXT_MAX_CHARS and context_lines:
            # Reached max context limit
            break

        context_lines.append(doc_total_text)
        current_chars += len(doc_total_text)

    return "\n".join(context_lines).strip()
