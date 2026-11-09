from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag import config
from rag.metadata import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    header: str
    text: str
    section: str = ""
    clause_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def full(self) -> str:
        if self.header:
            return f"{self.header}\n{self.text}"
        return self.text


SECTION_HEADER_RE = re.compile(r"^(\d{1,2})\.\s+([А-ЯЁA-Z0-9].*)$")
CLAUSE_LINE_RE = re.compile(r"^(\d{1,2}\.\d{1,2}(?:\.\d+)?)\.?\s+")
LAW_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\.\s+")
LAW_SUB_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)?)\)\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")


def build_chunk_header(doc: Document, section: str = "") -> str:
    """Build compact contextual header for a chunk (<= 250 chars)."""
    meta = doc.meta
    parts = []

    if doc.category == "law":
        code = meta.get("law_code") or "Закон"
        art = meta.get("article") or ""
        art_title = meta.get("article_title") or doc.title
        prefix = f"{code}, ст. {art} «{art_title}»" if art else f"{code} «{art_title}»"
        amends = meta.get("amend_notes", [])
        status_note = "действует"
        if amends:
            m = re.search(r"с\s+(\d{1,2}\s+[а-я]+\s+\d{4}\s*г\.|\d{2}\.\d{2}\.\d{4})", amends[0], re.I)
            if m:
                status_note = f"ред. с изм. от {m.group(1).strip()}"
        parts.append(f"[{prefix} | {status_note}]")
        if section:
            parts.append(f"Часть/пункт: {section}")
    else:
        # Company document
        title_core = doc.title
        doc_num = meta.get("doc_numbers", [])
        num_str = f"№ {doc_num[0]}" if doc_num else ""
        date_str = f"от {meta.get('doc_date')}" if meta.get("doc_date") else ""

        parties = meta.get("parties", [])
        parties_str = ""
        if len(parties) >= 2:
            parties_str = f" | {parties[0]['name']} — {parties[1]['name']}"
        elif len(parties) == 1:
            parties_str = f" | {parties[0]['name']}"

        status_str = "действует"
        if meta.get("status") == "repealed":
            to_d = meta.get("effective_to")
            status_str = f"УТРАТИЛ СИЛУ" + (f" (действовал до {to_d})" if to_d else "")
        elif meta.get("effective_from"):
            status_str = f"действует с {meta.get('effective_from')}"

        if doc.doc_type == "amendment":
            refs = meta.get("refs", [])
            target = f" к договору {refs[0].get('target_num', '')}" if refs else ""
            header_core = f"[{title_core}{target}{parties_str} | {status_str}]"
        else:
            header_core = f"[{title_core} {num_str} {date_str}{parties_str} | {status_str}]"
            # Clean double spaces
            header_core = re.sub(r"\s+", " ", header_core).replace(" | ]", "]").replace(" ]", "]")

        parts.append(header_core)
        if section and section != "Преамбула":
            parts.append(f"Раздел: {section}")
        elif section == "Преамбула":
            parts.append("Раздел: Преамбула")

    header = " ".join(parts).strip()
    return header[:250]


def extract_clauses_from_text(text: str) -> list[str]:
    """Find all clause numbers (e.g. 3.1, 3.2) mentioned at line starts."""
    clauses: list[str] = []
    for line in text.split("\n"):
        m = CLAUSE_LINE_RE.match(line.strip())
        if m:
            c = m.group(1).rstrip(".")
            if c not in clauses:
                clauses.append(c)
    return clauses


def chunk_company_document(doc: Document) -> list[Chunk]:
    """Chunk contract, regulation, memo, letter, or order."""
    text = doc.text
    lines = text.split("\n")

    # If small document without sections (memo, letter, short order)
    if len(text) <= config.SMALL_DOC_CHARS and not any(SECTION_HEADER_RE.match(l.strip()) for l in lines):
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}:0000",
                doc_id=doc.doc_id,
                seq=0,
                header=build_chunk_header(doc),
                text=text.strip(),
                section="",
                clause_ids=extract_clauses_from_text(text),
                notes=[],
            )
        ]

    # Parse sections
    # Find preamble and numbered sections
    sections: list[tuple[str, list[str]]] = []
    current_sec_name = "Преамбула"
    current_sec_lines: list[str] = []

    for line in lines:
        s_line = line.strip()
        m_sec = SECTION_HEADER_RE.match(s_line)
        if m_sec and len(s_line) < 90:
            if current_sec_lines:
                sections.append((current_sec_name, current_sec_lines))
                current_sec_lines = []
            current_sec_name = s_line
        else:
            current_sec_lines.append(line)

    if current_sec_lines:
        sections.append((current_sec_name, current_sec_lines))

    chunks: list[Chunk] = []
    seq = 0

    for sec_name, sec_lines in sections:
        sec_text = "\n".join(sec_lines).strip()
        if not sec_text:
            continue

        # If section fits in CHUNK_MAX, make 1 chunk
        if len(sec_text) <= config.CHUNK_MAX:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{seq:04d}",
                    doc_id=doc.doc_id,
                    seq=seq,
                    header=build_chunk_header(doc, sec_name),
                    text=sec_text,
                    section=sec_name,
                    clause_ids=extract_clauses_from_text(sec_text),
                    notes=[],
                )
            )
            seq += 1
            continue

        # Section is large: split by clauses or paragraphs
        # Group clauses into blocks up to CHUNK_TARGET
        clause_blocks: list[tuple[str, list[str]]] = []  # (clause_id, lines)
        cur_clause = ""
        cur_c_lines: list[str] = []

        for line in sec_lines:
            s_line = line.strip()
            m_cl = CLAUSE_LINE_RE.match(s_line)
            if m_cl:
                if cur_c_lines:
                    clause_blocks.append((cur_clause, cur_c_lines))
                    cur_c_lines = []
                cur_clause = m_cl.group(1).rstrip(".")
                cur_c_lines.append(line)
            else:
                cur_c_lines.append(line)
        if cur_c_lines:
            clause_blocks.append((cur_clause, cur_c_lines))

        # Pack clause blocks up to CHUNK_TARGET
        packed_text = ""
        packed_clauses: list[str] = []

        for cl_id, cl_lines in clause_blocks:
            block_text = "\n".join(cl_lines).strip()
            if not block_text:
                continue

            if packed_text and (len(packed_text) + len(block_text) > config.CHUNK_TARGET):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{seq:04d}",
                        doc_id=doc.doc_id,
                        seq=seq,
                        header=build_chunk_header(doc, sec_name),
                        text=packed_text.strip(),
                        section=sec_name,
                        clause_ids=list(packed_clauses),
                        notes=[],
                    )
                )
                seq += 1
                packed_text = block_text
                packed_clauses = [cl_id] if cl_id else []
            else:
                packed_text = f"{packed_text}\n\n{block_text}".strip() if packed_text else block_text
                if cl_id and cl_id not in packed_clauses:
                    packed_clauses.append(cl_id)

        if packed_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{seq:04d}",
                    doc_id=doc.doc_id,
                    seq=seq,
                    header=build_chunk_header(doc, sec_name),
                    text=packed_text.strip(),
                    section=sec_name,
                    clause_ids=list(packed_clauses),
                    notes=[],
                )
            )
            seq += 1

    return chunks


def chunk_law_document(doc: Document) -> list[Chunk]:
    """Chunk law article with structural splitting, notes attachment and header preservation."""
    raw_text = doc.text

    # Strip header passport (Источник:, URL:, Дата выгрузки:)
    clean_lines = []
    in_passport = True
    for line in raw_text.split("\n"):
        s_line = line.strip()
        if in_passport:
            if any(s_line.startswith(prefix) for prefix in ("Источник:", "URL:", "Дата выгрузки:")):
                continue
            if not s_line:
                continue
            # Passport ended
            in_passport = False
        clean_lines.append(line)

    clean_text = "\n".join(clean_lines).strip()

    # If small article <= SMALL_ARTICLE_CHARS, one chunk
    if len(clean_text) <= config.SMALL_ARTICLE_CHARS:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}:0000",
                doc_id=doc.doc_id,
                seq=0,
                header=build_chunk_header(doc),
                text=clean_text,
                section="",
                clause_ids=extract_clauses_from_text(clean_text),
                notes=list(doc.meta.get("amend_notes", [])),
            )
        ]

    # Split into paragraphs by empty lines
    raw_paras = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    processed_paras: list[tuple[str, str, list[str]]] = []  # (section_id, para_text, notes)
    pending_notes: list[str] = []
    current_sec = ""

    for p in raw_paras:
        # Check if note
        p_lower = p.lower()
        is_note = (
            len(p) < 400
            and any(w in p_lower for w in ("изменен", "дополнен", "утратил силу", "о конституционно-правовом", "см.", "положения"))
            and ("федеральн" in p_lower or "закон" in p_lower or "стать" in p_lower or "пункт" in p_lower)
        )
        if is_note:
            pending_notes.append(p)
            continue

        # Check section number
        m_sec = LAW_SECTION_RE.match(p)
        if m_sec:
            current_sec = m_sec.group(1)
        else:
            m_sub = LAW_SUB_CLAUSE_RE.match(p)
            if m_sub:
                current_sec = f"пп. {m_sub.group(1)}"

        # Attach pending notes to this paragraph
        para_notes = list(pending_notes)
        pending_notes = []

        # If single paragraph is giant (> CHUNK_MAX chars), split by sentences / semicolons
        if len(p) > config.CHUNK_MAX:
            sentences = SENTENCE_SPLIT_RE.split(p)
            cur_s_block = ""
            for s in sentences:
                if cur_s_block and (len(cur_s_block) + len(s) > config.LAW_CHUNK_TARGET):
                    processed_paras.append((current_sec, cur_s_block.strip(), para_notes))
                    para_notes = []
                    cur_s_block = s
                else:
                    cur_s_block = f"{cur_s_block} {s}".strip() if cur_s_block else s
            if cur_s_block:
                processed_paras.append((current_sec, cur_s_block.strip(), para_notes))
        else:
            processed_paras.append((current_sec, p, para_notes))

    # Group paragraphs into chunks up to LAW_CHUNK_TARGET
    chunks: list[Chunk] = []
    seq = 0
    cur_text = ""
    cur_sec = ""
    cur_notes: list[str] = []

    for sec_id, p_text, notes in processed_paras:
        # New chunk boundary if section changes and size is already reasonable (>= 400)
        # OR if size exceeds LAW_CHUNK_TARGET
        is_sec_boundary = (sec_id and sec_id != cur_sec and len(cur_text) >= 400)
        is_overflow = (len(cur_text) + len(p_text) > config.LAW_CHUNK_TARGET)

        if cur_text and (is_sec_boundary or is_overflow):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{seq:04d}",
                    doc_id=doc.doc_id,
                    seq=seq,
                    header=build_chunk_header(doc, cur_sec),
                    text=cur_text.strip(),
                    section=cur_sec,
                    clause_ids=extract_clauses_from_text(cur_text),
                    notes=list(cur_notes),
                )
            )
            seq += 1
            cur_text = p_text
            cur_sec = sec_id
            cur_notes = list(notes)
        else:
            if not cur_sec:
                cur_sec = sec_id
            cur_text = f"{cur_text}\n\n{p_text}".strip() if cur_text else p_text
            cur_notes.extend(notes)

    if cur_text:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{seq:04d}",
                doc_id=doc.doc_id,
                seq=seq,
                header=build_chunk_header(doc, cur_sec),
                text=cur_text.strip(),
                section=cur_sec,
                clause_ids=extract_clauses_from_text(cur_text),
                notes=list(cur_notes),
            )
        )

    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    """Route document to appropriate chunker."""
    if doc.category == "law":
        return chunk_law_document(doc)
    return chunk_company_document(doc)
