from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from rag import config
from rag.text import canonical_doc_num, extract_doc_numbers, norm_str


@dataclass
class Document:
    doc_id: str
    path: str
    title: str
    category: str  # "law" | "company"
    doc_type: str  # "contract" | "amendment" | "regulation" | "letter" | "memo" | "order" | "law_article" | "other"
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    removed_lines: list[str] = field(default_factory=list)


MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}

DATE_TEXT_RE = re.compile(
    r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
    re.I,
)
DATE_DOT_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")

PARTY_RE = re.compile(
    r"\b(ООО|АО|ПАО|ЗАО|ИП)\s+[«\"]([^»\"]+)[»\"](?:\s*\((?:ИНН\s*)?(\d{10,12})\))?(?:[,\s]+именуемое\s+в\s+дальнейшем\s+[«\"]([^»\"]+)[»\"])?",
    re.I,
)


def parse_date_to_iso(text: str) -> str | None:
    """Parse textual Russian date or dot date into YYYY-MM-DD."""
    m_dot = DATE_DOT_RE.search(text)
    if m_dot:
        day, month, year = int(m_dot.group(1)), int(m_dot.group(2)), int(m_dot.group(3))
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    m_text = DATE_TEXT_RE.search(text)
    if m_text:
        day = int(m_text.group(1))
        month_name = m_text.group(2).lower()
        year = int(m_text.group(3))
        month = MONTHS_RU.get(month_name, 1)
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass
    return None


def extract_parties(text: str) -> list[dict[str, str]]:
    """Extract contracting parties (company name, legal form, INN, contract role)."""
    parties: list[dict[str, str]] = []
    seen_names = set()

    for match in PARTY_RE.finditer(text[:1500]):
        form, name, inn, role = match.group(1), match.group(2), match.group(3) or "", match.group(4) or ""
        clean_name = name.strip()
        if clean_name.lower() in seen_names:
            continue
        seen_names.add(clean_name.lower())
        parties.append({
            "name": clean_name,
            "form": form.upper(),
            "inn": inn,
            "role": role.strip(),
        })
    return parties


def extract_subject(text: str) -> str:
    """Extract item 1.1 subject of agreement if present."""
    m = re.search(r"1\.1\.\s+([^\n]+(?:\n[^\n]+)?)", text)
    if m:
        s = " ".join(m.group(1).split())
        return s[:250]
    return ""


def parse_law_meta(path: str, text: str) -> dict[str, Any]:
    """Parse law article specifics from header and text."""
    meta: dict[str, Any] = {
        "law_code": "",
        "article": "",
        "article_title": "",
        "amend_notes": [],
        "export_date": "",
    }

    # Extract export date
    m_exp = re.search(r"Дата выгрузки:\s*(\d{2}\.\d{2}\.\d{4})", text)
    if m_exp:
        meta["export_date"] = parse_date_to_iso(m_exp.group(1)) or ""

    # Law code from filename or header
    norm_path = path.lower()
    if "тк_" in norm_path or "тк рф" in norm_path:
        meta["law_code"] = "ТК РФ"
    elif "нк_" in norm_path or "нк рф" in norm_path:
        meta["law_code"] = "НК РФ"
    elif "коап" in norm_path:
        meta["law_code"] = "КоАП РФ"
    elif "152фз" in norm_path or "152-фз" in norm_path:
        meta["law_code"] = "152-ФЗ"
    elif "мрот" in norm_path:
        meta["law_code"] = "82-ФЗ"

    # Article number and title
    m_art = re.search(r"Статья\s+(\d+(?:\.\d+)?)\.\s+([^\n]+)", text)
    if m_art:
        meta["article"] = m_art.group(1).strip()
        meta["article_title"] = m_art.group(2).strip()
    else:
        m_fn = re.search(r"ст(\d+(?:\.\d+)?)", path)
        if m_fn:
            meta["article"] = m_fn.group(1).strip()

    # Amendment notes (lines about changes)
    for line in text.split("\n")[:15]:
        line_clean = line.strip()
        if any(w in line_clean.lower() for w in ("изменен", "дополнен", "утратил силу")) and any(
            w in line_clean.lower() for w in ("федеральный закон", "фз", "статья")
        ):
            meta["amend_notes"].append(line_clean)

    return meta


def build_document_metadata(path: str, clean_text: str, removed_lines: list[str]) -> Document:
    """Extract metadata, classification and references for a document."""
    doc_id = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
    first_few_lines = "\n".join(lines[:10])

    is_law = (
        path.startswith("законы__")
        or "законы" in path.lower()
        or ("Источник:" in first_few_lines and "Статья" in first_few_lines)
    )
    category = "law" if is_law else "company"

    # Determine doc_type carefully
    doc_type = "other"
    if is_law:
        doc_type = "law_article"
    else:
        top_upper = "\n".join(lines[:5]).upper()
        if "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ" in top_upper:
            doc_type = "amendment"
        elif any(w in top_upper for w in ("РЕГЛАМЕНТ", "ПОЛОЖЕНИЕ", "ПОЛИТИКА", "ИНСТРУКЦИЯ")):
            doc_type = "regulation"
        elif "СЛУЖЕБНАЯ ЗАПИСКА" in top_upper:
            doc_type = "memo"
        elif "ПРИКАЗ" in top_upper:
            doc_type = "order"
        elif re.search(r"\bДОГОВОР\b", top_upper):
            doc_type = "contract"
        elif "ИСХ. №" in top_upper or "УВАЖАЕМЫЕ" in top_upper:
            doc_type = "letter"

    # Document title
    title = ""
    if lines:
        for line in lines[:4]:
            l_up = line.upper()
            if any(w in l_up for w in ("ДОГОВОР", "РЕГЛАМЕНТ", "ДОПОЛНИТЕЛЬНОЕ", "СЛУЖЕБНАЯ", "СТАТЬЯ", "О ")):
                if not l_up.startswith("РЕДАКЦИЯ УТРАТИЛА"):
                    title = line
                    break
        if not title:
            title = lines[0]
    if not title:
        title = path

    # Dates and validity
    doc_date = parse_date_to_iso(first_few_lines)
    status = "active"
    effective_from: str | None = None
    effective_to: str | None = None

    # Status detection
    lower_path = path.lower()
    first_two_lines_lower = "\n".join(lines[:2]).lower() if lines else ""

    if is_law:
        status = "active"
    else:
        if "утратил силу" in lower_path or "редакция утратила силу" in first_two_lines_lower:
            status = "repealed"

    m_eff_from = re.search(r"(?:действует\s+с|вступает\s+в\s+силу\s+с|редакция\s+от)\s+([^\n,]+)", first_few_lines, re.I)
    if m_eff_from:
        effective_from = parse_date_to_iso(m_eff_from.group(1))

    m_eff_to = re.search(r"утрати(?:ла|л|вает)\s+силу\s+с\s+([^\n,]+)", first_few_lines, re.I)
    if m_eff_to:
        to_iso = parse_date_to_iso(m_eff_to.group(1))
        if to_iso:
            try:
                d = date.fromisoformat(to_iso) - timedelta(days=1)
                effective_to = d.isoformat()
            except ValueError:
                effective_to = to_iso

    # Version group (for multi-version documents like regulations)
    version_group = ""
    if doc_type == "regulation":
        for line in lines[:6]:
            l_up = line.upper()
            if ("РЕГЛАМЕНТ" in l_up or "ПОЛОЖЕНИЕ" in l_up) and not l_up.startswith("РЕДАКЦИЯ УТРАТИЛА") and not l_up.startswith("УТВЕРЖД"):
                v_line = re.sub(r"\(.*?\)", "", line)
                version_group = norm_str(v_line).strip()
                break

    # References and document numbers
    refs: list[dict[str, str]] = []
    doc_numbers: list[str] = []

    if doc_type == "amendment":
        # Target contract referenced in amendment
        m_ref_num = re.search(r"к\s+договору[^\n№]*№\s*([A-Za-zА-Яа-яЁё\d/-]+)", clean_text[:500], re.I)
        if m_ref_num:
            target_nums = extract_doc_numbers(m_ref_num.group(0))
            if target_nums:
                refs.append({"type": "amends", "target_num": target_nums[0]})
        doc_numbers = []
    else:
        doc_numbers = extract_doc_numbers(path + "\n" + first_few_lines)

    if doc_type == "letter":
        m_reg = re.search(r"п\.\s*([\d.]+)\s+Регламента[^\n(]*\((?:ред\.\s*от\s*)?([^\)]+)\)", clean_text, re.I)
        if m_reg:
            refs.append({
                "type": "supplements",
                "target_title": "регламент согласования договоров",
                "target_clause": m_reg.group(1),
                "target_edition": parse_date_to_iso(m_reg.group(2)) or "",
            })

    parties = extract_parties(clean_text) if not is_law else []
    subject = extract_subject(clean_text) if doc_type == "contract" else ""

    meta: dict[str, Any] = {
        "category": category,
        "doc_type": doc_type,
        "doc_numbers": doc_numbers,
        "doc_date": doc_date,
        "parties": parties,
        "subject": subject,
        "status": status,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "version_group": version_group,
        "refs": refs,
        "broken_refs": [],
    }

    if is_law:
        law_meta = parse_law_meta(path, clean_text)
        meta.update(law_meta)
        if law_meta.get("article"):
            title = f"{law_meta.get('law_code', '')} ст. {law_meta.get('article', '')} «{law_meta.get('article_title', '')}»".strip()

        # Find mentions of other articles like "статьи 6, 10 и 11"
        mentioned_articles: list[str] = []
        for m in re.finditer(r"стать[ейяиюь]\s+([\d.,\sи-]+)(?:настоящего\s+(?:Федерального\s+закона|Кодекса))?", clean_text, re.I):
            num_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", m.group(1))
            for num in num_tokens:
                if num != meta.get("article") and num not in mentioned_articles:
                    mentioned_articles.append(num)
                    refs.append({
                        "type": "mentions",
                        "target_law": meta.get("law_code", ""),
                        "target_article": num,
                    })

    return Document(
        doc_id=doc_id,
        path=path,
        title=title,
        category=category,
        doc_type=doc_type,
        text=clean_text,
        meta=meta,
        removed_lines=removed_lines,
    )


def resolve_applicable_edition(
    doc: Document, event_date_str: str | None = None, today_str: str | None = None
) -> tuple[bool, str]:
    """Determine whether the document edition is applicable based on event_date or current date."""
    eff_from = doc.meta.get("effective_from")
    eff_to = doc.meta.get("effective_to")
    status = doc.meta.get("status", "active")

    if event_date_str:
        e_date = event_date_str[:7]
        f_date = (eff_from[:7] if eff_from else "0000-00")
        t_date = (eff_to[:7] if eff_to else "9999-99")
        if f_date <= e_date <= t_date:
            return True, f"Действовала на дату события ({event_date_str})"
        else:
            note = f"Не действовала на дату события ({event_date_str})"
            if eff_to and e_date > eff_to[:7]:
                note += f": утратила силу с {eff_to}"
            elif eff_from and e_date < eff_from[:7]:
                note += f": вступила в силу позже ({eff_from})"
            return False, note

    # No event date: evaluate against TODAY
    current_date = today_str or config.get_current_date()
    cur_m = current_date[:7]

    if status == "repealed" or (eff_to and cur_m > eff_to[:7]):
        return False, f"Утратила силу" + (f" с {eff_to}" if eff_to else "")

    if eff_from and cur_m < eff_from[:7]:
        return False, f"Вступает в силу с {eff_from}"

    return True, "Действующая редакция"


class DocGraph:
    """Graph of relations between documents (amends, supersedes, supplements, mentions)."""

    def __init__(self):
        self.adj: dict[str, list[tuple[str, str]]] = {}  # doc_id -> list of (target_id, relation)

    def add_edge(self, u: str, v: str, forward_rel: str, backward_rel: str) -> None:
        if u == v:
            return
        self.adj.setdefault(u, [])
        self.adj.setdefault(v, [])
        if not any(target == v and r == forward_rel for target, r in self.adj[u]):
            self.adj[u].append((v, forward_rel))
        if not any(target == u and r == backward_rel for target, r in self.adj[v]):
            self.adj[v].append((u, backward_rel))

    def get_neighbors(self, doc_id: str) -> list[tuple[str, str]]:
        return self.adj.get(doc_id, [])


def build_doc_graph(docs: list[Document]) -> DocGraph:
    """Build bidirectional document graph from metadata, numbers, and versions."""
    graph = DocGraph()

    # Build lookup indices: contracts have top priority for doc_numbers
    num_to_doc: dict[str, str] = {}
    version_groups: dict[str, list[Document]] = {}

    for d in docs:
        if d.doc_type != "amendment":
            for num in d.meta.get("doc_numbers", []):
                num_to_doc[num] = d.doc_id

        vg = d.meta.get("version_group")
        if vg:
            version_groups.setdefault(vg, []).append(d)

    # 1. Version links (supersedes / superseded_by)
    for vg, group in version_groups.items():
        if len(group) > 1:
            sorted_docs = sorted(group, key=lambda x: x.meta.get("effective_from") or "1900-01-01")
            for i in range(len(sorted_docs) - 1):
                older = sorted_docs[i]
                newer = sorted_docs[i + 1]
                if not older.meta.get("effective_to") and newer.meta.get("effective_from"):
                    try:
                        d = date.fromisoformat(newer.meta["effective_from"]) - timedelta(days=1)
                        older.meta["effective_to"] = d.isoformat()
                    except ValueError:
                        pass
                graph.add_edge(newer.doc_id, older.doc_id, "supersedes", "superseded_by")

    # 2. Reference links (amends, supplements, mentions)
    for d in docs:
        for ref in d.meta.get("refs", []):
            ref_type = ref.get("type")
            target_num = ref.get("target_num")
            if target_num and target_num in num_to_doc:
                target_id = num_to_doc[target_num]
                if ref_type == "amends":
                    graph.add_edge(d.doc_id, target_id, "amends", "amended_by")
            if ref_type == "mentions":
                if target_num and target_num in num_to_doc:
                    graph.add_edge(d.doc_id, num_to_doc[target_num], "mentions", "mentioned_by")
                elif ref.get("target_law") and ref.get("target_article"):
                    t_law = ref["target_law"]
                    t_art = ref["target_article"]
                    found_target = False
                    for candidate in docs:
                        if candidate.meta.get("law_code") == t_law and candidate.meta.get("article") == t_art:
                            graph.add_edge(d.doc_id, candidate.doc_id, "mentions", "mentioned_by")
                            found_target = True
                            break
                    if not found_target:
                        missing_ref = f"{t_law} ст. {t_art}"
                        if missing_ref not in d.meta.get("broken_refs", []):
                            d.meta.setdefault("broken_refs", []).append(missing_ref)

            if ref_type == "supplements":
                target_title = ref.get("target_title", "")
                target_ed = ref.get("target_edition", "")
                for candidate in docs:
                    vg = candidate.meta.get("version_group", "")
                    if target_title in vg or vg in target_title:
                        if not target_ed or candidate.meta.get("effective_from") == target_ed:
                            graph.add_edge(d.doc_id, candidate.doc_id, "supplements", "supplemented_by")
                            break

    return graph
