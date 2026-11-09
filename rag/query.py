from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rag import config
from rag.api import ApiClient
from rag.text import extract_doc_numbers, norm_str


@dataclass
class ParsedDialog:
    history: str
    last_user: str


@dataclass
class QueryAnalysis:
    standalone_question: str
    search_queries: list[str]
    entities: dict[str, list[str]] = field(default_factory=dict)
    event_date: str | None = None
    ambiguous_entities: list[str] = field(default_factory=list)
    question_type: str = "fact"


DIALOG_PREFIX_RE = re.compile(r"^(?:пользователь|ассистент|user|assistant):\s*", re.I)


def parse_dialog(raw_question: str) -> ParsedDialog:
    """Split multi-turn conversation text into history and the final user question."""
    lines = raw_question.strip().split("\n")
    user_blocks: list[str] = []
    has_prefixes = any(DIALOG_PREFIX_RE.match(line.strip()) for line in lines)

    if not has_prefixes:
        return ParsedDialog(history="", last_user=raw_question.strip())

    turns: list[tuple[str, str]] = []  # (role, text)
    current_role = "пользователь"
    current_lines: list[str] = []

    for line in lines:
        s = line.strip()
        m = DIALOG_PREFIX_RE.match(s)
        if m:
            if current_lines:
                turns.append((current_role, "\n".join(current_lines).strip()))
                current_lines = []
            prefix = m.group(0).lower()
            if "ассистент" in prefix or "assistant" in prefix:
                current_role = "ассистент"
            else:
                current_role = "пользователь"
            content = s[m.end():].strip()
            if content:
                current_lines.append(content)
        else:
            if s:
                current_lines.append(s)

    if current_lines:
        turns.append((current_role, "\n".join(current_lines).strip()))

    # Find the last turn by user
    last_user_turn_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "пользователь":
            last_user_turn_idx = i
            break

    if last_user_turn_idx == -1:
        return ParsedDialog(history="", last_user=raw_question.strip())

    last_user = turns[last_user_turn_idx][1]
    history_turns = turns[:last_user_turn_idx]
    history_lines = [f"{'Пользователь' if r == 'пользователь' else 'Ассистент'}: {t}" for r, t in history_turns]
    history = "\n".join(history_lines)

    return ParsedDialog(history=history, last_user=last_user)


MONTHS_GENITIVE = {
    "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
    "мая": "05", "июня": "06", "июля": "07", "августа": "08",
    "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12",
    "январе": "01", "феврале": "02", "марте": "03", "апреле": "04",
    "мае": "05", "июне": "06", "июле": "07", "августе": "08",
    "сентябре": "09", "октябре": "10", "ноябре": "11", "декабре": "12",
}


def extract_event_date_heuristic(text: str) -> str | None:
    """Extract event dates like 'в августе 2024 года', '01.10.2024', 'в 2024 году'."""
    # 1. DD.MM.YYYY
    m1 = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", text)
    if m1:
        return f"{m1.group(3)}-{m1.group(2)}-{m1.group(1)}"

    # 2. в августе 2024 [года]
    m2 = re.search(
        r"(?:в|к|с|до|после)\s+(январ[ея]|феврал[ея]|март[еа]|апрел[ея]|ма[ея]|июн[ея]|июл[ея]|август[еа]|сентябр[ея]|октябр[ея]|ноябр[ея]|декабр[ея])\s+(\d{4})",
        text,
        re.I,
    )
    if m2:
        m_str = m2.group(1).lower()
        mon = MONTHS_GENITIVE.get(m_str, "01")
        yr = m2.group(2)
        return f"{yr}-{mon}"

    # 3. в 2024 году
    m3 = re.search(r"\b(?:в|к|с|до)\s+(\d{4})\s*г(?:оду)?\b", text, re.I)
    if m3:
        return m3.group(1)

    return None


def extract_entities_heuristic(dialog: ParsedDialog) -> tuple[list[str], list[str]]:
    """Extract organizations and document numbers, prioritizing user's current turn."""
    full_text = f"{dialog.history}\n{dialog.last_user}" if dialog.history else dialog.last_user
    doc_nums = extract_doc_numbers(full_text)

    # Check organizations mentioned specifically in last user message
    curr_orgs: list[str] = []
    for match in re.finditer(r"[«\"]([^»\"]+)[»\"]", dialog.last_user):
        val = match.group(1).strip()
        if len(val) >= 3 and not val.lower().startswith("пользователь") and not val.lower().startswith("ассистент"):
            if val not in curr_orgs:
                curr_orgs.append(val)

    lower_last = dialog.last_user.lower()
    if re.search(r"\bальфа-трейдинг\b", lower_last):
        if "Альфа-Трейдинг" not in curr_orgs:
            curr_orgs.append("Альфа-Трейдинг")
    elif re.search(r"\bальфа-трейд\b", lower_last):
        if "Альфа-Трейд" not in curr_orgs:
            curr_orgs.append("Альфа-Трейд")

    if "северный склад" in lower_last and "Северный Склад" not in curr_orgs:
        curr_orgs.append("Северный Склад")

    # If user explicitly specified org in last turn, use that!
    if curr_orgs:
        return curr_orgs, doc_nums

    # Otherwise extract from full history
    orgs: list[str] = []
    for match in re.finditer(r"[«\"]([^»\"]+)[»\"]", full_text):
        val = match.group(1).strip()
        if len(val) >= 3 and not val.lower().startswith("пользователь") and not val.lower().startswith("ассистент"):
            if val not in orgs:
                orgs.append(val)

    lower_t = full_text.lower()
    if re.search(r"\bальфа-трейдинг\b", lower_t):
        if "Альфа-Трейдинг" not in orgs:
            orgs.append("Альфа-Трейдинг")
    elif re.search(r"\bальфа-трейд\b", lower_t):
        if "Альфа-Трейд" not in orgs:
            orgs.append("Альфа-Трейд")
    elif "альф" in lower_t and not any("альфа" in o.lower() for o in orgs):
        orgs.append("Альфа")

    if "северный склад" in lower_t and "Северный Склад" not in orgs:
        orgs.append("Северный Склад")

    return orgs, doc_nums


QUERY_ANALYZER_SYSTEM = """Ты — модуль анализа запросов юридического ассистента ООО «Меридиан». Ты НЕ отвечаешь на вопрос, а готовишь поисковые запросы по внутренней базе знаний (договоры компании, внутренние регламенты, письма, статьи ТК РФ, НК РФ, КоАП РФ, 152-ФЗ и др.).
Верни ТОЛЬКО JSON без пояснений:
{
 "standalone_question": "последний вопрос пользователя, переписанный так, чтобы он был понятен без истории диалога: подставь названия контрагентов, номера договоров, предмет обсуждения из истории",
 "search_queries": ["2–4 коротких поисковых запроса разными формулировками, языком документа (юридические термины: 'срок испытания', 'приёмка по качеству', 'арендная плата', 'одобрение генеральным директором')"],
 "entities": {"organizations": ["точные названия как в вопросе/истории, без ООО и кавычек"], "doc_numbers": ["П-17/2024"], "laws": ["ТК РФ ст. 70"]},
 "event_date": "YYYY-MM-DD или YYYY-MM или null — дата события/сделки, о которой спрашивают, если указана",
 "ambiguous_entities": ["если пользователь назвал контрагента неоднозначно, например просто 'Альфа'"],
 "question_type": "fact|yes_no|why|procedure|comparison|calculation|other"
}
Правила: не добавляй фактов, которых нет в вопросе/истории; не отвечай на вопрос; если в истории ассистент уточнил контрагента/номер, а пользователь подтвердил — используй уточнённое значение."""


def analyze_query(
    raw_question: str, client: ApiClient | None = None
) -> tuple[ParsedDialog, QueryAnalysis]:
    """Parse dialogue and analyze user question with LLM and robust heuristic fallback."""
    dialog = parse_dialog(raw_question)
    last_user = dialog.last_user
    history = dialog.history

    # Run heuristic extractor
    full_text = f"{history}\n{last_user}" if history else last_user
    h_orgs, h_doc_nums = extract_entities_heuristic(dialog)
    h_event_date = extract_event_date_heuristic(full_text)

    # Check for ambiguity in entities
    ambiguous: list[str] = []
    for org in h_orgs:
        if org.lower() in {"альфа", "северный"}:
            ambiguous.append(org)

    # Heuristic fallback standalone question
    h_standalone = last_user
    if history:
        # If user refers to prior context (e.g. "А за сколько дней...", "П-17/2024. Доставка...")
        h_standalone = f"{last_user} (Контекст: {', '.join(h_doc_nums + h_orgs)})" if (h_doc_nums or h_orgs) else last_user

    h_queries = [last_user]
    if h_standalone != last_user:
        h_queries.append(h_standalone)
    # Add subject-specific queries
    if h_doc_nums:
        for num in h_doc_nums:
            h_queries.append(f"договор {num}")

    analysis = QueryAnalysis(
        standalone_question=h_standalone,
        search_queries=h_queries,
        entities={"organizations": h_orgs, "doc_numbers": h_doc_nums, "laws": []},
        event_date=h_event_date,
        ambiguous_entities=ambiguous,
        question_type="fact",
    )

    # If LLM is disabled or client is None or API key is absent, return heuristic
    if not config.ENABLE_QUERY_LLM or client is None or not config.LLM_KEY:
        return dialog, analysis

    # Attempt LLM query analysis
    try:
        user_prompt = f"История диалога:\n{history or '—'}\n\nПоследний вопрос пользователя:\n{last_user}"
        resp_str = client.chat_completion(
            messages=[
                {"role": "system", "content": QUERY_ANALYZER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        # Parse JSON from response
        first_brace = resp_str.find("{")
        last_brace = resp_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            data = json.loads(resp_str[first_brace:last_brace + 1])
            llm_standalone = str(data.get("standalone_question") or "").strip()
            llm_queries = [str(q).strip() for q in data.get("search_queries", []) if str(q).strip()]
            llm_entities = data.get("entities") or {}
            llm_event_date = data.get("event_date")
            llm_ambiguous = data.get("ambiguous_entities") or []

            # Merge with heuristics
            all_queries = [llm_standalone] if llm_standalone else [last_user]
            for q in llm_queries + h_queries:
                if q and q not in all_queries:
                    all_queries.append(q)

            merged_orgs = list(set(h_orgs + [str(o) for o in llm_entities.get("organizations", [])]))
            merged_nums = list(set(h_doc_nums + [str(n) for n in llm_entities.get("doc_numbers", [])]))
            merged_laws = [str(l) for l in llm_entities.get("laws", [])]

            merged_date = llm_event_date if llm_event_date and llm_event_date != "null" else h_event_date

            analysis = QueryAnalysis(
                standalone_question=llm_standalone or h_standalone,
                search_queries=all_queries[:5],
                entities={
                    "organizations": merged_orgs,
                    "doc_numbers": merged_nums,
                    "laws": merged_laws,
                },
                event_date=str(merged_date) if merged_date else None,
                ambiguous_entities=list(set(ambiguous + [str(a) for a in llm_ambiguous])),
                question_type=str(data.get("question_type") or "fact"),
            )
    except Exception as exc:
        print(f"[QUERY] LLM analysis fallback to heuristic: {exc}", file=sys.stderr)

    return dialog, analysis
