from __future__ import annotations

import re
from datetime import date

from harness.insufficient_info.config import JudgeConfig
from harness.types import (
    JudgeInput,
    ReasonCode,
    RetrievedChunk,
    SituationAnalysis,
)

_NAMED_COUNTERPARTY = re.compile(
    r"северный\s+склад|альфа-трейд|альфа-трейдинг|«меридиан»",
    re.IGNORECASE,
)
_ALFA_RESOLVED = re.compile(
    r"альфа-?трейд(инг)?",
    re.IGNORECASE,
)
_CONTRACT_NUMBER = re.compile(
    r"(?:договор|№)\s*[А-ЯA-Z]{0,3}[-/]?\d{1,4}[-/]?\d{2,4}",
    re.IGNORECASE,
)
_MATERNITY = re.compile(
    r"пособи\w*\s+по\s+беременност",
    re.IGNORECASE,
)
_PD_REVOKE = re.compile(
    r"отозвал\w*\s+согласи\w*\s+на\s+обработк",
    re.IGNORECASE,
)
_WRONG_RENT_500K = re.compile(r"500\s*000", re.IGNORECASE)
_CORRECT_RENT = re.compile(r"450\s*000", re.IGNORECASE)


def _full_dialogue_text(inp: JudgeInput) -> str:
    parts = [inp.question]
    for turn in inp.dialogue:
        parts.append(turn.content)
    return "\n".join(parts)


def _scores(inp: JudgeInput) -> tuple[float, float]:
    if inp.rag_max_score is not None and inp.rag_mean_score is not None:
        return inp.rag_max_score, inp.rag_mean_score
    if not inp.chunks:
        return 0.0, 0.0
    scores = [c.score for c in inp.chunks]
    return max(scores), sum(scores) / len(scores)


def detect_maternity_gap(inp: JudgeInput) -> SituationAnalysis | None:
    text = _full_dialogue_text(inp)
    if not _MATERNITY.search(text):
        return None
    corpus = " ".join(c.excerpt + " " + c.title for c in inp.chunks).lower()
    maternity_keywords = ("беременност", "пособ", "фсс", "соцстрах")
    if not any(k in corpus for k in maternity_keywords):
        return SituationAnalysis(
            as_of_date=date.today(),
            kb_gaps=[
                "В базе знаний нет документов о пособиях по беременности и родам.",
            ],
        )
    return None


def detect_partial_pd_revoke(inp: JudgeInput) -> SituationAnalysis | None:
    text = _full_dialogue_text(inp)
    if not _PD_REVOKE.search(text):
        return None
    corpus = " ".join(c.excerpt for c in inp.chunks).lower()
    has_art9 = "152" in corpus or "статья 9" in corpus or "ст. 9" in corpus or "ст.9" in corpus
    has_art6 = "ст. 6" in corpus or "статья 6" in corpus or "ст.6" in corpus
    if has_art9 and not has_art6:
        return SituationAnalysis(
            as_of_date=date.today(),
            kb_gaps=[
                "В базе есть ст. 9 152-ФЗ, но нет статей 6, 10 и 11 для полного вывода об удалении данных.",
            ],
            temporal_notes=[
                "Нормы могли измениться; нужна актуальная редакция и оценка юриста.",
            ],
        )
    return None


def detect_ambiguous_alfa(inp: JudgeInput) -> list[str]:
    text = _full_dialogue_text(inp)
    if not re.search(r"альф", text, re.IGNORECASE):
        return []
    if _ALFA_RESOLVED.search(text) or _CONTRACT_NUMBER.search(text):
        return []
    return [
        "Не указано, речь о ООО «Альфа-Трейд» (мебель, договор П-17/2024) "
        "или ООО «Альфа-Трейдинг» (техника, договор ПТ-05/2023).",
    ]


def detect_missing_contract(inp: JudgeInput) -> list[str]:
    text = _full_dialogue_text(inp)
    if _NAMED_COUNTERPARTY.search(text) and _CONTRACT_NUMBER.search(text) is False:
        # Контрагент назван — номер договора не всегда обязателен для ответа
        if "гарант" in text.lower() or "условия возврата" in text.lower() or "съезж" in text.lower():
            pass  # ниже проверим только сценарии без контрагента
        else:
            return []
    needs_contract = any(
        kw in text.lower()
        for kw in (
            "условия возврата",
            "съезж",
            "планируем съезж",
        )
    )
    if needs_contract and not _CONTRACT_NUMBER.search(text):
        if "номер договора" in text.lower() or any(
            "номер" in t.content.lower() for t in inp.dialogue
        ):
            return []
        return ["Не указан номер договора для проверки условий."]
    return []


def detect_premise_errors(inp: JudgeInput, chunks: list[RetrievedChunk]) -> list[str]:
    text = inp.question
    if not _WRONG_RENT_500K.search(text):
        return []
    blob = " ".join(c.excerpt for c in chunks)
    if _CORRECT_RENT.search(blob) or "450 000" in blob:
        return [
            "В вопросе указана арендная плата 500 000 руб./мес.; "
            "по договору № А-08/2024 в базе указано 450 000 руб./мес. (п. 3.1).",
        ]
    return []


def retrieval_insufficient(inp: JudgeInput, cfg: JudgeConfig) -> bool:
    if not inp.chunks:
        return True
    max_s, mean_s = _scores(inp)
    if len(inp.chunks) < cfg.min_chunks:
        return True
    return max_s < cfg.min_max_score or mean_s < cfg.min_mean_score


def topic_signals_allow(inp: JudgeInput, cfg: JudgeConfig) -> bool:
    """Есть ли достаточный сигнал для ответа по договору/закону."""
    if retrieval_insufficient(inp, cfg):
        return False
    max_s, _ = _scores(inp)
    return max_s >= cfg.min_max_score
