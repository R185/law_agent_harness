"""Общие типы слоя тона."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Violation:
    """Одно нарушение регистра или формата.

    code     — машинный код правила (chatbot_opening, second_person, ...);
    severity — hard (оркестратору нужно переписать ответ) или soft (достаточно лога);
    quote    — фрагмент ответа, по которому сработало правило;
    hint     — что именно исправить, формулировка для промпта переписывания;
    source   — regex или agent: откуда пришло нарушение.
    """

    code: str
    severity: str
    quote: str
    hint: str
    source: str = "regex"
