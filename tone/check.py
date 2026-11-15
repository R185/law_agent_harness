"""Проверка соответствия ответа контракту регистра.

check() — чистая функция: не вызывает модель, не делает ретраев, не меняет
текст и не знает, что такое лимит попыток. Она возвращает список нарушений,
решение о переписывании принимает оркестратор.
"""

from __future__ import annotations

import re

from tone import settings
from tone.agent import Agent, agent_check
from tone.rules import ALL_RULES, HARD, SOFT
from tone.types import Violation

EMPTY_HINT = "ответ пуст: верни отказ по правилам контракта"
QUOTE_WINDOW = 30
MAX_PER_RULE = 2


def _quote(answer: str, start: int, end: int) -> str:
    left = max(0, start - QUOTE_WINDOW)
    right = min(len(answer), end + QUOTE_WINDOW)
    return " ".join(answer[left:right].split())


def _is_quoted(answer_fragment: str, context: str) -> bool:
    """Фрагмент дословно встречается в контексте — значит это цитата, не нарушение."""
    fragment = answer_fragment.strip()
    if not fragment or len(fragment) < 2:
        return False
    pattern = r"(?<![\w-])" + re.escape(fragment) + r"(?![\w-])"
    return re.search(pattern, context, re.IGNORECASE) is not None


def _scan(answer: str, context: str | None) -> list[Violation]:
    found: list[Violation] = []
    seen: set[tuple[str, str]] = set()

    for rule in ALL_RULES:
        count = 0
        for match in rule.regex.finditer(answer):
            snippet = match.group(0).strip()
            if context and rule.category != "format" and _is_quoted(snippet, context):
                continue
            key = (rule.code, snippet.lower())
            if key in seen:
                continue
            seen.add(key)
            found.append(
                Violation(
                    code=rule.code,
                    severity=rule.severity,
                    quote=_quote(answer, match.start(), match.end()),
                    hint=rule.hint,
                    source="regex",
                )
            )
            count += 1
            if count >= MAX_PER_RULE:
                break

    return found


def _order(violations: list[Violation]) -> list[Violation]:
    return sorted(violations, key=lambda v: 0 if v.severity == HARD else 1)


def check(
    answer: str,
    context: str | None = None,
    agent: Agent | None = None,
) -> list[Violation]:
    """Возвращает нарушения регистра и формата; пустой список — ответ соответствует контракту.

    context — фрагменты базы знаний: дословно совпавший фрагмент считается
    цитатой и не флагуется. Это не проверка фактов, только защита от ложных
    срабатываний на языке документов.

    agent — функция вызова модели-валидатора; используется, если задан и
    включён ENABLE_TONE_AGENT. Без неё проверка полностью локальна.
    """
    if not answer or not answer.strip():
        return [Violation(code="empty_answer", severity=HARD, quote="", hint=EMPTY_HINT)]

    found = _scan(answer, context)

    if agent is not None and settings.agent_enabled():
        found.extend(agent_check(answer, agent, context))

    return _order(found)


def hard_only(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == HARD]


def soft_only(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if v.severity == SOFT]


def describe(violations: list[Violation]) -> str:
    """Короткое описание нарушений для лога оркестратора."""
    return "; ".join(f"{v.code}: «{v.quote}»" for v in violations)
