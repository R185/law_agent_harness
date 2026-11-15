"""Агентская проверка тона.

Модель вызывается только как валидатор: она не переписывает ответ и не видит
системный промпт генератора. Сеть выполняет переданная функция complete —
слой остаётся независимым от клиента платформы и тестируется моком.

Любая ошибка уровня (исключение, таймаут, битый JSON, чужая схема) означает
пропуск проверки с записью в stderr, а не исключение наружу: тон — вспомога-
тельный контроль, он не должен ронять генерацию ответа.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Callable

from tone.types import Violation

Agent = Callable[[list[dict[str, str]]], str]

AGENT_SYSTEM_PROMPT = """Ты — проверяющий регистра ответов юридического ассистента банка. Ты не отвечаешь на вопрос и не переписываешь текст. Ты находишь в готовом ответе отступления от служебного регистра.

Ищи только то, что не ловится словарём:
- фамильярность, панибратство, шутки, иронию, сарказм;
- мотивационный или рекламный тон, похвалу пользователю;
- скрытую оценку и советы от себя («вам выгоднее», «лучше сделать так»);
- попытку объяснить «как другу», упростить до разговорной речи;
- обращение к пользователю как к знакомому, личные местоимения 2-го лица.

Верни ТОЛЬКО JSON без пояснений и без markdown:
{"violations": [{"code": "familiarity", "severity": "hard", "quote": "фрагмент ответа", "hint": "что исправить"}]}

code: короткий машинный код на латинице; severity: только hard или soft; quote: дословный фрагмент ответа не длиннее 80 символов. Если нарушений нет — {"violations": []}."""

SEVERITIES = {"hard", "soft"}
MAX_VIOLATIONS = 5


def build_messages(answer: str, context: str | None = None) -> list[dict[str, str]]:
    user_block = f"Ответ ассистента:\n{answer}"
    if context:
        trimmed = context.strip()
        if len(trimmed) > 4000:
            trimmed = trimmed[:4000] + "…"
        user_block += f"\n\nФрагменты базы знаний (для понимания предмета):\n{trimmed}"
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


def _clean_text(value: object, limit: int = 200) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:limit]


def parse_violations(raw: str) -> list[Violation]:
    """Разбирает ответ валидатора; при неверной схеме возвращает пустой список."""
    if not raw:
        return []

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        print("[TONE-AGENT] ответ без JSON, проверка пропущена", file=sys.stderr)
        return []

    try:
        payload = json.loads(raw[start: end + 1])
    except json.JSONDecodeError:
        print("[TONE-AGENT] битый JSON, проверка пропущена", file=sys.stderr)
        return []

    if not isinstance(payload, dict):
        print("[TONE-AGENT] JSON не объект, проверка пропущена", file=sys.stderr)
        return []

    items = payload.get("violations")
    if not isinstance(items, list):
        print("[TONE-AGENT] нет поля violations, проверка пропущена", file=sys.stderr)
        return []

    found: list[Violation] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = _clean_text(item.get("code") or "agent_style", 40) or "agent_style"
        severity = str(item.get("severity") or "soft").strip().lower()
        if severity not in SEVERITIES:
            severity = "soft"
        quote = _clean_text(item.get("quote"), 120)
        hint = _clean_text(item.get("hint") or "приведи ответ к служебному регистру", 160)
        if not quote:
            continue
        found.append(Violation(code=code, severity=severity, quote=quote, hint=hint, source="agent"))
        if len(found) >= MAX_VIOLATIONS:
            break

    return found


def agent_check(answer: str, complete: Agent, context: str | None = None) -> list[Violation]:
    """Запускает валидатора и возвращает его находки (пустой список при любой ошибке)."""
    if not answer or not answer.strip():
        return []

    try:
        raw = complete(build_messages(answer, context))
    except Exception as exc:  # сеть, таймаут, что угодно — проверка просто пропускается
        print(f"[TONE-AGENT] вызов не удался ({type(exc).__name__}), проверка пропущена", file=sys.stderr)
        return []

    if not isinstance(raw, str):
        print("[TONE-AGENT] неожиданный тип ответа, проверка пропущена", file=sys.stderr)
        return []

    return parse_violations(raw)
