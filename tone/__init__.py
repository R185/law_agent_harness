"""Слой тона юридического ассистента.

Публичный интерфейс:

    build_system_prompt(base)      -> строка системного промпта с контрактом
    normalize(answer)              -> Normalized(text, removed_artifacts)
    check(answer, context, agent)  -> list[Violation]
    Violation, Normalized          -> типы результата

Слой ничего не переписывает и не вызывает модель по умолчанию: он сообщает,
что не так с ответом. Решение о переписывании, лимит попыток и текст отказа —
зона оркестратора.
"""

from __future__ import annotations

from tone.check import check, hard_only, soft_only
from tone.contract import STYLE_CONTRACT, build_system_prompt
from tone.normalize import Normalized, normalize
from tone.rules import ALL_RULES, HARD, SOFT
from tone.types import Violation

__all__ = [
    "ALL_RULES",
    "HARD",
    "SOFT",
    "STYLE_CONTRACT",
    "Normalized",
    "Violation",
    "build_system_prompt",
    "check",
    "hard_only",
    "normalize",
    "soft_only",
]
