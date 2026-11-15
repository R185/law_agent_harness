"""Настройки слоя тона.

Слой не зависит от конфигурации пайплайна: единственные флаги, которые его
касаются, читаются здесь. Флаги переписывания (лимит попыток, FALLBACK)
принадлежат оркестратору и в этот пакет не входят.
"""

from __future__ import annotations

import os

TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return default


def tone_enabled() -> bool:
    """ENABLE_TONE: вставлять ли контракт регистра в системный промпт."""
    return env_bool("ENABLE_TONE", True)


def agent_enabled() -> bool:
    """ENABLE_TONE_AGENT: запускать ли агентскую проверку, если она передана."""
    return env_bool("ENABLE_TONE_AGENT", False)
