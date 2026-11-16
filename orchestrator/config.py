"""Настройки оркестратора.

Всё, что зависит от платформы, читается здесь. Секреты не хранятся в коде.
"""

from __future__ import annotations

import os
from pathlib import Path

# Локальный .env не входит в ZIP и не переопределяет уже заданные
# переменные платформы. На платформе файла нет, поэтому импорт безопасен.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_bool(name: str, default: bool) -> bool:
    value = env(name, "").lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name, str(default)))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(env(name, str(default)))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", "data"))

API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL") or env("DEEPSEEK_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY") or env("DEEPSEEK_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL") or env("DEEPSEEK_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

# TOP_K и MAX_TOKENS платформа задаёт для baseline (у него TOP_K=4, MAX_TOKENS=1024).
# Свои значения читаем из своих имён, платформенные только логируем.
PLATFORM_TOP_K = env("TOP_K")
PLATFORM_MAX_TOKENS = env("MAX_TOKENS")
TOP_K = env_int("ORCH_TOP_K", 10, 1, 30)
CONTEXT_MAX_CHARS = env_int("CONTEXT_MAX_CHARS", 12000, 2000, 30000)
# «Думающая» модель тратит этот лимит и на рассуждение: при 1024 длинный ответ может оборваться.
MAX_TOKENS = env_int("ORCH_MAX_TOKENS", 3000, 256, 8192)
TEMPERATURE = env_float("TEMPERATURE", 0.0, 0.0, 2.0)
REWRITE_LIMIT = env_int("REWRITE_LIMIT", 1, 0, 3)
# Повторы вызова модели сверх трёх попыток ApiClient (429/5xx/таймаут, пустой ответ).
LLM_RETRIES = env_int("LLM_RETRIES", 3, 0, 6)
# Вопросы независимы: обрабатываем параллельно. MAX_WORKERS задаёт платформа с учётом лимитов прокси.
WORKERS = env_int("ORCH_WORKERS", env_int("MAX_WORKERS", 4, 1, 16), 1, 16)
# Тайм-аут платформы — 20 минут на весь запуск.
TIME_BUDGET_SEC = env_int("TIME_BUDGET_SEC", 17 * 60, 60, 3600)
REWRITE_CUTOFF = 0.75   # после этой доли бюджета ответы не переписываются

# Без ключа или URL платформа не сможет вызвать модель. Для локального
# прогона можно включить извлекающий ответ, чтобы проверить CSV и пайплайн.
OFFLINE = env_bool("ORCHESTRATOR_OFFLINE", False) or not (API_BASE and LLM_KEY)
ENABLE_DENSE = env_bool("ENABLE_DENSE", True)
ENABLE_G1 = env_bool("GUARD_G1", True)
GUARD_LLM = env_bool("GUARD_LLM", True)
GUARD_LLM_MODE = env("GUARD_LLM_MODE", "auto") or "auto"
ENABLE_JUDGE = env_bool("JUDGE", True)

FALLBACK = "В предоставленной базе знаний отсутствуют документы, позволяющие ответить на поставленный вопрос."
# Ответ, если показать нечего: пустой ответ модели или утечка служебных инструкций.
SAFE_FALLBACK = "В предоставленной базе знаний не удалось найти подтверждённый ответ на поставленный вопрос."
# Ответы на запросы без вопроса по базе знаний (guard: decline). Регистр — по контракту tone.
DECLINE_TEXTS = {
    "rules_attack": "Изменение правил работы, роли и источников ответа не допускается. "
                    "Ответ формируется только по документам предоставленной базы знаний.",
    "prompt_leak": "Внутренние инструкции и настройки не раскрываются. "
                   "Ответ формируется только по документам предоставленной базы знаний.",
    "offtopic": "В предоставленной базе знаний отсутствуют документы по данному запросу: "
                "база содержит договоры, внутренние документы компании и нормы законодательства.",
}

BASE_PROMPT = """Ты готовишь служебную правовую справку для юриста банка.
Отвечай только по фрагментам базы знаний, которые даны ниже. Не используй
внешние знания и не придумывай нормы, сроки, суммы и номера документов.
Главное правило: отвечай только на заданный вопрос — вывод, нужное значение
(срок, сумма, условие) и ссылка на документ и пункт, обычно 1–2 предложения.
Не упоминай другие пункты, суммы, даты, стороны договора, реквизиты, приказы
об утверждении, дополнительные соглашения и редакции, если они не меняют ответ
на вопрос, даже если они есть во фрагментах. Не пиши, что какой-то документ
что-то «не изменяет» или «не содержит», если об этом не спрашивали.
Особые случаи — только когда они действительно есть:
— ответа в базе нет: одной фразой скажи, что в предоставленной базе знаний
отсутствуют документы о теме вопроса (назови тему); не перечисляй содержимое
фрагментов, не отвечай по памяти и не называй статьи и нормы, которых нет во
фрагментах; можно добавить, что за ответом следует
обратиться к профильному специалисту;
— ответ есть частично: ответь на эту часть и одной фразой укажи, каких норм
или документов в базе не хватает;
— вопрос неоднозначен и подходят несколько документов: кратко ответь по каждому,
называя документ, без встречного вопроса;
— в вопросе неверная посылка: прямо укажи, что сказано в документе;
— указана дата события: применяй редакцию, действовавшую на эту дату;
— вывод зависит от обстоятельств, которых нет в вопросе: одна короткая оговорка.
Ответ — одна строка, без markdown и списков."""
