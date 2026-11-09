from __future__ import annotations

import os
from pathlib import Path


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int = -2_000_000_000, high: int = 2_000_000_000) -> int:
    try:
        val = int(env(name, str(default)))
        return max(low, min(high, val))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 1000.0) -> float:
    try:
        val = float(env(name, str(default)))
        return max(low, min(high, val))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    val = env(name, "").lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on", "y"}


# Platform paths and endpoints
DATA_DIR = Path(env("DATA_DIR", "data"))

_raw_base = env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")
API_BASE = _raw_base.rstrip("/") if _raw_base else ""

LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "cohere/rerank-5-fast")

MAX_WORKERS = env_int("MAX_WORKERS", 2, 1, 16)
TEMPERATURE = env_float("TEMPERATURE", 0.0, 0.0, 2.0)
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 64, 8192)

# Retrieval parameters
BM25_TOP_K = env_int("BM25_TOP_K", 30, 5, 100)
DENSE_TOP_K = env_int("DENSE_TOP_K", 30, 5, 100)
RRF_K = env_int("RRF_K", 60, 1, 200)
FUSED_TOP_K = env_int("FUSED_TOP_K", 30, 5, 100)
MAX_CANDIDATES = env_int("MAX_CANDIDATES", 45, 10, 100)
CONTEXT_TOP_N = env_int("CONTEXT_TOP_N", 10, 1, 30)
CONTEXT_MAX_CHARS = env_int("CONTEXT_MAX_CHARS", 12000, 1000, 50000)

RERANK_EMPTY_THRESHOLD = env_float("RERANK_EMPTY_THRESHOLD", 0.15, 0.0, 1.0)
RERANK_WITH_INSTRUCTION = env_bool("RERANK_WITH_INSTRUCTION", True)
ENABLE_JUDGE = env_bool("ENABLE_JUDGE", False)  # Disabled by default until judge is tuned
ENABLE_FOLLOWUP = env_bool("ENABLE_FOLLOWUP", False)
ENABLE_QUERY_LLM = env_bool("ENABLE_QUERY_LLM", True)

# Boosts and penalties
BOOST_TARGET = env_float("BOOST_TARGET", 2.0)
PENALTY_CONFUSABLE = env_float("PENALTY_CONFUSABLE", 0.3)
PENALTY_REPEALED = env_float("PENALTY_REPEALED", 0.8)
BOOST_EVENT_VERSION = env_float("BOOST_EVENT_VERSION", 1.5)

# Chunking thresholds
CHUNK_TARGET = env_int("CHUNK_TARGET", 1000, 200, 3000)
CHUNK_MAX = env_int("CHUNK_MAX", 1500, 400, 4000)
LAW_CHUNK_TARGET = env_int("LAW_CHUNK_TARGET", 1200, 300, 3000)
SMALL_DOC_CHARS = env_int("SMALL_DOC_CHARS", 1800, 500, 5000)
SMALL_ARTICLE_CHARS = env_int("SMALL_ARTICLE_CHARS", 2500, 500, 10000)

TODAY = env("TODAY", "")
FALLBACK = "Не удалось получить ответ."
