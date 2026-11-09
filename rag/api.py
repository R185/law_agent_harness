from __future__ import annotations

import math
import sys
import time
from typing import Any

import httpx

from rag import config


class ApiClient:
    """HTTP client for OpenAI-compatible platform endpoints with retries and backoff."""

    def __init__(self, timeout: float = 90.0, trust_env: bool = False):
        # trust_env=False avoids crashing on unsupported socks proxies in local environments
        self._client = httpx.Client(timeout=timeout, trust_env=trust_env)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _post(self, path: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not config.API_BASE:
            raise ValueError("API_BASE (LLM_BASE_URL) is not set")
        if not key:
            raise ValueError(f"API key missing for endpoint {path}")

        url = config.API_BASE + path
        headers = {"Authorization": f"Bearer {key}"}

        max_attempts = 3
        backoffs = [1.0, 2.0, 4.0]

        for attempt in range(max_attempts):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_attempts - 1:
                        sleep_time = backoffs[attempt]
                        print(
                            f"[API] Warning: status {resp.status_code} for {path}. Retry {attempt + 1}/{max_attempts} after {sleep_time}s",
                            file=sys.stderr,
                        )
                        time.sleep(sleep_time)
                        continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < max_attempts - 1:
                    sleep_time = backoffs[attempt]
                    print(
                        f"[API] Warning: {type(exc).__name__} for {path}. Retry {attempt + 1}/{max_attempts} after {sleep_time}s",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_time)
                    continue
                raise

        raise RuntimeError(f"Failed to post to {path} after {max_attempts} attempts")

    def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Compute L2-normalized embeddings via POST /embeddings."""
        if not texts:
            return []

        model_name = model or config.EMB_MODEL
        payload = {
            "model": model_name,
            "input": texts,
            "encoding_format": "float",
        }
        res = self._post("/embeddings", config.EMBED_KEY, payload)
        data = res.get("data", [])
        vectors: list[list[float] | None] = [None] * len(texts)

        for pos, item in enumerate(data):
            idx = int(item.get("index", pos))
            if 0 <= idx < len(texts):
                raw_vec = item.get("embedding", [])
                norm = math.sqrt(sum(float(x) ** 2 for x in raw_vec)) or 1.0
                vectors[idx] = [float(x) / norm for x in raw_vec]

        if any(v is None for v in vectors):
            raise ValueError(f"Incomplete embeddings received: got {sum(v is not None for v in vectors)} of {len(texts)}")
        return [v for v in vectors if v is not None]

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None, model: str | None = None
    ) -> list[tuple[int, float]]:
        """Rank documents via POST /rerank. Returns list of (original_index, relevance_score)."""
        if not documents:
            return []

        model_name = model or config.RERANKER_MODEL
        n = top_n if top_n is not None else len(documents)
        payload = {
            "model": model_name,
            "query": query,
            "documents": documents,
            "top_n": min(n, len(documents)),
        }
        res = self._post("/rerank", config.RERANK_KEY, payload)
        items = res.get("results", res.get("data", []))

        results: list[tuple[int, float]] = []
        for it in items:
            idx = int(it.get("index", 0))
            score = float(it.get("relevance_score", it.get("score", 0.0)))
            results.append((idx, score))
        return results

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call POST /chat/completions."""
        model_name = model or config.LLM_MODEL
        t = temperature if temperature is not None else config.TEMPERATURE
        m_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": t,
            "max_tokens": m_tokens,
        }
        res = self._post("/chat/completions", config.LLM_KEY, payload)
        choices = res.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from LLM")
        msg = choices[0].get("message", {})
        return msg.get("content", "").strip()
