"""Необязательная отправка вопросов на свой VDS.

Это только зеркало входящих вопросов. Ошибка VDS не должна ломать
submission: отправка best-effort, после первого сбоя пишем в stderr и
продолжаем отвечать.
"""

from __future__ import annotations

import httpx

from orchestrator import config


class QuestionForwarder:
    def __init__(self) -> None:
        self.url = config.VDS_WEBHOOK_URL
        self.token = config.VDS_TOKEN
        self._client = (
            httpx.Client(timeout=config.VDS_TIMEOUT, trust_env=False)
            if self.url
            else None
        )
        self._failed = 0

    def send(self, qid: str, question: str) -> None:
        if self._client is None:
            return
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self._client.post(
                self.url,
                headers=headers,
                json={"id": str(qid), "question": question},
            )
            response.raise_for_status()
        except Exception as exc:
            self._failed += 1
            if self._failed <= 3:
                print(f"[vds] вопрос {qid} не отправлен: {type(exc).__name__}", flush=True)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> "QuestionForwarder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
