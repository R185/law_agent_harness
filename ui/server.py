"""Тонкая HTTP-обёртка над orchestrator для UI «ПСБ — Юридический ассистент».

Запуск:
    .venv/bin/uvicorn ui.server:app --port 8765
    ORCHESTRATOR_OFFLINE=1 .venv/bin/uvicorn ui.server:app --port 8765   # без LLM

UI отдаётся статикой, ответ — POST /api/ask {question, history?} -> {answer, sources}.
Источники берутся из RetrievalResult, который оркестратор строит по пути к ответу:
подкласс перехватывает _retrieve и запоминает последний результат, ядро не патчится.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from orchestrator.config import DATA_DIR
from orchestrator.pipeline import Orchestrator

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
KB_DIR = Path(os.getenv("KB_DIR", "") or (ROOT.parent / "knowledge-base"))


class CapturingOrchestrator(Orchestrator):
    """Оркестратор, запоминающий результат поиска для панели источников."""

    last_result: Any = None

    def _retrieve(self, question: str, history: str | None = None):
        result = super()._retrieve(question, history=history)
        self.last_result = result
        return result


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)


app = FastAPI(title="ПСБ · Юридический ассистент", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=ROOT), name="assets")

_lock = threading.Lock()
_orchestrator: CapturingOrchestrator | None = None


def orchestrator() -> CapturingOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CapturingOrchestrator(source_path=KB_DIR)
    return _orchestrator


def dialog_text(question: str, history: list[dict[str, str]]) -> str:
    """Оркестратор принимает диалог текстом: parse_dialog разбирает реплики."""
    lines: list[str] = []
    for turn in history[-8:]:
        role = "Пользователь" if turn.get("role") == "user" else "Ассистент"
        text = (turn.get("content") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    lines.append(f"Пользователь: {question.strip()}")
    return "\n".join(lines)


def cited_clauses(answer: str) -> set[str]:
    """Пункты, которые ответ реально называет: 'п. 4.2', 'п.4.2, 4.3'."""
    found: set[str] = set()
    for chunk in re.findall(r"\bп(?:\.|ункт[а-я]*)\s*([0-9][0-9.,\s]*)", answer or "", flags=re.IGNORECASE):
        for part in re.findall(r"[0-9]+(?:\.[0-9]+)+", chunk):
            found.add(part)
    return found


def source_rows(result: Any, answer: str = "", limit: int = 3) -> list[dict[str, str]]:
    """Evidence + DocumentMatch -> строки-ссылки для UI: сначала пункты, на которые ответ ссылается."""
    cited = cited_clauses(answer)
    docs = {getattr(d, "doc_id", ""): d for d in (getattr(result, "documents", None) or [])}
    rows: list[tuple[tuple[int, float], dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    for evidence in (getattr(result, "evidence", None) or []):
        doc_id = getattr(evidence, "doc_id", "") or ""
        clause = ", ".join(getattr(evidence, "clause_ids", None) or []) or getattr(evidence, "section", "")
        key = (doc_id, clause)
        if key in seen:
            continue
        seen.add(key)
        doc = docs.get(doc_id)
        title = (getattr(doc, "title", "") or getattr(evidence, "title", "") or doc_id).strip()
        if clause:
            title = f"{title}, п. {clause}" if not clause.lower().startswith("п.") else f"{title}, {clause}"
        meta_bits: list[str] = []
        if doc is not None:
            parties = getattr(doc, "parties", None) or []
            if parties:
                meta_bits.append(parties[0])
            status = str(getattr(doc, "status", "") or "")
            if status == "active":
                meta_bits.append("действует")
            elif status == "repealed":
                meta_bits.append("утратил силу")
            if getattr(doc, "applicable_edition", False) is False and getattr(doc, "edition_note", ""):
                note = doc.edition_note
                if "утратил" not in note.lower():
                    meta_bits.append(note)
        rows.append(((1 if set(cited) & set(getattr(evidence, "clause_ids", None) or []) else 0,
                      float(getattr(evidence, "relevance", 0.0) or 0.0)), {
            "n": "0",
            "docId": doc_id,
            "title": title,
            "meta": " · ".join(bit for bit in meta_bits if bit),
            "snippet": " ".join((getattr(evidence, "text", "") or "").split())[:600],
        }))
    rows.sort(key=lambda pair: pair[0], reverse=True)
    out: list[dict[str, str]] = []
    for _, row in rows[:limit]:
        row["n"] = str(len(out) + 1)
        out.append(row)
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> JSONResponse:
    try:
        orc = orchestrator()
    except Exception as exc:  # noqa: BLE001 — health должен отвечать, а не падать
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=503)
    return JSONResponse({
        "ok": True,
        "knowledge_base": str(KB_DIR),
        "documents": len(getattr(orc.rag, "docs", []) or []),
        "offline": bool(getattr(orc, "offline", False)),
        "data_dir": str(DATA_DIR),
    })


@app.post("/api/ask")
def ask(payload: AskRequest) -> JSONResponse:
    question = payload.question.strip()
    if not question:
        return JSONResponse({"detail": "empty question"}, status_code=422)
    try:
        with _lock:
            orc = orchestrator()
            orc.last_result = None
            answer = orc.answer(dialog_text(question, payload.history), qid="ui")
            rows = source_rows(orc.last_result, answer) if orc.last_result is not None else []
    except Exception as exc:  # noqa: BLE001 — UI должен показать причину, а не 500 без тела
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)
    return JSONResponse({"answer": answer, "sources": rows})
