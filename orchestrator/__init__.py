"""Оркестратор минимального submission.

RAG отдаёт знания, tone проверяет регистр, оркестратор принимает решение
и пишет submission.csv. Слой не содержит кода RAG и tone.
"""

from __future__ import annotations

from orchestrator.pipeline import Orchestrator

__all__ = ["Orchestrator"]
