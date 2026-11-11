from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.insufficient_info.clarify import build_clarification
from harness.insufficient_info.judge import evaluate_insufficient_info
from harness.types import JudgeDecision, JudgeInput, RetrievedChunk


CASES_PATH = Path(__file__).parent / "demo_cases.json"


def _load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_demo_decision(case: dict) -> None:
    chunks = [RetrievedChunk(**c) for c in case.get("chunks", [])]
    verdict = evaluate_insufficient_info(
        JudgeInput(question=case["question"], chunks=chunks)
    )
    assert verdict.decision.value == case["expected_decision"]
    if case.get("expect_premise_error"):
        assert verdict.analysis.premise_errors


def test_clarify_alfa_questions() -> None:
    case = next(c for c in _load_cases() if c["id"] == "17_turn1")
    chunks = [RetrievedChunk(**c) for c in case["chunks"]]
    verdict = evaluate_insufficient_info(JudgeInput(question=case["question"], chunks=chunks))
    assert verdict.decision == JudgeDecision.CLARIFY
    clar = build_clarification(verdict)
    assert clar.questions
    assert any("Альфа" in q for q in clar.questions)


def test_partial_pd_clarification() -> None:
    case = next(c for c in _load_cases() if c["id"] == "13")
    chunks = [RetrievedChunk(**c) for c in case["chunks"]]
    verdict = evaluate_insufficient_info(JudgeInput(question=case["question"], chunks=chunks))
    clar = build_clarification(verdict)
    assert len(clar.questions) >= 1
