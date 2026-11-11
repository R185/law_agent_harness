"""LLM-as-judge для оценки аргументации отказа/уточнения (опционально, нужен API)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from harness.insufficient_info.clarify import build_clarification
from harness.insufficient_info.judge import evaluate_insufficient_info
from harness.types import JudgeInput, RetrievedChunk

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RUBRIC = """
Оцени ответ судьи юридического ассистента банка по шкале 1-5:
5 — корректный класс решения, деловой тон, явная аргументация (БЗ, дата, неоднозначность).
1 — неверное решение или галлюцинации.
Верни JSON: {"score": int, "reason": "..."}
"""


def run_case(case: dict) -> dict:
    chunks = [RetrievedChunk(**c) for c in case.get("chunks", [])]
    inp = JudgeInput(question=case["question"], chunks=chunks)
    verdict = evaluate_insufficient_info(inp)
    clar = build_clarification(verdict)
    return {
        "id": case["id"],
        "decision": verdict.decision.value,
        "user_facing_text": verdict.user_facing_text,
        "questions": clar.questions,
        "analysis": verdict.analysis.model_dump(mode="json"),
    }


def llm_score(payload: dict) -> dict | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    message = (
        f"Эталон expected_decision: {payload.get('expected_decision')}\n"
        f"Фактический вывод:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        + RUBRIC
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        temperature=0,
    )
    text = resp.choices[0].message.content or "{}"
    return json.loads(text)


def main() -> int:
    cases_path = Path(__file__).parent / "demo_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        out = run_case(case)
        out["expected_decision"] = case["expected_decision"]
        score = llm_score(out)
        out["llm_judge"] = score
        results.append(out)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")):
        print("LLM judge skipped: no API key", file=sys.stderr)
        return 0
    scores = [r["llm_judge"]["score"] for r in results if r.get("llm_judge") and "score" in r["llm_judge"]]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"Average LLM judge score: {avg:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
