from __future__ import annotations

from harness.types import ClarificationResult, JudgeDecision, JudgeVerdict, ReasonCode


def build_clarification(verdict: JudgeVerdict) -> ClarificationResult:
    """Задача 6: вопросы пользователю после вердикта судьи."""
    if verdict.decision not in (
        JudgeDecision.CLARIFY,
        JudgeDecision.REJECT_PARTIAL_COVERAGE,
        JudgeDecision.REJECT_INSUFFICIENT_KB,
    ):
        return ClarificationResult(questions=[], rationale="Уточнения не требуются.")

    questions: list[str] = []

    if ReasonCode.AMBIGUOUS_COUNTERPARTY in verdict.reason_codes:
        questions.append(
            "Уточните, пожалуйста: поставщик — ООО «Альфа-Трейд» или ООО «Альфа-Трейдинг»?"
        )

    if ReasonCode.MISSING_CONTRACT_REFERENCE in verdict.reason_codes:
        questions.append("Подскажите номер договора, чтобы проверить условия.")

    for amb in verdict.analysis.ambiguities:
        if amb not in questions and "?" not in amb:
            questions.append(amb if amb.endswith("?") else amb + " Уточните, пожалуйста.")

    if verdict.decision == JudgeDecision.REJECT_PARTIAL_COVERAGE:
        if ReasonCode.PARTIAL_LEGAL_BASIS in verdict.reason_codes:
            questions.append(
                "Какие категории персональных данных обрабатываются и на каком "
                "законном основании (трудовой договор, исполнение договора и т.д.)?"
            )
            questions.append(
                "Нужен ли срочный ответ сотруднику или достаточно подготовить "
                "заключение юриста с учётом полной редакции 152-ФЗ?"
            )

    if (
        verdict.decision == JudgeDecision.REJECT_INSUFFICIENT_KB
        and ReasonCode.TOPIC_NOT_IN_KB in verdict.reason_codes
    ):
        questions.append(
            "Есть ли внутренний регламент или приказ банка по этой теме, "
            "который можно добавить в базу знаний?"
        )

    # Уникальные вопросы, максимум 3
    seen: set[str] = set()
    unique: list[str] = []
    for q in questions:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q.strip())
    unique = unique[:3]

    rationale = "Уточнение снимает неоднозначность или компенсирует пробел в базе знаний."
    if verdict.analysis.kb_gaps:
        rationale = verdict.analysis.kb_gaps[0]

    return ClarificationResult(questions=unique, rationale=rationale)


def load_clarify_system_prompt() -> str:
    from pathlib import Path

    path = Path(__file__).parent / "prompts" / "clarify_system.md"
    return path.read_text(encoding="utf-8")
