from __future__ import annotations

from datetime import date

from harness.insufficient_info.config import DEFAULT_JUDGE_CONFIG, JudgeConfig
from harness.insufficient_info.heuristics import (
    detect_ambiguous_alfa,
    detect_maternity_gap,
    detect_missing_contract,
    detect_partial_pd_revoke,
    detect_premise_errors,
    retrieval_insufficient,
    topic_signals_allow,
)
from harness.types import (
    JudgeDecision,
    JudgeInput,
    JudgeVerdict,
    ReasonCode,
    SituationAnalysis,
)

_REJECT_INSUFFICIENT_TEMPLATE = (
    "{gap} По памяти отвечать не буду: нормы могли измениться. "
    "Для ответа обратитесь в профильный подраздел или к юристу."
)

_REJECT_PARTIAL_TEMPLATE = (
    "{gap} Из имеющихся фрагментов нельзя сделать однозначный вывод для вашей ситуации. "
    "{temporal}"
)


def evaluate_insufficient_info(
    inp: JudgeInput,
    config: JudgeConfig | None = None,
) -> JudgeVerdict:
    """Судья недостатка информации (задача 5): reject / clarify / allow."""
    cfg = config or DEFAULT_JUDGE_CONFIG
    analysis = SituationAnalysis(as_of_date=date.today())
    reason_codes: list[ReasonCode] = []

    ambiguities = detect_ambiguous_alfa(inp)
    if ambiguities:
        analysis.ambiguities = ambiguities
        return JudgeVerdict(
            decision=JudgeDecision.CLARIFY,
            reason_codes=[ReasonCode.AMBIGUOUS_COUNTERPARTY],
            analysis=analysis,
        )

    missing = detect_missing_contract(inp)
    if missing:
        analysis.ambiguities = missing
        return JudgeVerdict(
            decision=JudgeDecision.CLARIFY,
            reason_codes=[ReasonCode.MISSING_CONTRACT_REFERENCE],
            analysis=analysis,
        )

    maternity = detect_maternity_gap(inp)
    if maternity:
        analysis.kb_gaps.extend(maternity.kb_gaps)
        gap = " ".join(maternity.kb_gaps)
        return JudgeVerdict(
            decision=JudgeDecision.REJECT_INSUFFICIENT_KB,
            reason_codes=[ReasonCode.TOPIC_NOT_IN_KB, ReasonCode.LOW_RETRIEVAL_SCORE],
            analysis=analysis,
            user_facing_text=_REJECT_INSUFFICIENT_TEMPLATE.format(gap=gap),
        )

    partial = detect_partial_pd_revoke(inp)
    if partial:
        analysis.kb_gaps.extend(partial.kb_gaps)
        analysis.temporal_notes.extend(partial.temporal_notes)
        temporal = " ".join(partial.temporal_notes)
        gap = " ".join(partial.kb_gaps)
        return JudgeVerdict(
            decision=JudgeDecision.REJECT_PARTIAL_COVERAGE,
            reason_codes=[ReasonCode.PARTIAL_LEGAL_BASIS],
            analysis=analysis,
            user_facing_text=_REJECT_PARTIAL_TEMPLATE.format(gap=gap, temporal=temporal),
        )

    if retrieval_insufficient(inp, cfg):
        if not inp.chunks:
            reason_codes.append(ReasonCode.NO_CHUNKS)
        else:
            reason_codes.append(ReasonCode.LOW_RETRIEVAL_SCORE)
        analysis.kb_gaps.append(
            "Недостаточно релевантных фрагментов в базе знаний для ответа по документам."
        )
        return JudgeVerdict(
            decision=JudgeDecision.REJECT_INSUFFICIENT_KB,
            reason_codes=reason_codes,
            analysis=analysis,
            user_facing_text=_REJECT_INSUFFICIENT_TEMPLATE.format(
                gap=analysis.kb_gaps[0]
            ),
        )

    premise_errors = detect_premise_errors(inp, inp.chunks)
    if premise_errors:
        analysis.premise_errors = premise_errors

    if topic_signals_allow(inp, cfg):
        return JudgeVerdict(
            decision=JudgeDecision.ALLOW_ANSWER,
            reason_codes=[ReasonCode.SUFFICIENT_CONTEXT],
            analysis=analysis,
        )

    return JudgeVerdict(
        decision=JudgeDecision.REJECT_INSUFFICIENT_KB,
        reason_codes=[ReasonCode.LOW_RETRIEVAL_SCORE],
        analysis=analysis,
        user_facing_text=_REJECT_INSUFFICIENT_TEMPLATE.format(
            gap="Недостаточно релевантных источников в базе знаний."
        ),
    )


def load_judge_system_prompt() -> str:
    from pathlib import Path

    path = Path(__file__).parent / "prompts" / "judge_system.md"
    return path.read_text(encoding="utf-8")
