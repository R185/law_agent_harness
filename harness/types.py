from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JudgeDecision(str, Enum):
    ALLOW_ANSWER = "allow_answer"
    REJECT_INSUFFICIENT_KB = "reject_insufficient_kb"
    REJECT_PARTIAL_COVERAGE = "reject_partial_coverage"
    CLARIFY = "clarify"


class ReasonCode(str, Enum):
    NO_CHUNKS = "no_chunks"
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    TOPIC_NOT_IN_KB = "topic_not_in_kb"
    AMBIGUOUS_COUNTERPARTY = "ambiguous_counterparty"
    MISSING_CONTRACT_REFERENCE = "missing_contract_reference"
    PARTIAL_LEGAL_BASIS = "partial_legal_basis"
    SUFFICIENT_CONTEXT = "sufficient_context"


class RetrievedChunk(BaseModel):
    doc_id: str
    title: str = ""
    excerpt: str
    score: float = Field(ge=0.0, le=1.0)


class DialogueTurn(BaseModel):
    role: str
    content: str


class SituationAnalysis(BaseModel):
    """Разбор ситуации для юриста (дата, ошибки, многозначность, актуальность)."""

    as_of_date: date | None = None
    premise_errors: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    temporal_notes: list[str] = Field(default_factory=list)
    kb_gaps: list[str] = Field(default_factory=list)


class JudgeInput(BaseModel):
    question: str
    dialogue: list[DialogueTurn] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    rag_max_score: float | None = None
    rag_mean_score: float | None = None


class JudgeVerdict(BaseModel):
    decision: JudgeDecision
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    analysis: SituationAnalysis = Field(default_factory=SituationAnalysis)
    user_facing_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClarificationResult(BaseModel):
    questions: list[str]
    rationale: str = ""
