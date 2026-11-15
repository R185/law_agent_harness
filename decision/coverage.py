"""Dialogue and pre-answer recommendation models.

Shapes only. This module does not choose a decision and does not compare
premises. It does not import a question writer or a classifier client, and
it does not read the clock, the network, or the environment.
"""

from __future__ import annotations

from enum import Enum, unique

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts import Ambiguity, Candidate, RetrievalResult
from decision.config import FailureReason, LayerStatus, ModelMode


@unique
class DialogueRole(str, Enum):
    """Speaker of one dialogue turn."""

    user = "user"
    assistant = "assistant"


class DialogueTurn(BaseModel):
    """One earlier turn. ``content`` is the turn text."""

    model_config = ConfigDict(extra="forbid")

    role: DialogueRole
    content: str


@unique
class Decision(str, Enum):
    """The single suggested outcome."""

    allow_answer = "allow_answer"
    reject_insufficient_kb = "reject_insufficient_kb"
    reject_partial_coverage = "reject_partial_coverage"
    clarify = "clarify"


@unique
class ReasonCode(str, Enum):
    """Closed set of analysis codes."""

    topic_mismatch = "topic_mismatch"
    insufficient_relevance = "insufficient_relevance"
    low_confidence = "low_confidence"
    sufficiency_scale_unavailable = "sufficiency_scale_unavailable"
    ambiguous_entity = "ambiguous_entity"
    missing_identifier = "missing_identifier"
    named_not_found = "named_not_found"
    partial_clause_coverage = "partial_clause_coverage"
    premise_error = "premise_error"
    ambiguous_edition = "ambiguous_edition"
    date_conflict = "date_conflict"
    edition_missing = "edition_missing"
    missing_facts = "missing_facts"
    reference_error = "reference_error"


@unique
class MissingField(str, Enum):
    """An identifier the document choice still needs."""

    contract = "contract"
    counterparty = "counterparty"
    period = "period"


class PremiseError(BaseModel):
    """A lawyer value the materials do not support.

    ``statement`` is the lawyer's text. ``evidence_quote`` is a fragment
    phrase, or empty. ``expected`` is the value from the materials, or empty
    when the materials do not contain the correct value. ``actual`` is the
    value from the lawyer's words.
    """

    model_config = ConfigDict(extra="forbid")

    statement: str
    evidence_quote: str = ""
    expected: str = ""
    actual: str


class Gap(BaseModel):
    """A named absence.

    ``code`` is the reason that names the gap. ``detail`` says which clause,
    edition, or reference is absent. ``detail`` is not a refusal text.
    """

    model_config = ConfigDict(extra="forbid")

    code: ReasonCode
    detail: str


class Coverage(BaseModel):
    """Clause ids required by policy, confirmed by retrieval, or still missing."""

    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)
    confirmed: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class CoverageRequest(BaseModel):
    """The current question, earlier turns, and one retrieval.

    ``question_kind`` selects a policy list. A ``policy`` mapping replaces
    stored policy data for this call; null leaves that choice to the caller.
    The request has no expected verdict.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    dialogue: list[DialogueTurn] = Field(default_factory=list)
    retrieval: RetrievalResult
    question_kind: str | None = None
    policy: dict[str, list[str]] | None = None


def _require_unique(values: list[object], field_name: str) -> None:
    """Reject a repeated entry. The list is not reordered."""

    seen: set[object] = set()
    for item in values:
        if item in seen:
            raise ValueError(f"{field_name} must not contain duplicates")
        seen.add(item)


class Recommendation(BaseModel):
    """Structured coverage analysis before an answer exists.

    The model has no field for lawyer-facing prose, a prepared clarification,
    or an answer. ``failure_reason`` is null when ``layer_status`` is ok, and
    is set when the status is not ok.
    """

    model_config = ConfigDict(extra="forbid")

    suggested_decision: Decision
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    premise_errors: list[PremiseError] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    event_date: str | None
    applicable_editions: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    missing_fields: list[MissingField] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    layer_status: LayerStatus
    failure_reason: FailureReason | None
    model_mode: ModelMode

    @field_validator("reason_codes")
    @classmethod
    def _reason_codes_unique(cls, value: list[ReasonCode]) -> list[ReasonCode]:
        _require_unique(value, "reason_codes")
        return value

    @field_validator("missing_fields")
    @classmethod
    def _missing_fields_bounded_unique(
        cls, value: list[MissingField]
    ) -> list[MissingField]:
        if len(value) > 3:
            raise ValueError("missing_fields accepts at most three values")
        _require_unique(value, "missing_fields")
        return value

    @model_validator(mode="after")
    def _failure_reason_matches_status(self) -> Recommendation:
        if self.layer_status == LayerStatus.ok:
            if self.failure_reason is not None:
                raise ValueError(
                    "failure_reason must be null when layer_status is ok"
                )
        elif self.failure_reason is None:
            raise ValueError(
                "failure_reason is required when layer_status is not ok"
            )
        return self


def _failed(config, reason) -> Recommendation:
    import logging

    logging.getLogger("decision.coverage").warning("%s", reason.value)
    return Recommendation(
        suggested_decision=Decision.reject_insufficient_kb,
        reason_codes=[],
        premise_errors=[],
        ambiguities=[],
        event_date=None,
        applicable_editions=[],
        gaps=[],
        coverage=Coverage(),
        missing_fields=[],
        candidates=[],
        trace=[],
        layer_status=LayerStatus.failed,
        failure_reason=reason,
        model_mode=config.model_mode,
    )


def evaluate(request, config=None, transport=None):
    """Recommend one decision from a retrieval. Does not write lawyer text.

    ``transport`` is accepted for the classifier and ignored while the model
    mode is off. A bad threshold or an unreadable contract is returned, not
    raised.
    """

    from pydantic import ValidationError

    from decision.config import DecisionConfig
    from decision.gates import apply_gate, recommendation_from_gate

    if config is None:
        config = DecisionConfig()
    if config.thresholds_invalid():
        return _failed(config, FailureReason.config_invalid)
    try:
        parsed = (
            request
            if isinstance(request, CoverageRequest)
            else CoverageRequest.model_validate(request)
        )
    except ValidationError:
        return _failed(config, FailureReason.contract_invalid)
    gate = apply_gate(parsed, config)
    result = recommendation_from_gate(parsed, config, gate)
    if config.model_mode != ModelMode.on:
        return result
    from decision.llm import review_coverage

    return review_coverage(result, parsed, config, transport)


__all__ = [
    "DialogueRole",
    "DialogueTurn",
    "Decision",
    "ReasonCode",
    "MissingField",
    "PremiseError",
    "Gap",
    "Coverage",
    "CoverageRequest",
    "Recommendation",
    "evaluate",
]
