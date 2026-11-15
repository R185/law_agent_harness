"""Ordered coverage gate.

The first matching rule sets the decision. Later rules still add reason
codes and do not reopen ``allow_answer``. The questioner is imported inside
the call so importing this module stays free of that dependency's cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from decision.coverage import (
    Coverage,
    Decision,
    Gap,
    ReasonCode,
    Recommendation,
)
from decision.editions import analyze_editions
from decision.policy import confirmed_clauses, required_clauses
from decision.premise import analyze_premise


@dataclass
class GateResult:
    decision: Decision
    reason_codes: list[ReasonCode] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    coverage: Coverage = field(default_factory=Coverage)
    premise_errors: list = field(default_factory=list)
    missing_fields: list = field(default_factory=list)
    candidates: list = field(default_factory=list)


def _add(codes: list[ReasonCode], code: ReasonCode) -> None:
    if code not in codes:
        codes.append(code)


def apply_gate(request, config) -> GateResult:
    retrieval = request.retrieval
    signals = retrieval.signals
    required = required_clauses(request.question_kind, request.policy)
    confirmed, missing = confirmed_clauses(retrieval, required)
    coverage = Coverage(required=list(required), confirmed=confirmed, missing=missing)
    premise = analyze_premise(request.question, request.dialogue, retrieval)
    editions = analyze_editions(retrieval)

    from decision.questioner import collect

    clarification = collect(
        request.question,
        request.dialogue,
        retrieval,
        config,
    )

    codes: list[ReasonCode] = []
    gaps: list[Gap] = []
    decision: Decision | None = None

    if signals.topic_match is False:
        decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.topic_mismatch)

    if signals.score_kind == "order_only":
        if decision is None:
            decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.sufficiency_scale_unavailable)

    if not retrieval.documents and not retrieval.evidence:
        if decision is None:
            decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.missing_facts)
        gaps.append(Gap(code=ReasonCode.missing_facts, detail="no documents and no evidence"))

    if retrieval.reference_errors or editions.ambiguous_edition:
        if decision is None:
            decision = Decision.reject_insufficient_kb
        for code in editions.reference_codes:
            _add(codes, code)
        gaps.extend(editions.gaps)
        if editions.ambiguous_edition:
            _add(codes, ReasonCode.ambiguous_edition)

    fields: list = []
    chosen: list = []
    if clarification.applies:
        for code in clarification.reason_codes:
            _add(codes, code)
        if decision is None:
            decision = Decision.clarify
        for field in clarification.missing_fields:
            if field not in fields:
                fields.append(field)
        seen_candidates: set[tuple[str, str]] = set()
        for item in chosen:
            seen_candidates.add((item.doc_id, item.label))
        for item in clarification.candidates:
            key = (item.doc_id, item.label)
            if key not in seen_candidates:
                seen_candidates.add(key)
                chosen.append(item)

    if missing:
        if decision is None:
            decision = Decision.reject_partial_coverage
        _add(codes, ReasonCode.partial_clause_coverage)
        for clause_id in missing:
            gaps.append(Gap(code=ReasonCode.partial_clause_coverage, detail=clause_id))

    unresolved = retrieval.unresolved
    if (
        unresolved.contract_numbers
        or unresolved.party_names
        or unresolved.law_articles
        or unresolved.clauses
    ):
        if decision is None:
            decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.named_not_found)

    if signals.low_confidence:
        if decision is None:
            decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.low_confidence)

    kind = signals.score_kind
    if kind == "semantic":
        threshold = config.min_relevance_semantic
    elif kind == "lexical":
        threshold = config.min_relevance_lexical
    else:
        threshold = None
    if (
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and signals.max_relevance < threshold
    ):
        if decision is None:
            decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.insufficient_relevance)

    if editions.date_conflict:
        if decision is None:
            decision = Decision.reject_insufficient_kb
        _add(codes, ReasonCode.date_conflict)

    has_materials = bool(retrieval.documents or retrieval.evidence)
    if premise.errors and has_materials:
        _add(codes, ReasonCode.premise_error)
    if premise.gaps and has_materials:
        _add(codes, ReasonCode.missing_facts)
        gaps.extend(premise.gaps)

    if decision is None:
        decision = Decision.allow_answer

    return GateResult(
        decision=decision,
        reason_codes=codes,
        gaps=gaps,
        coverage=coverage,
        premise_errors=list(premise.errors) if has_materials else [],
        missing_fields=fields,
        candidates=chosen,
    )


def recommendation_from_gate(request, config, gate: GateResult) -> Recommendation:
    retrieval = request.retrieval
    return Recommendation(
        suggested_decision=gate.decision,
        reason_codes=gate.reason_codes,
        premise_errors=gate.premise_errors,
        ambiguities=list(retrieval.ambiguity),
        event_date=retrieval.signals.event_date,
        applicable_editions=list(retrieval.signals.applicable_editions),
        gaps=gate.gaps,
        coverage=gate.coverage,
        missing_fields=gate.missing_fields,
        candidates=gate.candidates,
        trace=list(retrieval.trace),
        layer_status="ok",
        failure_reason=None,
        model_mode=config.model_mode,
    )
