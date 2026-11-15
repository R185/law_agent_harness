"""Clarification fields for a document the retrieval did not choose.

The stub returns a closed clarification. The clarification story replaces
the function body and keeps this signature. This module does not import
``evaluate`` and does not build a question sentence.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from contracts import Candidate, RetrievalResult
from decision.config import DecisionConfig
from decision.coverage import DialogueRole, DialogueTurn, MissingField, ReasonCode

_CONTRACT_RE = re.compile(
    r"(?iu)(?:договор\w*|контракт\w*)\s+(?:№\s*)?"
    r"([0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё./-]*)"
)
_YEAR_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
_SEPARATOR_RE = re.compile(r"[-/]")


class Clarification(BaseModel):
    """Fields and candidates. There is no question string on this model."""

    model_config = ConfigDict(extra="forbid")

    applies: bool
    missing_fields: list[MissingField] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)


def _closed() -> Clarification:
    return Clarification(
        applies=False,
        missing_fields=[],
        candidates=[],
        reason_codes=[],
    )


def _lawyer_text(question: str, dialogue: list[DialogueTurn]) -> str:
    parts = [question]
    for turn in dialogue:
        if turn.role == DialogueRole.user or turn.role == "user":
            parts.append(turn.content)
    return "\n".join(parts)


def _contract_identifiers(text: str) -> list[str]:
    found: list[str] = []
    for match in _CONTRACT_RE.finditer(text):
        token = match.group(1)
        if re.fullmatch(r"\d+", token):
            continue
        if not (_LETTER_RE.search(token) or _SEPARATOR_RE.search(token)):
            continue
        found.append(token)
    return found


def _years(text: str) -> list[str]:
    return _YEAR_RE.findall(text)


def _distinct_numbers(retrieval: RetrievalResult) -> list[str]:
    seen: list[str] = []
    for document in retrieval.documents:
        number = document.doc_number.strip()
        if number and number not in seen:
            seen.append(number)
    return seen


def _distinct_labels(retrieval: RetrievalResult) -> list[str]:
    seen: list[str] = []
    folded: set[str] = set()
    for group in retrieval.ambiguity:
        for candidate in group.candidates:
            key = candidate.label.casefold()
            if candidate.label and key not in folded:
                folded.add(key)
                seen.append(candidate.label)
    return seen


def _distinct_periods(retrieval: RetrievalResult) -> list[tuple[str | None, str | None]]:
    seen: list[tuple[str | None, str | None]] = []
    for document in retrieval.documents:
        pair = (document.effective_from, document.effective_to)
        if pair not in seen:
            seen.append(pair)
    return seen


def _label_named(text: str, labels: list[str]) -> bool:
    folded = text.casefold()
    return any(label.casefold() in folded for label in labels)


def _period_named(text: str) -> bool:
    return bool(_years(text))


def _candidates(retrieval: RetrievalResult) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    chosen: list[Candidate] = []
    for group in retrieval.ambiguity:
        for candidate in group.candidates:
            key = (candidate.doc_id, candidate.label)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(candidate)
    return chosen


def _candidate_count(retrieval: RetrievalResult) -> int:
    return sum(len(group.candidates) for group in retrieval.ambiguity)


def _leaves_one(text: str, retrieval: RetrievalResult, identifiers: list[str]) -> bool:
    documents = list(retrieval.documents)
    if not documents:
        return False
    folded = text.casefold()
    for identifier in identifiers:
        hits = [
            document
            for document in documents
            if document.doc_number
            and document.doc_number.casefold() == identifier.casefold()
        ]
        if len(hits) == 1:
            return True
    matched_numbers: list = []
    for document in documents:
        number = document.doc_number.strip()
        if number and number.casefold() in folded:
            matched_numbers.append(document)
    if len(matched_numbers) == 1:
        return True
    label_ids: set[str] = set()
    named_label = False
    for group in retrieval.ambiguity:
        for candidate in group.candidates:
            if candidate.label and candidate.label.casefold() in folded:
                named_label = True
                label_ids.add(candidate.doc_id)
    if named_label and len(label_ids) == 1:
        return True
    for year in _years(text):
        hits = []
        for document in documents:
            blob = f"{document.effective_from or ''} {document.effective_to or ''}"
            if year in blob:
                hits.append(document)
        if len(hits) == 1:
            return True
    return False


def collect(
    question: str,
    dialogue: list[DialogueTurn],
    retrieval: RetrievalResult,
    config: DecisionConfig,
) -> Clarification:
    """Return at most three missing fields and the ambiguity candidates."""

    if not retrieval.documents and not retrieval.evidence:
        return _closed()

    text = _lawyer_text(question, dialogue)
    identifiers = _contract_identifiers(text)
    labels = _distinct_labels(retrieval)
    missing: list[MissingField] = []
    if len(_distinct_numbers(retrieval)) > 1 and not identifiers:
        missing.append(MissingField.contract)
    if len(labels) > 1 and not _label_named(text, labels):
        missing.append(MissingField.counterparty)
    if len(_distinct_periods(retrieval)) > 1 and not _period_named(text):
        missing.append(MissingField.period)

    case_one = (
        bool(missing)
        and _candidate_count(retrieval) > 1
        and retrieval.signals.margin_top1_top2 < config.max_clarify_margin
    )
    # One document already left does not open case 2, even with several candidates.
    case_two = (
        len(retrieval.documents) != 1
        and not _leaves_one(text, retrieval, identifiers)
        and (
            len(retrieval.documents) > 1
            or (
                _candidate_count(retrieval) > 1
                and not retrieval.documents
                and retrieval.evidence
            )
        )
    )

    if not case_one and not case_two:
        return _closed()

    codes: list[ReasonCode] = []
    if case_one:
        codes.append(ReasonCode.ambiguous_entity)
    if case_two:
        codes.append(ReasonCode.missing_identifier)
    return Clarification(
        applies=True,
        missing_fields=missing,
        candidates=_candidates(retrieval),
        reason_codes=codes,
    )


__all__ = ["Clarification", "collect"]
