"""Compare lawyer numbers, dates, and parties with the retrieval.

The run date is not read. Assistant turns are ignored. A value the lawyer
stated is a premise error when the materials contain a different value of
the same kind, and a missing-facts gap when they contain none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from decision.coverage import DialogueRole, Gap, PremiseError, ReasonCode

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}")
_NUMBER = re.compile(r"\d{1,3}(?: \d{3})+|\d+")
_GUILLEMET = re.compile(r"«([^»]+)»")


@dataclass(frozen=True)
class _Value:
    surface: str
    key: str


@dataclass(frozen=True)
class PremiseAnalysis:
    errors: list[PremiseError]
    gaps: list[Gap]


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < stop and end > begin for begin, stop in spans)


def _date_key(surface: str) -> str | None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", surface):
        year, month, day = surface.split("-")
    else:
        day, month, year = surface.split(".")
    try:
        year_n, month_n, day_n = int(year), int(month), int(day)
    except ValueError:
        return None
    if not (1 <= month_n <= 12 and 1 <= day_n <= 31):
        return None
    return f"{year_n:04d}-{month_n:02d}-{day_n:02d}"


def _dates(text: str) -> list[_Value]:
    found: list[_Value] = []
    for match in _DATE.finditer(text):
        key = _date_key(match.group(0))
        if key is None:
            continue
        found.append(_Value(match.group(0), key))
    return found


def _numbers(text: str) -> list[_Value]:
    date_spans = [(match.start(), match.end()) for match in _DATE.finditer(text)]
    found: list[_Value] = []
    for match in _NUMBER.finditer(text):
        if _overlaps(match.start(), match.end(), date_spans):
            continue
        surface = match.group(0)
        found.append(_Value(surface, surface.replace(" ", "")))
    return found


def _parties_in_text(text: str) -> list[_Value]:
    found: list[_Value] = []
    for match in _GUILLEMET.finditer(text):
        surface = match.group(1).strip()
        if surface:
            found.append(_Value(surface, surface.casefold()))
    return found


def _material_text(retrieval) -> list[str]:
    parts = [item.text for item in retrieval.evidence]
    for document in retrieval.documents:
        parts.extend(
            [
                document.title,
                document.doc_number,
                document.edition_note,
                " ".join(document.why),
            ]
        )
    return [part for part in parts if part]


def _material_values(retrieval) -> tuple[list[_Value], list[_Value], list[_Value]]:
    dates: list[_Value] = []
    numbers: list[_Value] = []
    for part in _material_text(retrieval):
        dates.extend(_dates(part))
        numbers.extend(_numbers(part))
    for document in retrieval.documents:
        if document.effective_from:
            dates.extend(_dates(document.effective_from))
        if document.effective_to:
            dates.extend(_dates(document.effective_to))
    parties: list[_Value] = []
    seen: set[str] = set()
    for document in retrieval.documents:
        for name in document.parties:
            key = name.casefold()
            if name and key not in seen:
                seen.add(key)
                parties.append(_Value(name, key))
    for group in retrieval.ambiguity:
        for candidate in group.candidates:
            key = candidate.label.casefold()
            if candidate.label and key not in seen:
                seen.add(key)
                parties.append(_Value(candidate.label, key))
    return dates, numbers, parties


def _lawyer_passages(question: str, dialogue) -> list[str]:
    passages = [question]
    for turn in dialogue:
        role = turn.role
        if role == DialogueRole.user or role == "user":
            passages.append(turn.content)
    return passages


def _confirmed(passage: str, materials: list[_Value]) -> set[str]:
    folded = passage.casefold()
    return {item.key for item in materials if item.surface.casefold() in folded}


def _pair(
    stated: list[tuple[str, _Value]],
    materials: list[_Value],
    confirmed: set[str],
) -> tuple[list[tuple[str, _Value, _Value | None]], list[tuple[str, _Value]]]:
    """Return mismatches ``(statement, actual, expected)`` and absences."""

    material_keys = {item.key for item in materials}
    mismatches: list[tuple[str, _Value, _Value | None]] = []
    absences: list[tuple[str, _Value]] = []
    seen: set[str] = set()
    for statement, value in stated:
        if value.key in seen or value.key in confirmed or value.key in material_keys:
            seen.add(value.key)
            continue
        seen.add(value.key)
        if materials:
            mismatches.append((statement, value, materials[0]))
        else:
            absences.append((statement, value))
    return mismatches, absences


def analyze_premise(question: str, dialogue, retrieval) -> PremiseAnalysis:
    material_dates, material_numbers, material_parties = _material_values(retrieval)
    passages = _lawyer_passages(question, dialogue)
    confirmed_parties: set[str] = set()
    stated_dates: list[tuple[str, _Value]] = []
    stated_numbers: list[tuple[str, _Value]] = []
    stated_parties: list[tuple[str, _Value]] = []
    for passage in passages:
        confirmed_parties.update(_confirmed(passage, material_parties))
        for value in _dates(passage):
            stated_dates.append((passage, value))
        for value in _numbers(passage):
            stated_numbers.append((passage, value))
        for value in _parties_in_text(passage):
            stated_parties.append((passage, value))

    errors: list[PremiseError] = []
    gaps: list[Gap] = []
    groups = (
        (stated_numbers, material_numbers, set()),
        (stated_dates, material_dates, set()),
        (stated_parties, material_parties, confirmed_parties),
    )
    for stated, materials, confirmed in groups:
        mismatches, absences = _pair(stated, materials, confirmed)
        for statement, actual, expected in mismatches:
            errors.append(
                PremiseError(
                    statement=statement,
                    evidence_quote=expected.surface if expected is not None else "",
                    expected=expected.surface if expected is not None else "",
                    actual=actual.surface,
                )
            )
        for statement, actual in absences:
            errors.append(
                PremiseError(
                    statement=statement,
                    evidence_quote="",
                    expected="",
                    actual=actual.surface,
                )
            )
            gaps.append(Gap(code=ReasonCode.missing_facts, detail=actual.surface))
    return PremiseAnalysis(errors=errors, gaps=gaps)
