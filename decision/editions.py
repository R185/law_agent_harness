"""Edition and reference-error analysis.

Dates are parsed from the strings the retrieval already supplied. The run
date is not read.
"""

from __future__ import annotations

from dataclasses import dataclass

from decision.coverage import Gap, ReasonCode


@dataclass(frozen=True)
class EditionAnalysis:
    date_conflict: bool
    ambiguous_edition: bool
    reference_codes: list[ReasonCode]
    gaps: list[Gap]


def _parse(value: str | None):
    if not value:
        return None
    # Imported locally so a test can patch this module's date class.
    from datetime import date

    year_s, month_s, day_s = value.split("-")
    return date(int(year_s), int(month_s), int(day_s))


def analyze_editions(retrieval) -> EditionAnalysis:
    documents = list(retrieval.documents)
    applicable = [document for document in documents if document.applicable_edition]
    edition_ids = list(retrieval.signals.applicable_editions)
    event = _parse(retrieval.signals.event_date) if retrieval.signals.event_date else None

    date_conflict = False
    for document in applicable:
        start = _parse(document.effective_from)
        end = _parse(document.effective_to)
        if start is not None and end is not None and start > end:
            date_conflict = True
            continue
        if event is None:
            continue
        if start is not None and event < start:
            date_conflict = True
        elif end is not None and event > end:
            date_conflict = True

    several = len(edition_ids) > 1 or len(applicable) > 1
    silent = (
        retrieval.signals.event_date is not None
        and bool(documents)
        and not edition_ids
        and not applicable
    )
    ambiguous = several or silent

    codes: list[ReasonCode] = []
    gaps: list[Gap] = []
    saw_missing = False
    saw_other = False
    for error in retrieval.reference_errors:
        if error.kind == "edition_missing" or getattr(error.kind, "value", None) == "edition_missing":
            if not saw_missing:
                codes.append(ReasonCode.edition_missing)
                saw_missing = True
            gaps.append(Gap(code=ReasonCode.edition_missing, detail=error.raw))
        else:
            if not saw_other:
                codes.append(ReasonCode.reference_error)
                saw_other = True
            gaps.append(Gap(code=ReasonCode.reference_error, detail=error.raw))
    return EditionAnalysis(
        date_conflict=date_conflict,
        ambiguous_edition=ambiguous,
        reference_codes=codes,
        gaps=gaps,
    )
