"""Claim list for one draft.

Passed claims are kept as given and are not relabeled. A missing list is
filled from the draft for numbers, dates, parties, and periods. Rules and
conclusions are never assigned here.
"""

from __future__ import annotations

import re

from decision.factcheck import Claim, ClaimKind
from decision.premise import _GUILLEMET, _dates, _numbers

_EDITION = re.compile(r"edition-\d+", re.IGNORECASE)
_DOC = re.compile(r"\bdoc-[0-9A-Za-z-]+\b")
_CLAUSE = re.compile(
    r"(?iu)\b(?:clause|статья|ст\.?)\s*([0-9A-Za-zА-Яа-яЁё-]+)"
)
_PERIOD = re.compile(
    r"\d{4}-\d{2}-\d{2}\s*(?:—|-|по|to)\s*\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)


def _without(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(" ", text)


def _one(
    kind: ClaimKind,
    index: int,
    surface: str,
    *,
    quote: str | None = None,
    value: str | None = None,
    doc_id: str | None = None,
    clause: str | None = None,
) -> Claim:
    shown = surface if quote is None else quote
    return Claim(
        claim_id=f"c-{kind.value}-{index}",
        text=shown,
        kind=kind,
        quote=shown,
        value=surface if value is None else value,
        doc_id=doc_id,
        clause=clause,
    )


def extract_claims(answer: str) -> list[Claim]:
    """Pull comparable phrases out of an unmarked draft.

    Rules and conclusions are not assigned. Identifiers are stable for one
    draft because they follow the phrase order.
    """

    claims: list[Claim] = []
    for index, match in enumerate(_PERIOD.finditer(answer), start=1):
        claims.append(_one(ClaimKind.period, index, match.group(0)))
    plain = _without(_without(answer, _EDITION), _DOC)
    for index, value in enumerate(_dates(plain), start=1):
        claims.append(_one(ClaimKind.date, index, value.surface))
    for index, value in enumerate(_numbers(plain), start=1):
        claims.append(_one(ClaimKind.number, index, value.surface))
    for index, match in enumerate(_DOC.finditer(answer), start=1):
        surface = match.group(0)
        claims.append(_one(ClaimKind.document, index, surface, doc_id=surface))
    for index, match in enumerate(_CLAUSE.finditer(answer), start=1):
        surface = match.group(1)
        claims.append(_one(ClaimKind.clause, index, surface, clause=surface))
    for index, match in enumerate(_GUILLEMET.finditer(answer), start=1):
        surface = match.group(1).strip()
        if not surface:
            continue
        claims.append(
            _one(ClaimKind.party, index, surface, quote=match.group(0))
        )
    return claims


def resolve_claims(answer: str, claims: list[Claim] | None) -> list[Claim]:
    """Use the orchestrator's list, or extract one when it was not passed.

    An empty list is a provided list and is not replaced. Kinds are not
    changed.
    """

    if claims is not None:
        return list(claims)
    return extract_claims(answer)


__all__ = ["extract_claims", "resolve_claims"]
