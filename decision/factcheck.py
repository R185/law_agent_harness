"""Draft-check models and the fact_check handle.

Importing this module does not open the network and does not read an API key.
The classifier, when enabled, is imported inside ``fact_check``.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts import CitationCheck, RetrievalResult
from decision.config import FailureReason, LayerStatus, ModelMode


class ClaimKind(str, Enum):
    """Kind of one phrase in the draft."""

    number = "number"
    date = "date"
    period = "period"
    document = "document"
    clause = "clause"
    party = "party"
    rule = "rule"
    conclusion = "conclusion"


class Claim(BaseModel):
    """One assertion from the draft.

    ``quote`` is the exact draft phrase. ``subject`` says what a number
    or a date is about. ``value`` is the extracted value. ``doc_id`` and
    ``clause`` are set when the draft names that support.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    kind: ClaimKind
    quote: str
    subject: str | None = None
    value: str | None = None
    doc_id: str | None = None
    clause: str | None = None


class FindingCode(str, Enum):
    """Closed defect codes. Members are not aliases of each other."""

    fabricated_citation = "fabricated_citation"
    citation_not_in_context = "citation_not_in_context"
    citation_check_missing = "citation_check_missing"
    wrong_number = "wrong_number"
    wrong_date = "wrong_date"
    wrong_period = "wrong_period"
    wrong_document = "wrong_document"
    wrong_clause = "wrong_clause"
    wrong_party = "wrong_party"
    wrong_edition = "wrong_edition"
    repealed_doc = "repealed_doc"
    contradicts_evidence = "contradicts_evidence"
    unsupported_claim = "unsupported_claim"
    missing_citation = "missing_citation"
    unverifiable = "unverifiable"


class Severity(str, Enum):
    """How serious one finding is."""

    low = "low"
    medium = "medium"
    high = "high"


class FactFinding(BaseModel):
    """One defect in the draft.

    ``hint`` tells the orchestrator what to correct. It is not
    lawyer-facing text. ``expected`` is the value from the materials.
    ``actual`` is the value from the draft. ``evidence_ids`` lists the
    support when there is any.
    """

    model_config = ConfigDict(extra="forbid")

    code: FindingCode
    severity: Severity
    claim_id: str
    claim: str
    quote: str
    evidence_ids: list[str] = Field(default_factory=list)
    expected: str | None = None
    actual: str | None = None
    fixable: bool
    hint: str


class Action(str, Enum):
    """What the orchestrator should do with this draft."""

    accept = "accept"
    rewrite = "rewrite"
    refuse = "refuse"


class FactCheckInput(BaseModel):
    """The draft and the retrieval the check reads.

    ``answer`` is the draft unchanged. ``citation_check`` null means the
    search passed no check; that is not an empty successful check.
    ``claims`` null means no marked list was passed. ``event_date`` null
    and ``check_rules`` null are stored as null: this model does not
    copy ``retrieval.signals.event_date``, does not read
    ``DecisionConfig``, and does not substitute the run date.

    A gold label and a flag that a rewrite was already used are not fields.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    retrieval: RetrievalResult
    citation_check: CitationCheck | None = None
    claims: list[Claim] | None = None
    event_date: str | None = None
    check_rules: bool | None = None


class FactCheckVerdict(BaseModel):
    """Result of one draft check.

    The verdict has no coverage decision and no field for a refusal text
    or a rewritten draft. ``checked_claims`` and ``unverified_claims``
    hold claim ids. One claim id is never in both lists. ``failure_reason``
    is null when ``layer_status`` is ``ok`` and is set otherwise.
    """

    model_config = ConfigDict(extra="forbid")

    suggested_action: Action
    findings: list[FactFinding] = Field(default_factory=list)
    checked_claims: list[str] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    layer_status: LayerStatus
    failure_reason: FailureReason | None
    model_mode: ModelMode

    @model_validator(mode="after")
    def _claim_lists_are_disjoint(self) -> FactCheckVerdict:
        shared = sorted(
            set(self.checked_claims).intersection(self.unverified_claims)
        )
        if shared:
            listed = ", ".join(repr(claim_id) for claim_id in shared)
            raise ValueError(
                "one claim id cannot be in both checked_claims and "
                f"unverified_claims: {listed}"
            )
        return self

    @model_validator(mode="after")
    def _failure_reason_matches_status(self) -> FactCheckVerdict:
        if self.layer_status == LayerStatus.ok:
            if self.failure_reason is not None:
                raise ValueError(
                    "failure_reason must be None when layer_status is ok"
                )
        elif self.failure_reason is None:
            raise ValueError(
                "failure_reason is required when layer_status is not ok"
            )
        return self


_EDITION = re.compile(r"edition-\d+", re.IGNORECASE)
_LOGGER = logging.getLogger("decision.factcheck")
_HINTS = {
    FindingCode.fabricated_citation: "Drop the reference that is not in the materials.",
    FindingCode.citation_not_in_context: "Point the citation at a document in the materials.",
    FindingCode.citation_check_missing: "Pass the citation check before another review.",
    FindingCode.wrong_number: "Use the number from the fragment.",
    FindingCode.wrong_date: "Use the date from the fragment.",
    FindingCode.wrong_period: "Use the period from the fragment.",
    FindingCode.wrong_document: "Use the document id from the materials.",
    FindingCode.wrong_clause: "Use the clause id from the materials.",
    FindingCode.wrong_party: "Use the party named in the materials.",
    FindingCode.wrong_edition: "Use the edition named in the materials.",
    FindingCode.repealed_doc: "Use a document that is still in force.",
    FindingCode.contradicts_evidence: "The fragment states the opposite.",
    FindingCode.unsupported_claim: "The fragments do not mention this subject.",
    FindingCode.missing_citation: "Attach the fragment id on the citation.",
    FindingCode.unverifiable: "The materials do not confirm this phrase.",
}


def _failed(config, reason: FailureReason) -> FactCheckVerdict:
    _LOGGER.warning("%s", reason.value)
    return FactCheckVerdict(
        suggested_action=Action.refuse,
        findings=[],
        checked_claims=[],
        unverified_claims=[],
        layer_status=LayerStatus.failed,
        failure_reason=reason,
        model_mode=config.model_mode,
    )


def _parts(retrieval) -> list[str]:
    parts = [item.text for item in retrieval.evidence]
    for doc in retrieval.documents:
        status = doc.status.value if hasattr(doc.status, "value") else str(doc.status)
        parts.extend(
            [
                doc.doc_id,
                doc.title,
                doc.doc_number,
                doc.doc_type,
                doc.edition_note,
                status,
                doc.effective_from or "",
                doc.effective_to or "",
            ]
        )
        parts.extend(doc.parties)
        parts.extend(doc.matched_clauses)
        parts.extend(doc.why)
    return [part for part in parts if part]


def _grounded(value: str | None, parts: list[str]) -> bool:
    return bool(value) and any(value in part for part in parts)


def _finding(
    code: FindingCode,
    severity: Severity,
    claim_id: str,
    claim: str,
    quote: str,
    *,
    expected: str | None = None,
    actual: str | None = None,
    fixable: bool,
    evidence_ids: list[str] | None = None,
) -> FactFinding:
    return FactFinding(
        code=code,
        severity=severity,
        claim_id=claim_id,
        claim=claim,
        quote=quote,
        evidence_ids=list(evidence_ids or []),
        expected=expected,
        actual=actual,
        fixable=fixable,
        hint=_HINTS[code],
    )


def _evidence_ids(check) -> list[str]:
    if check is None:
        return []
    found: list[str] = []
    for citation in check.citations:
        found.extend(citation.evidence_ids)
    return found


def _flat(text: str) -> str:
    return " ".join(text.casefold().split())


def _contradicts(quote: str, texts: list[str]) -> bool:
    folded = _flat(quote).rstrip(".")
    if not folded:
        return False
    if " not " not in f" {folded} ":
        for text in texts:
            body = _flat(text)
            if " not " not in body:
                continue
            positive = body.replace(" not ", " ")
            if folded in positive and folded not in body:
                return True
    for text in texts:
        body = _flat(text)
        for marker in (" не ", " не\u00a0"):
            if marker not in body:
                continue
            if marker.strip() in folded:
                continue
            positive = body.replace(marker, " ")
            if folded in positive and folded not in body:
                return True
    return False


def _subject_missing(subject: str | None, texts: list[str]) -> bool:
    if not subject or not subject.strip():
        return False
    folded = subject.casefold()
    blob = "\n".join(texts).casefold()
    return folded not in blob


def _citation_binds_claim(claim: Claim, citation) -> bool:
    if claim.doc_id and claim.doc_id != citation.doc_id:
        return False
    if claim.clause and claim.clause != citation.clause:
        return False
    if claim.doc_id or claim.clause:
        return True
    if claim.quote and claim.quote in citation.raw:
        return True
    return claim.clause is None and claim.doc_id is None


def _content_numbers(retrieval, subject: str | None):
    from decision.premise import _numbers

    primary = []
    rest = []
    folded = (subject or "").casefold()
    for item in retrieval.evidence:
        found = _numbers(item.text)
        if folded and folded in item.text.casefold():
            primary.extend(found)
        else:
            rest.extend(found)
    for doc in retrieval.documents:
        for part in (doc.title, doc.edition_note, " ".join(doc.why)):
            rest.extend(_numbers(part))
    return primary or rest


def _content_dates(retrieval, subject: str | None):
    from decision.premise import _dates

    primary = []
    rest = []
    folded = (subject or "").casefold()
    for item in retrieval.evidence:
        found = _dates(item.text)
        if folded and folded in item.text.casefold():
            primary.extend(found)
        else:
            rest.extend(found)
    for doc in retrieval.documents:
        for part in (doc.effective_from or "", doc.effective_to or "", doc.edition_note):
            rest.extend(_dates(part))
    return primary or rest


def _party_names(retrieval) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for doc in retrieval.documents:
        for name in doc.parties:
            key = name.casefold()
            if name and key not in seen:
                seen.add(key)
                names.append(name)
    for group in retrieval.ambiguity:
        for candidate in group.candidates:
            key = candidate.label.casefold()
            if candidate.label and key not in seen:
                seen.add(key)
                names.append(candidate.label)
    return names


def _clause_ids(retrieval) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for doc in retrieval.documents:
        for clause in doc.matched_clauses:
            if clause and clause not in seen:
                seen.add(clause)
                found.append(clause)
    for item in retrieval.evidence:
        for clause in item.clause_ids:
            if clause and clause not in seen:
                seen.add(clause)
                found.append(clause)
    return found


def _unknown_tokens(check) -> set[str]:
    if check is None:
        return set()
    return set(check.unknown_docs) | set(check.unknown_clauses)


def _claim_is_unknown(claim: Claim, banned: set[str]) -> bool:
    for token in (claim.value, claim.doc_id, claim.clause, claim.quote, claim.text):
        if token and token in banned:
            return True
    return False


def _quote_confirmed(quote: str, parts: list[str]) -> bool:
    folded = quote.casefold().strip()
    if len(folded) < 2:
        return False
    return any(folded in part.casefold() for part in parts)


def _replacement_doc(retrieval, parts: list[str], banned: str) -> str | None:
    for doc in retrieval.documents:
        if doc.doc_id and doc.doc_id != banned and _grounded(doc.doc_id, parts):
            if doc.status.value != "repealed":
                return doc.doc_id
    for doc in retrieval.documents:
        if doc.doc_id and doc.doc_id != banned and _grounded(doc.doc_id, parts):
            return doc.doc_id
    return None


def _active_doc_id(retrieval, parts: list[str], repealed_id: str) -> str | None:
    for doc in retrieval.documents:
        if doc.doc_id == repealed_id:
            continue
        if doc.status.value == "active" and _grounded(doc.doc_id, parts):
            return doc.doc_id
    return None


def _action(findings: list[FactFinding]) -> Action:
    if any(item.severity == Severity.high and not item.fixable for item in findings):
        return Action.refuse
    if any(item.severity != Severity.low for item in findings):
        return Action.rewrite
    return Action.accept


def _place(claim_id: str, bucket: str | None, checked: list[str], unverified: list[str]) -> None:
    if bucket == "checked" and claim_id not in checked:
        checked.append(claim_id)
    elif bucket == "unverified" and claim_id not in unverified and claim_id not in checked:
        unverified.append(claim_id)


def _classify_claim(claim: Claim, request: FactCheckInput, parts: list[str], rules_on: bool, ids: list[str]):
    """Return findings for one claim and which list receives its id."""

    from decision.premise import _date_key

    retrieval = request.retrieval
    texts = [item.text for item in retrieval.evidence]
    blob = "\n".join(texts).casefold()
    if claim.kind in {ClaimKind.rule, ClaimKind.conclusion}:
        if rules_on and _quote_confirmed(claim.quote, parts):
            return [], "checked"
        return [
            _finding(
                FindingCode.unverifiable,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    if _contradicts(claim.quote, texts):
        return [
            _finding(
                FindingCode.contradicts_evidence,
                Severity.high,
                claim.claim_id,
                claim.text,
                claim.quote,
                actual=claim.value,
                fixable=False,
                evidence_ids=ids,
            )
        ], None
    if (
        _subject_missing(claim.subject, texts)
        and claim.kind
        in {ClaimKind.number, ClaimKind.date, ClaimKind.period}
    ):
        return [
            _finding(
                FindingCode.unsupported_claim,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    subject = claim.subject or ""
    if claim.kind == ClaimKind.number and subject and subject.casefold() not in blob:
        return [
            _finding(
                FindingCode.unsupported_claim,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    if claim.kind == ClaimKind.number and not (claim.value and claim.value.strip()):
        return [
            _finding(
                FindingCode.unverifiable,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    if claim.kind == ClaimKind.number:
        key = claim.value.replace(" ", "") if claim.value else ""
        numbers = _content_numbers(retrieval, claim.subject)
        if any(item.key == key for item in numbers):
            return _supported_citation(claim, request, parts, ids), "checked"
        if numbers:
            surface = numbers[0].surface
            expected = surface if _grounded(surface, parts) else None
            return [
                _finding(
                    FindingCode.wrong_number,
                    Severity.high,
                    claim.claim_id,
                    claim.text,
                    claim.quote,
                    expected=expected,
                    actual=claim.value,
                    fixable=expected is not None,
                    evidence_ids=ids,
                )
            ], None
        return [
            _finding(
                FindingCode.unverifiable,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    if claim.kind == ClaimKind.date:
        if _subject_missing(claim.subject, texts):
            return [
                _finding(
                    FindingCode.unsupported_claim,
                    Severity.low,
                    claim.claim_id,
                    claim.text,
                    claim.quote,
                    fixable=False,
                    evidence_ids=ids,
                )
            ], "unverified"
        key = _date_key(claim.value) if claim.value else None
        dates = _content_dates(retrieval, claim.subject)
        if key and any(item.key == key for item in dates):
            return _supported_citation(claim, request, parts, ids), "checked"
        if dates:
            surface = dates[0].surface
            expected = surface if _grounded(surface, parts) else None
            return [
                _finding(
                    FindingCode.wrong_date,
                    Severity.high,
                    claim.claim_id,
                    claim.text,
                    claim.quote,
                    expected=expected,
                    actual=claim.value,
                    fixable=expected is not None,
                    evidence_ids=ids,
                )
            ], None
        return [
            _finding(
                FindingCode.unverifiable,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    if claim.kind == ClaimKind.period:
        if _subject_missing(claim.subject, texts):
            return [
                _finding(
                    FindingCode.unsupported_claim,
                    Severity.low,
                    claim.claim_id,
                    claim.text,
                    claim.quote,
                    fixable=False,
                    evidence_ids=ids,
                )
            ], "unverified"
        dates = _content_dates(retrieval, claim.subject)
        if claim.value and dates and any(item.surface in claim.value for item in dates):
            citation_findings = _supported_citation(claim, request, parts, ids)
            if citation_findings:
                return citation_findings, None
            return [], "checked"
        if dates:
            surface = dates[0].surface
            expected = surface if _grounded(surface, parts) else None
            return [
                _finding(
                    FindingCode.wrong_period,
                    Severity.high,
                    claim.claim_id,
                    claim.text,
                    claim.quote,
                    expected=expected,
                    actual=claim.value,
                    fixable=expected is not None,
                    evidence_ids=ids,
                )
            ], None
        return [
            _finding(
                FindingCode.unverifiable,
                Severity.low,
                claim.claim_id,
                claim.text,
                claim.quote,
                fixable=False,
                evidence_ids=ids,
            )
        ], "unverified"
    if claim.kind == ClaimKind.document:
        return _classify_document(claim, request, retrieval, parts, ids)
    if claim.kind == ClaimKind.clause:
        return _classify_clause(claim, request, retrieval, parts, ids)
    if claim.kind == ClaimKind.party:
        return _classify_party(claim, request, retrieval, parts, ids)
    return [
        _finding(
            FindingCode.unverifiable,
            Severity.low,
            claim.claim_id,
            claim.text,
            claim.quote,
            fixable=False,
            evidence_ids=ids,
        )
    ], "unverified"


def _supported_citation(claim: Claim, request: FactCheckInput, parts: list[str], ids: list[str]):
    check = request.citation_check
    if check is None:
        return []
    if claim.subject and _subject_missing(claim.subject, [item.text for item in request.retrieval.evidence]):
        return []
    if not (claim.value and claim.value.strip()):
        return []
    if not check.citations:
        return [
            _finding(
                FindingCode.missing_citation,
                Severity.medium,
                claim.claim_id,
                claim.text,
                claim.quote,
                expected=claim.value,
                actual=claim.value,
                fixable=True,
                evidence_ids=ids,
            )
        ]
    bound = [
        citation
        for citation in check.citations
        if _citation_binds_claim(claim, citation)
    ]
    if bound and any(citation.evidence_ids for citation in bound):
        return []
    expected = claim.value if _grounded(claim.value, parts) else None
    if expected is None:
        for part in parts:
            if claim.value and claim.value in part:
                expected = claim.value
                break
    if expected is None:
        return []
    return [
        _finding(
            FindingCode.missing_citation,
            Severity.medium,
            claim.claim_id,
            claim.text,
            claim.quote,
            expected=expected,
            actual=claim.value,
            fixable=True,
            evidence_ids=ids,
        )
    ]


def _classify_document(claim: Claim, request: FactCheckInput, retrieval, parts: list[str], ids: list[str]):
    ident = claim.doc_id or claim.value or ""
    found = None
    for doc in retrieval.documents:
        if ident and ident in {doc.doc_id, doc.doc_number}:
            found = doc
            break
    if found is not None and found.status.value == "repealed":
        replacement = _active_doc_id(retrieval, parts, found.doc_id)
        return [
            _finding(
                FindingCode.repealed_doc,
                Severity.high,
                claim.claim_id,
                claim.text,
                claim.quote,
                expected=replacement,
                actual=ident,
                fixable=replacement is not None,
                evidence_ids=ids,
            )
        ], None
    if found is not None:
        citation_findings = _supported_citation(claim, request, parts, ids)
        if citation_findings:
            return citation_findings, None
        return [], "checked"
    replacement = _replacement_doc(retrieval, parts, ident)
    if replacement:
        return [
            _finding(
                FindingCode.wrong_document,
                Severity.high,
                claim.claim_id,
                claim.text,
                claim.quote,
                expected=replacement,
                actual=ident,
                fixable=True,
                evidence_ids=ids,
            )
        ], None
    return [
        _finding(
            FindingCode.unverifiable,
            Severity.low,
            claim.claim_id,
            claim.text,
            claim.quote,
            fixable=False,
            evidence_ids=ids,
        )
    ], "unverified"


def _classify_clause(claim: Claim, request: FactCheckInput, retrieval, parts: list[str], ids: list[str]):
    ident = claim.value or claim.clause or ""
    clauses = _clause_ids(retrieval)
    if ident and ident in clauses:
        citation_findings = _supported_citation(claim, request, parts, ids)
        if citation_findings:
            return citation_findings, None
        return [], "checked"
    replacement = next((clause for clause in clauses if _grounded(clause, parts)), None)
    if replacement:
        return [
            _finding(
                FindingCode.wrong_clause,
                Severity.high,
                claim.claim_id,
                claim.text,
                claim.quote,
                expected=replacement,
                actual=ident,
                fixable=True,
                evidence_ids=ids,
            )
        ], None
    return [
        _finding(
            FindingCode.unverifiable,
            Severity.low,
            claim.claim_id,
            claim.text,
            claim.quote,
            fixable=False,
            evidence_ids=ids,
        )
    ], "unverified"


def _classify_party(claim: Claim, request: FactCheckInput, retrieval, parts: list[str], ids: list[str]):
    from decision.premise import _GUILLEMET

    stated = (claim.value or "").strip()
    if not stated:
        match = _GUILLEMET.search(claim.quote)
        stated = match.group(1).strip() if match else ""
    names = _party_names(retrieval)
    folded = stated.casefold()
    quote = claim.quote.casefold()
    material_blob = "\n".join(parts).casefold()
    if stated and any(name.casefold() == folded or name.casefold() in quote for name in names):
        citation_findings = _supported_citation(claim, request, parts, ids)
        if citation_findings:
            return citation_findings, None
        return [], "checked"
    if stated and stated.casefold() not in material_blob:
        grounded = [name for name in names if _grounded(name, parts)]
        if not grounded:
            return [
                _finding(
                    FindingCode.unverifiable,
                    Severity.low,
                    claim.claim_id,
                    claim.text,
                    claim.quote,
                    fixable=False,
                    evidence_ids=ids,
                )
            ], "unverified"
    grounded = [name for name in names if _grounded(name, parts)]
    expected = grounded[0] if grounded else None
    return [
        _finding(
            FindingCode.wrong_party,
            Severity.high,
            claim.claim_id,
            claim.text,
            claim.quote,
            expected=expected,
            actual=stated or None,
            fixable=expected is not None,
            evidence_ids=ids,
        )
    ], None


def _edition_finding(answer: str, retrieval, parts: list[str], ids: list[str]):
    applicable = list(retrieval.signals.applicable_editions)
    if not applicable:
        return None
    allowed = {item.casefold() for item in applicable}
    remainder = answer.casefold()
    for item in sorted(applicable, key=len, reverse=True):
        remainder = remainder.replace(item.casefold(), " ")
    wrong = []
    for token in _EDITION.findall(remainder):
        if token.casefold() not in allowed and token not in wrong:
            wrong.append(token)
    if not wrong:
        return None
    expected = next((item for item in applicable if _grounded(item, parts)), None)
    for doc in retrieval.documents:
        if expected:
            break
        if doc.applicable_edition and _grounded(doc.edition_note, parts):
            expected = doc.edition_note
    return _finding(
        FindingCode.wrong_edition,
        Severity.high,
        "c-edition",
        wrong[0],
        wrong[0],
        expected=expected,
        actual=wrong[0],
        fixable=expected is not None,
        evidence_ids=ids,
    )


def fact_check(request, config=None, transport=None):
    """Compare a draft with the retrieval. Does not rewrite it.

    ``transport`` is accepted for the classifier and ignored while the
    model mode is off. A bad threshold or an unreadable contract is
    returned, not raised. The run date is not read.
    """

    from pydantic import ValidationError

    from decision.claims import resolve_claims
    from decision.config import DecisionConfig

    if config is None:
        config = DecisionConfig()
    if config.thresholds_invalid():
        return _failed(config, FailureReason.config_invalid)
    try:
        parsed = (
            request
            if isinstance(request, FactCheckInput)
            else FactCheckInput.model_validate(request)
        )
    except ValidationError:
        return _failed(config, FailureReason.contract_invalid)
    rules_on = (
        parsed.check_rules
        if parsed.check_rules is not None
        else config.check_rules
    )
    parts = _parts(parsed.retrieval)
    ids = _evidence_ids(parsed.citation_check)
    findings: list[FactFinding] = []
    checked: list[str] = []
    unverified: list[str] = []
    check = parsed.citation_check
    if check is None:
        findings.append(
            _finding(
                FindingCode.citation_check_missing,
                Severity.medium,
                "citation-check",
                "",
                "",
                fixable=True,
                evidence_ids=[],
            )
        )
    else:
        for reference in check.unknown_docs:
            findings.append(
                _finding(
                    FindingCode.fabricated_citation,
                    Severity.high,
                    "c-unknown-doc",
                    reference,
                    reference,
                    actual=reference,
                    fixable=False,
                    evidence_ids=ids,
                )
            )
        for reference in check.unknown_clauses:
            findings.append(
                _finding(
                    FindingCode.fabricated_citation,
                    Severity.high,
                    "c-unknown-clause",
                    reference,
                    reference,
                    actual=reference,
                    fixable=False,
                    evidence_ids=ids,
                )
            )
        for reference in check.unsupported:
            replacement = _replacement_doc(parsed.retrieval, parts, reference)
            findings.append(
                _finding(
                    FindingCode.citation_not_in_context,
                    Severity.medium,
                    "c-unsupported",
                    reference,
                    reference,
                    expected=replacement,
                    actual=reference,
                    fixable=replacement is not None,
                    evidence_ids=ids,
                )
            )
    banned = _unknown_tokens(check)
    for claim in resolve_claims(parsed.answer, parsed.claims):
        if _claim_is_unknown(claim, banned):
            continue
        extra, bucket = _classify_claim(claim, parsed, parts, rules_on, ids)
        findings.extend(extra)
        _place(claim.claim_id, bucket, checked, unverified)
    edition = _edition_finding(parsed.answer, parsed.retrieval, parts, ids)
    if edition is not None:
        findings.append(edition)
    verdict = FactCheckVerdict(
        suggested_action=_action(findings),
        findings=findings,
        checked_claims=checked,
        unverified_claims=unverified,
        layer_status=LayerStatus.ok,
        failure_reason=None,
        model_mode=config.model_mode,
    )
    if config.model_mode != ModelMode.on:
        return verdict
    from decision.llm import review_fact_check

    return review_fact_check(verdict, parsed, config, transport)


__all__ = [
    "Action",
    "Claim",
    "ClaimKind",
    "FactCheckInput",
    "FactCheckVerdict",
    "FactFinding",
    "FindingCode",
    "Severity",
    "fact_check",
]
