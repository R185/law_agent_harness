"""Draft-check tests for decision.fact_check.

Numbers, dates, parties, and document ids are synthetic. The handle is not
implemented; these tests describe the comparisons it must make.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager

import pytest

import decision
from contracts import Citation, CitationCheck
from decision import (
    Action,
    Claim,
    ClaimKind,
    DecisionConfig,
    FactCheckInput,
    FactCheckVerdict,
    FailureReason,
    FindingCode,
    LayerStatus,
    Severity,
)
from tests.support import citation_check, document, evidence, retrieval, signals

FEE = "100"
DRAFT_FEE = "250"
PARTY = "North Warehouse LLC"
OTHER_PARTY = "Other Party LLC"
EVENT_DATE = "2024-06-01"
DOC_ID = "doc-1"
CLAUSE = "A"
EDITION = "edition-1"
OTHER_EDITION = "edition-9"
RULE_QUOTE = "Storage must stay dry."
CONCLUSION_QUOTE = "The warehouse remains liable."

LAWYER_KEYS = {
    "summary",
    "user_facing_text",
    "refusal",
    "refusal_text",
    "rewritten_answer",
    "rewritten_draft",
    "answer",
    "suggested_decision",
    "decision",
    "coverage",
    "gold_label",
    "rewrite_used",
    "rewrite_budget",
    "event_date",
}

FALSE_PREFIX = "wrong_"


def claim(
    claim_id: str,
    kind: ClaimKind,
    quote: str,
    *,
    subject: str | None = None,
    value: str | None = None,
    doc_id: str | None = None,
    clause: str | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=quote,
        kind=kind,
        quote=quote,
        subject=subject,
        value=value,
        doc_id=doc_id,
        clause=clause,
    )


def cite(
    doc_id: str = DOC_ID,
    clause: str = CLAUSE,
    evidence_ids: list[str] | None = None,
    **overrides: object,
) -> CitationCheck:
    if evidence_ids is None:
        evidence_ids = ["ev-1"]
    payload: dict[str, object] = {
        "citations": [
            Citation(
                raw=f"{doc_id} clause {clause}",
                doc_id=doc_id,
                clause=clause,
                evidence_ids=list(evidence_ids),
            )
        ]
    }
    payload.update(overrides)
    return citation_check(**payload)


def draft(
    answer: str,
    found: object,
    *,
    citation: CitationCheck | None = None,
    claims: list[Claim] | None = None,
    event_date: str | None = None,
    check_rules: bool | None = None,
    question: str = "What fee applies?",
) -> FactCheckInput:
    return FactCheckInput(
        question=question,
        answer=answer,
        retrieval=found,
        citation_check=citation,
        claims=claims,
        event_date=event_date,
        check_rules=check_rules,
    )


def material_parts(found: object) -> list[str]:
    parts: list[str] = [item.text for item in found.evidence]
    for doc in found.documents:
        parts.extend(
            [
                doc.doc_id,
                doc.title,
                doc.doc_number,
                doc.doc_type,
                doc.edition_note,
                doc.status.value,
                doc.effective_from or "",
                doc.effective_to or "",
            ]
        )
        parts.extend(doc.parties)
        parts.extend(doc.matched_clauses)
        parts.extend(doc.why)
    return [part for part in parts if part]


def assert_shape(verdict: FactCheckVerdict) -> None:
    assert verdict.suggested_action in {
        Action.accept,
        Action.rewrite,
        Action.refuse,
    }
    assert set(verdict.checked_claims).isdisjoint(verdict.unverified_claims)
    assert all(isinstance(item, str) for item in verdict.checked_claims)
    assert all(isinstance(item, str) for item in verdict.unverified_claims)
    assert LAWYER_KEYS.isdisjoint(verdict.model_dump())


def assert_readable(verdict: FactCheckVerdict) -> None:
    assert_shape(verdict)
    assert verdict.layer_status == LayerStatus.ok
    assert verdict.failure_reason is None


def assert_fixable_grounded(verdict: FactCheckVerdict, found: object) -> None:
    """fixable is true only when the replacement sits in the materials.

    citation_check_missing is the exception: the defect is the missing check.
    """

    parts = material_parts(found)
    for item in verdict.findings:
        if not item.fixable or item.code == FindingCode.citation_check_missing:
            continue
        assert item.expected, item
        assert any(item.expected in part for part in parts), item.expected


def codes(verdict: FactCheckVerdict) -> set[FindingCode]:
    return {item.code for item in verdict.findings}


def one(verdict: FactCheckVerdict, code: FindingCode):
    matched = [item for item in verdict.findings if item.code == code]
    assert matched, verdict.findings
    return matched[0]


def assert_not_false(item) -> None:
    assert not item.code.value.startswith(FALSE_PREFIX)
    assert item.code != FindingCode.contradicts_evidence
    assert item.code != FindingCode.fabricated_citation
    assert item.code != FindingCode.repealed_doc


def assert_not_gap(verdict: FactCheckVerdict) -> None:
    blob = json.dumps(verdict.model_dump(mode="json"))
    assert "missing_facts" not in blob
    assert "reject_insufficient_kb" not in blob
    dumped = verdict.model_dump()
    assert "suggested_decision" not in dumped
    assert "coverage" not in dumped


def fee_subject_materials() -> object:
    """Subject 'fee' is in the fragment. No digit is in the text or fields."""

    doc = document(
        doc_id="doc-fee",
        doc_number="",
        title="Internal rule",
        parties=[PARTY],
        edition_note="",
        effective_from=None,
        effective_to=None,
        matched_clauses=[CLAUSE],
        applicable_edition=True,
    )
    fragment = evidence(
        evidence_id="ev-fee",
        doc_id="doc-fee",
        text="The monthly fee applies under the schedule.",
        clause_ids=[CLAUSE],
    )
    return retrieval(
        documents=[doc],
        evidence=[fragment],
        signals=signals(event_date=None, applicable_editions=[], entity_match=[]),
        trace=[],
    )


def number_without_value() -> Claim:
    return claim(
        "c-fee",
        ClaimKind.number,
        "The monthly fee applies.",
        subject="fee",
        value=None,
        doc_id="doc-fee",
        clause=CLAUSE,
    )


def matching_materials(fee_in_draft: str) -> tuple[object, str, list[Claim]]:
    fee_quote = f"The monthly fee is {fee_in_draft}."
    date_quote = f"The event date is {EVENT_DATE}."
    doc_quote = f"Document {DOC_ID}"
    clause_quote = f"clause {CLAUSE}"
    evidence_text = " ".join(
        [
            f"The monthly fee is {FEE}.",
            date_quote,
            PARTY,
            doc_quote,
            clause_quote,
            EDITION,
            RULE_QUOTE,
        ]
    )
    answer = " ".join(
        [
            fee_quote,
            date_quote,
            PARTY,
            doc_quote,
            clause_quote,
            EDITION,
            RULE_QUOTE,
            CONCLUSION_QUOTE,
        ]
    )
    found = retrieval(
        documents=[document(edition_note=EDITION)],
        evidence=[evidence(text=evidence_text)],
        signals=signals(event_date=EVENT_DATE, applicable_editions=[EDITION]),
    )
    claims = [
        claim(
            "c-fee",
            ClaimKind.number,
            fee_quote,
            subject="fee",
            value=fee_in_draft,
            doc_id=DOC_ID,
            clause=CLAUSE,
        ),
        claim(
            "c-date",
            ClaimKind.date,
            date_quote,
            subject="event date",
            value=EVENT_DATE,
            doc_id=DOC_ID,
            clause=CLAUSE,
        ),
        claim(
            "c-doc",
            ClaimKind.document,
            doc_quote,
            value=DOC_ID,
            doc_id=DOC_ID,
            clause=CLAUSE,
        ),
        claim(
            "c-clause",
            ClaimKind.clause,
            clause_quote,
            value=CLAUSE,
            doc_id=DOC_ID,
            clause=CLAUSE,
        ),
        claim(
            "c-party",
            ClaimKind.party,
            PARTY,
            value=PARTY,
            doc_id=DOC_ID,
            clause=CLAUSE,
        ),
        claim("c-rule", ClaimKind.rule, RULE_QUOTE),
        claim("c-conclusion", ClaimKind.conclusion, CONCLUSION_QUOTE),
    ]
    return found, answer, claims


def rule_findings(verdict: FactCheckVerdict, claim_id: str) -> list:
    return [item for item in verdict.findings if item.claim_id == claim_id]


def assert_unchecked_statement(verdict: FactCheckVerdict, claim_id: str) -> None:
    items = rule_findings(verdict, claim_id)
    assert items
    assert claim_id in verdict.unverified_claims
    assert claim_id not in verdict.checked_claims
    for item in items:
        assert item.code == FindingCode.unverifiable
        assert item.severity == Severity.low
        assert item.fixable is False
        assert_not_false(item)
        assert item.code != FindingCode.unsupported_claim


@contextmanager
def patched_calendar(year: int, month: int, day: int):
    """Freeze date.today and datetime.now without touching an immutable builtin."""

    import datetime as datetime_module

    original_date = datetime_module.date
    original_datetime = datetime_module.datetime

    class FrozenDate(original_date):
        @classmethod
        def today(cls):
            return cls(year, month, day)

    class FrozenDateTime(original_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(year, month, day)

        @classmethod
        def today(cls):
            return cls(year, month, day)

        @classmethod
        def utcnow(cls):
            return cls(year, month, day)

    replaced: list[tuple[object, str, object]] = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if (
            name != "datetime"
            and not name.startswith("decision")
            and not name.startswith("contracts")
        ):
            continue
        for attr, value in list(vars(module).items()):
            if attr.startswith("__"):
                continue
            if value is original_date:
                replaced.append((module, attr, value))
                setattr(module, attr, FrozenDate)
            elif value is original_datetime:
                replaced.append((module, attr, value))
                setattr(module, attr, FrozenDateTime)
    try:
        yield
    finally:
        for module, attr, value in replaced:
            setattr(module, attr, value)


def test_wrong_number_with_replacement_rewrites() -> None:
    found = retrieval(
        evidence=[evidence(text="The monthly fee is 100.")],
    )
    request = draft(
        "The monthly fee is 250.",
        found,
        citation=cite(),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 250.",
                subject="fee",
                value=DRAFT_FEE,
                doc_id=DOC_ID,
                clause=CLAUSE,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.wrong_number)
    assert item.severity == Severity.high
    assert item.fixable is True
    assert item.expected and FEE in item.expected
    assert verdict.suggested_action == Action.rewrite


@pytest.mark.parametrize("slot", ["unknown_docs", "unknown_clauses"])
def test_unknown_reference_without_replacement_is_fabricated(slot: str) -> None:
    if slot == "unknown_docs":
        reference = "doc-missing"
        found = retrieval(
            documents=[],
            evidence=[],
            signals=signals(),
            trace=[],
        )
        made = claim(
            "c-ref",
            ClaimKind.document,
            reference,
            value=reference,
            doc_id=reference,
        )
        unknown = cite(
            unknown_docs=[reference],
            unknown_clauses=[],
            unsupported=[],
            citations=[],
            valid=False,
        )
    else:
        reference = "clause-Z"
        found = retrieval(
            documents=[
                document(
                    matched_clauses=[],
                    edition_note="",
                    parties=[PARTY],
                )
            ],
            evidence=[evidence(text="The schedule is stored.", clause_ids=[])],
            signals=signals(),
            trace=[],
        )
        made = claim(
            "c-ref",
            ClaimKind.clause,
            reference,
            value=reference,
            clause=reference,
        )
        unknown = cite(
            unknown_docs=[],
            unknown_clauses=[reference],
            unsupported=[],
            citations=[],
            valid=False,
        )
    assert reference not in " ".join(material_parts(found))
    request = draft(f"See {reference}.", found, citation=unknown, claims=[made])
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.fabricated_citation)
    assert item.severity == Severity.high
    assert item.fixable is False
    assert verdict.suggested_action == Action.refuse


def test_missing_citation_check_rewrites_and_repeats() -> None:
    found = retrieval(evidence=[evidence(text="The monthly fee is 100.")])
    request = draft(
        "The monthly fee is 100.",
        found,
        citation=None,
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 100.",
                subject="fee",
                value=FEE,
                doc_id=DOC_ID,
                clause=CLAUSE,
            )
        ],
    )
    assert request.citation_check is None
    first = decision.fact_check(request, config=None)
    second = decision.fact_check(request, config=None)
    for verdict in (first, second):
        assert_readable(verdict)
        assert_fixable_grounded(verdict, found)
        assert_not_gap(verdict)
        item = one(verdict, FindingCode.citation_check_missing)
        assert item.severity == Severity.medium
        assert item.fixable is True
        assert FindingCode.missing_citation not in codes(verdict)
        assert verdict.suggested_action == Action.rewrite
        assert verdict.suggested_action != Action.accept
        assert verdict.suggested_action != Action.refuse
    assert first.suggested_action == second.suggested_action
    assert [item.code for item in first.findings] == [
        item.code for item in second.findings
    ]


def test_number_subject_without_value_is_unverifiable() -> None:
    found = fee_subject_materials()
    request = draft(
        "The monthly fee applies.",
        found,
        citation=cite(doc_id="doc-fee", evidence_ids=["ev-fee"]),
        claims=[number_without_value()],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.unverifiable)
    assert item.severity == Severity.low
    assert item.fixable is False
    assert FindingCode.wrong_number not in codes(verdict)
    assert FindingCode.unsupported_claim not in codes(verdict)
    assert FindingCode.missing_citation not in codes(verdict)
    assert verdict.suggested_action == Action.accept


def test_number_subject_without_value_is_not_missing_citation() -> None:
    found = fee_subject_materials()
    request = draft(
        "The monthly fee applies.",
        found,
        citation=cite(doc_id="doc-fee", evidence_ids=[]),
        claims=[number_without_value()],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.unverifiable)
    assert item.claim_id == "c-fee"
    assert item.severity == Severity.low
    assert item.fixable is False
    assert FindingCode.missing_citation not in codes(verdict)
    assert FindingCode.wrong_number not in codes(verdict)
    assert verdict.suggested_action == Action.accept


def test_matching_claims_accept_unchecked_rule_is_not_false() -> None:
    found, answer, claims = matching_materials(FEE)
    request = draft(
        answer,
        found,
        citation=cite(),
        claims=claims,
        check_rules=None,
    )
    assert request.check_rules is None
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    matched = {"c-fee", "c-date", "c-doc", "c-clause", "c-party"}
    assert matched <= set(verdict.checked_claims)
    assert_unchecked_statement(verdict, "c-rule")
    assert_unchecked_statement(verdict, "c-conclusion")
    assert all(item.severity == Severity.low for item in verdict.findings)
    assert verdict.suggested_action == Action.accept


def test_unchecked_rule_plus_wrong_number_rewrites() -> None:
    found, answer, claims = matching_materials(DRAFT_FEE)
    request = draft(answer, found, citation=cite(), claims=claims, check_rules=False)
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.wrong_number)
    assert item.claim_id == "c-fee"
    assert item.fixable is True
    assert item.severity == Severity.high
    assert item.expected and FEE in item.expected
    assert "c-fee" not in verdict.checked_claims
    assert "c-fee" not in verdict.unverified_claims
    assert_unchecked_statement(verdict, "c-rule")
    assert_unchecked_statement(verdict, "c-conclusion")
    assert {"c-date", "c-doc", "c-clause", "c-party"} <= set(verdict.checked_claims)
    assert verdict.suggested_action == Action.rewrite


def test_wrong_edition_with_replacement_rewrites() -> None:
    found = retrieval(
        documents=[document(edition_note=EDITION, applicable_edition=True)],
        evidence=[evidence(text="The in-force text names edition-1.")],
        signals=signals(event_date=EVENT_DATE, applicable_editions=[EDITION]),
    )
    answer = "The draft relies on edition-9."
    assert OTHER_EDITION not in found.signals.applicable_editions
    request = draft(answer, found, citation=cite(), claims=[])
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.wrong_edition)
    assert item.severity == Severity.high
    assert item.fixable is True
    assert item.expected and EDITION in item.expected
    assert verdict.suggested_action == Action.rewrite


def test_repealed_document_with_replacement_rewrites() -> None:
    repealed = document(
        doc_id="doc-repealed",
        status="repealed",
        applicable_edition=False,
        edition_note="",
        parties=[PARTY],
    )
    current = document(
        doc_id=DOC_ID,
        status="active",
        applicable_edition=True,
        edition_note=EDITION,
        parties=[PARTY],
    )
    found = retrieval(
        documents=[repealed, current],
        evidence=[
            evidence(
                doc_id=DOC_ID,
                text="The current document is doc-1.",
            )
        ],
        signals=signals(),
    )
    request = draft(
        "The draft cites doc-repealed.",
        found,
        citation=cite(doc_id="doc-repealed", evidence_ids=["ev-1"]),
        claims=[
            claim(
                "c-doc",
                ClaimKind.document,
                "doc-repealed",
                value="doc-repealed",
                doc_id="doc-repealed",
                clause=CLAUSE,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.repealed_doc)
    assert item.severity == Severity.high
    assert item.fixable is True
    assert item.expected and DOC_ID in item.expected
    assert verdict.suggested_action == Action.rewrite


def test_wrong_party_with_replacement_rewrites() -> None:
    found = retrieval(
        documents=[document(parties=[PARTY])],
        evidence=[evidence(text="The schedule is stored.")],
        signals=signals(entity_match=[]),
    )
    answer = "The counterparty is «Other Party LLC»."
    request = draft(
        answer,
        found,
        citation=cite(),
        claims=[
            claim(
                "c-party",
                ClaimKind.party,
                "«Other Party LLC»",
                subject="counterparty",
                value=OTHER_PARTY,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.wrong_party)
    assert item.severity == Severity.high
    assert item.fixable is True
    assert item.expected and PARTY in item.expected
    assert verdict.suggested_action == Action.rewrite


def test_wrong_edition_without_replacement_refuses() -> None:
    found = retrieval(
        documents=[
            document(
                edition_note="",
                applicable_edition=False,
                parties=[PARTY],
            )
        ],
        evidence=[evidence(text="The schedule is stored.")],
        signals=signals(event_date=EVENT_DATE, applicable_editions=[EDITION]),
    )
    assert not any(EDITION in part for part in material_parts(found))
    request = draft(
        "The draft relies on edition-9.",
        found,
        citation=cite(),
        claims=[],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.wrong_edition)
    assert item.severity == Severity.high
    assert item.fixable is False
    assert verdict.suggested_action == Action.refuse


def test_repealed_document_without_replacement_refuses() -> None:
    found = retrieval(
        documents=[
            document(
                doc_id="doc-repealed",
                status="repealed",
                applicable_edition=False,
                edition_note="",
                parties=[PARTY],
            )
        ],
        evidence=[
            evidence(
                doc_id="doc-repealed",
                text="The schedule is stored.",
            )
        ],
        signals=signals(),
    )
    assert all(doc.status.value == "repealed" for doc in found.documents)
    request = draft(
        "The draft cites doc-repealed.",
        found,
        citation=cite(doc_id="doc-repealed"),
        claims=[
            claim(
                "c-doc",
                ClaimKind.document,
                "doc-repealed",
                value="doc-repealed",
                doc_id="doc-repealed",
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.repealed_doc)
    assert item.severity == Severity.high
    assert item.fixable is False
    assert verdict.suggested_action == Action.refuse


def test_party_absent_from_materials_is_unverifiable_not_wrong_party() -> None:
    found = retrieval(
        documents=[document(parties=[])],
        evidence=[evidence(text="The schedule is stored.")],
        signals=signals(entity_match=[]),
        ambiguity=[],
    )
    assert found.documents[0].parties == []
    assert not any("LLC" in part or "Party" in part for part in material_parts(found))
    request = draft(
        "The counterparty is «Other Party LLC».",
        found,
        citation=cite(),
        claims=[
            claim(
                "c-party",
                ClaimKind.party,
                "«Other Party LLC»",
                subject="counterparty",
                value=OTHER_PARTY,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.unverifiable)
    assert item.severity == Severity.low
    assert item.fixable is False
    assert FindingCode.wrong_party not in codes(verdict)
    assert verdict.suggested_action == Action.accept


def test_fixable_finding_with_contradiction_refuses() -> None:
    # The fragment negates the shipment quote and gives that claim no replacement.
    text = "The monthly fee is 100. The shipment is not allowed."
    found = retrieval(evidence=[evidence(text=text)])
    request = draft(
        "The monthly fee is 250. The shipment is allowed.",
        found,
        citation=cite(),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 250.",
                subject="fee",
                value=DRAFT_FEE,
                doc_id=DOC_ID,
                clause=CLAUSE,
            ),
            claim(
                "c-ship",
                ClaimKind.number,
                "The shipment is allowed",
                subject="shipment",
                value=None,
            ),
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    wrong = one(verdict, FindingCode.wrong_number)
    assert wrong.fixable is True
    denied = one(verdict, FindingCode.contradicts_evidence)
    assert denied.severity == Severity.high
    assert denied.fixable is False
    assert denied.claim_id == "c-ship"
    assert verdict.suggested_action == Action.refuse


def test_contradiction_refuses_even_when_citation_check_is_missing() -> None:
    found = retrieval(
        evidence=[evidence(text="The shipment is not allowed.")],
    )
    request = draft(
        "The shipment is allowed.",
        found,
        citation=None,
        claims=[
            claim(
                "c-ship",
                ClaimKind.number,
                "The shipment is allowed",
                subject="shipment",
                value=None,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    denied = one(verdict, FindingCode.contradicts_evidence)
    assert denied.severity == Severity.high
    assert denied.fixable is False
    missing = one(verdict, FindingCode.citation_check_missing)
    assert missing.severity == Severity.medium
    assert missing.fixable is True
    assert verdict.suggested_action == Action.refuse


def test_supported_number_with_empty_evidence_ids_is_missing_citation() -> None:
    found = retrieval(evidence=[evidence(text="The monthly fee is 100.")])
    assert FEE in found.evidence[0].text
    request = draft(
        "The monthly fee is 100.",
        found,
        citation=cite(evidence_ids=[]),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 100.",
                subject="fee",
                value=FEE,
                doc_id=DOC_ID,
                clause=CLAUSE,
            )
        ],
    )
    assert request.citation_check is not None
    assert request.citation_check.citations[0].evidence_ids == []
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.missing_citation)
    assert item.severity == Severity.medium
    assert item.fixable is True
    assert item.expected and FEE in item.expected
    assert FindingCode.wrong_number not in codes(verdict)
    assert verdict.suggested_action == Action.rewrite


def test_unmentioned_subject_is_unsupported_claim() -> None:
    found = fee_subject_materials()
    assert not any("berth" in part.casefold() for part in material_parts(found))
    request = draft(
        "The berth assignment stands.",
        found,
        citation=cite(doc_id="doc-fee", evidence_ids=[]),
        claims=[
            claim(
                "c-berth",
                ClaimKind.number,
                "The berth assignment stands.",
                subject="berth",
                value=None,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.unsupported_claim)
    assert item.claim_id == "c-berth"
    assert item.severity == Severity.low
    assert item.fixable is False
    assert_not_false(item)
    assert FindingCode.unverifiable not in codes(verdict)
    assert FindingCode.missing_citation not in codes(verdict)
    assert verdict.suggested_action == Action.accept


def test_only_low_severity_findings_accept_without_lawyer_text() -> None:
    found = fee_subject_materials()
    request = draft(
        "The monthly fee applies. The berth assignment stands.",
        found,
        citation=cite(doc_id="doc-fee", evidence_ids=["ev-fee"]),
        claims=[
            number_without_value(),
            claim(
                "c-berth",
                ClaimKind.number,
                "The berth assignment stands.",
                subject="berth",
                value=None,
            ),
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    assert FindingCode.unverifiable in codes(verdict)
    assert FindingCode.unsupported_claim in codes(verdict)
    assert verdict.findings
    assert all(item.severity == Severity.low for item in verdict.findings)
    assert verdict.suggested_action == Action.accept
    dumped = verdict.model_dump()
    for key in (
        "summary",
        "user_facing_text",
        "refusal",
        "rewritten_answer",
        "answer",
        "suggested_decision",
        "coverage",
    ):
        assert key not in dumped


def test_citation_not_in_context_with_replacement_rewrites() -> None:
    found = retrieval(
        documents=[document()],
        evidence=[evidence(text="The monthly fee is 100.")],
    )
    reference = "doc-missing"
    request = draft(
        "See doc-missing.",
        found,
        citation=citation_check(
            valid=False,
            citations=[],
            unknown_docs=[],
            unknown_clauses=[],
            unsupported=[reference],
        ),
        claims=[],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.citation_not_in_context)
    assert item.severity == Severity.medium
    assert item.fixable is True
    assert item.expected and DOC_ID in item.expected
    assert verdict.suggested_action == Action.rewrite
    assert verdict.suggested_action != Action.refuse


def test_citation_not_in_context_without_replacement_rewrites() -> None:
    found = retrieval(
        documents=[],
        evidence=[],
        signals=signals(entity_match=[]),
        trace=[],
    )
    reference = "doc-missing"
    assert reference not in " ".join(material_parts(found))
    request = draft(
        "See doc-missing.",
        found,
        citation=citation_check(
            valid=False,
            citations=[],
            unknown_docs=[],
            unknown_clauses=[],
            unsupported=[reference],
        ),
        claims=[],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    item = one(verdict, FindingCode.citation_not_in_context)
    assert item.severity == Severity.medium
    assert item.fixable is False
    assert item.severity != Severity.high
    assert FindingCode.fabricated_citation not in codes(verdict)
    assert verdict.suggested_action == Action.rewrite
    assert verdict.suggested_action != Action.refuse


def test_refuse_is_stricter_than_rewrite() -> None:
    found = retrieval(
        documents=[
            document(
                doc_id="doc-repealed",
                status="repealed",
                applicable_edition=False,
                edition_note="",
            )
        ],
        evidence=[
            evidence(
                doc_id="doc-repealed",
                text="The monthly fee is 100.",
            )
        ],
        signals=signals(),
    )
    request = draft(
        "The monthly fee is 250. The draft cites doc-repealed.",
        found,
        citation=cite(doc_id="doc-repealed"),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 250.",
                subject="fee",
                value=DRAFT_FEE,
            ),
            claim(
                "c-doc",
                ClaimKind.document,
                "doc-repealed",
                value="doc-repealed",
                doc_id="doc-repealed",
            ),
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    assert one(verdict, FindingCode.wrong_number).fixable is True
    assert one(verdict, FindingCode.repealed_doc).fixable is False
    assert verdict.suggested_action == Action.refuse


def test_rewrite_is_stricter_than_accept() -> None:
    found = retrieval(evidence=[evidence(text="The monthly fee is 100.")])
    request = draft(
        "The monthly fee is 250. The berth assignment stands.",
        found,
        citation=cite(),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 250.",
                subject="fee",
                value=DRAFT_FEE,
            ),
            claim(
                "c-berth",
                ClaimKind.number,
                "The berth assignment stands.",
                subject="berth",
                value=None,
            ),
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    assert one(verdict, FindingCode.wrong_number).fixable is True
    unsupported = one(verdict, FindingCode.unsupported_claim)
    assert unsupported.severity == Severity.low
    assert_not_false(unsupported)
    assert verdict.suggested_action == Action.rewrite


def test_event_date_uses_signals_not_the_run_date() -> None:
    found = retrieval(
        documents=[
            document(
                edition_note=EDITION,
                applicable_edition=True,
                effective_from="2024-01-01",
                effective_to="2024-12-31",
            )
        ],
        evidence=[evidence(text="The in-force text names edition-1.")],
        signals=signals(event_date=EVENT_DATE, applicable_editions=[EDITION]),
    )
    request = draft(
        "The draft relies on edition-9.",
        found,
        citation=cite(),
        claims=[],
        event_date=None,
    )
    assert request.event_date is None
    assert found.signals.event_date == EVENT_DATE
    assert OTHER_EDITION not in found.signals.applicable_editions
    actions = []
    # 2020-01-01 is outside the document window. 2024-06-01 is inside it.
    for year, month, day in ((2020, 1, 1), (2024, 6, 1)):
        with patched_calendar(year, month, day):
            verdict = decision.fact_check(request, config=None)
            assert_readable(verdict)
            assert_fixable_grounded(verdict, found)
            item = one(verdict, FindingCode.wrong_edition)
            assert item.fixable is True
            assert item.severity == Severity.high
            actions.append(verdict.suggested_action)
    assert actions[0] == actions[1]
    assert actions[0] == Action.rewrite


def test_check_rules_null_uses_decision_config() -> None:
    assert DecisionConfig().check_rules is False
    text = "Storage stays indoors."
    found = retrieval(evidence=[evidence(text=text)])
    confirmed = claim("c-rule", ClaimKind.rule, text)
    request = draft(
        text,
        found,
        citation=cite(),
        claims=[confirmed],
        check_rules=None,
    )
    assert request.check_rules is None
    off = decision.fact_check(request, config=None)
    assert_readable(off)
    assert_fixable_grounded(off, found)
    assert_unchecked_statement(off, "c-rule")
    assert off.suggested_action == Action.accept

    on = decision.fact_check(
        request,
        config=DecisionConfig(check_rules=True),
    )
    assert_readable(on)
    assert_fixable_grounded(on, found)
    assert "c-rule" in on.checked_claims
    assert "c-rule" not in on.unverified_claims
    for item in rule_findings(on, "c-rule"):
        assert item.code != FindingCode.unverifiable
        assert_not_false(item)
    assert on.suggested_action == Action.accept


def test_check_rules_true_leaves_unconfirmed_rule_unverifiable() -> None:
    confirmed_quote = "Storage stays indoors."
    unconfirmed_quote = "Cranes halt at dusk."
    found = retrieval(evidence=[evidence(text=confirmed_quote)])
    request = draft(
        f"{confirmed_quote} {unconfirmed_quote}",
        found,
        citation=cite(),
        claims=[
            claim("c-yes", ClaimKind.rule, confirmed_quote),
            claim("c-no", ClaimKind.rule, unconfirmed_quote),
        ],
        check_rules=True,
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert_fixable_grounded(verdict, found)
    assert "c-yes" in verdict.checked_claims
    assert "c-yes" not in verdict.unverified_claims
    assert_unchecked_statement(verdict, "c-no")
    assert verdict.suggested_action == Action.accept


def test_unreadable_retrieval_refuses_as_contract_invalid() -> None:
    payload = {
        "question": "q",
        "answer": "a",
        "retrieval": {"signals": {"max_relevance": 2}},
    }
    verdict = decision.fact_check(payload, config=None)
    assert_shape(verdict)
    assert verdict.suggested_action == Action.refuse
    assert verdict.layer_status == LayerStatus.failed
    assert verdict.failure_reason == FailureReason.contract_invalid
    assert verdict.findings == []
    assert_not_gap(verdict)


def test_other_citation_evidence_ids_do_not_suppress_missing_citation() -> None:
    found = retrieval(
        evidence=[
            evidence(evidence_id="ev-fee", text="The monthly fee is 100."),
            evidence(evidence_id="ev-other", text="Other clause text."),
        ]
    )
    request = draft(
        "The monthly fee is 100.",
        found,
        citation=citation_check(
            valid=True,
            citations=[
                Citation(
                    raw="other",
                    doc_id="doc-other",
                    clause="Z",
                    evidence_ids=["ev-other"],
                ),
                Citation(
                    raw="fee",
                    doc_id=DOC_ID,
                    clause=CLAUSE,
                    evidence_ids=[],
                ),
            ],
            unknown_docs=[],
            unknown_clauses=[],
            unsupported=[],
        ),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 100.",
                subject="fee",
                value=FEE,
                doc_id=DOC_ID,
                clause=CLAUSE,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert one(verdict, FindingCode.missing_citation)


def test_date_subject_absent_emits_unsupported_not_wrong_date() -> None:
    found = retrieval(evidence=[evidence(text="The deadline is 2024-06-01.")])
    request = draft(
        "The berth closes on 2024-07-01.",
        found,
        citation=cite(),
        claims=[
            claim(
                "c-date",
                ClaimKind.date,
                "2024-07-01",
                subject="berth",
                value="2024-07-01",
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert FindingCode.wrong_date not in codes(verdict)
    assert one(verdict, FindingCode.unsupported_claim)


def test_statute_edition_identifier_is_not_split_for_wrong_edition() -> None:
    edition = "statute-edition-1994"
    found = retrieval(
        evidence=[evidence(text=f"Text under {edition}.")],
        signals=signals(applicable_editions=[edition]),
    )
    request = draft(
        f"The draft relies on {edition}.",
        found,
        citation=cite(),
        claims=[],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    assert FindingCode.wrong_edition not in codes(verdict)
    assert verdict.suggested_action == Action.accept


def test_russian_negation_in_fragment_refuses() -> None:
    found = retrieval(evidence=[evidence(text="Поставка не разрешена.")])
    request = draft(
        "Поставка разрешена.",
        found,
        citation=cite(),
        claims=[
            claim(
                "c-ship",
                ClaimKind.number,
                "Поставка разрешена",
                subject="поставка",
                value=None,
            )
        ],
    )
    verdict = decision.fact_check(request, config=None)
    assert_readable(verdict)
    denied = one(verdict, FindingCode.contradicts_evidence)
    assert denied.severity == Severity.high
    assert verdict.suggested_action == Action.refuse


def test_threshold_outside_range_refuses_as_config_invalid() -> None:
    request = draft(
        "The monthly fee is 100.",
        retrieval(),
        citation=cite(),
        claims=[
            claim(
                "c-fee",
                ClaimKind.number,
                "The monthly fee is 100.",
                subject="fee",
                value=FEE,
            )
        ],
    )
    verdict = decision.fact_check(
        request,
        config=DecisionConfig(min_relevance_semantic=-1),
    )
    assert_shape(verdict)
    assert verdict.suggested_action == Action.refuse
    assert verdict.layer_status == LayerStatus.failed
    assert verdict.failure_reason == FailureReason.config_invalid
    assert verdict.findings == []
    assert_not_gap(verdict)
