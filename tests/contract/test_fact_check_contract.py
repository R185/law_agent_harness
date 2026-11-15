"""Contract tests for decision.fact_check.

The handle is not implemented. These tests only describe the contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import decision
from contracts import CitationCheck
from decision import (
    Action,
    Claim,
    ClaimKind,
    FactCheckInput,
    FactCheckVerdict,
    FindingCode,
)
from tests.support import citation_check, evidence, retrieval

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


def _number_claim() -> Claim:
    return Claim(
        claim_id="c-fee",
        text="The monthly fee is 100.",
        kind=ClaimKind.number,
        quote="The monthly fee is 100.",
        subject="fee",
        value="100",
        doc_id="doc-1",
        clause="A",
    )


def _matching_input(*, check: CitationCheck | None) -> FactCheckInput:
    found = retrieval(
        evidence=[evidence(text="The monthly fee is 100.")],
    )
    return FactCheckInput(
        question="What is the monthly fee?",
        answer="The monthly fee is 100.",
        retrieval=found,
        citation_check=check,
        claims=[_number_claim()],
        event_date=None,
        check_rules=False,
    )


def _assert_verdict_shape(verdict: FactCheckVerdict) -> None:
    assert verdict.suggested_action in {
        Action.accept,
        Action.rewrite,
        Action.refuse,
    }
    assert set(verdict.checked_claims).isdisjoint(verdict.unverified_claims)
    assert all(isinstance(item, str) for item in verdict.checked_claims)
    assert all(isinstance(item, str) for item in verdict.unverified_claims)
    dumped = verdict.model_dump()
    assert LAWYER_KEYS.isdisjoint(dumped)


def test_verdict_is_one_action_and_hides_lawyer_text() -> None:
    """One action, disjoint claim ids, and no refusal, draft, or coverage key."""

    assert "event_date" not in FactCheckVerdict.model_fields
    assert LAWYER_KEYS.isdisjoint(FactCheckVerdict.model_fields)
    request = _matching_input(check=citation_check())
    verdict = decision.fact_check(request, config=None)
    _assert_verdict_shape(verdict)
    assert "event_date" not in verdict.model_dump()


def test_none_citation_check_is_not_an_empty_check() -> None:
    """citation_check None is a missing check. An empty CitationCheck is not."""

    found = retrieval(evidence=[evidence(text="The monthly fee is 100.")])
    base = {
        "question": "What is the monthly fee?",
        "answer": "The monthly fee is 100.",
        "retrieval": found,
        "claims": [_number_claim()],
        "event_date": None,
        "check_rules": False,
    }
    missing = FactCheckInput(**base, citation_check=None)
    empty = FactCheckInput(
        **base,
        citation_check=CitationCheck(
            valid=True,
            citations=[],
            unknown_docs=[],
            unknown_clauses=[],
            unsupported=[],
        ),
    )
    assert missing.citation_check is None
    assert isinstance(empty.citation_check, CitationCheck)
    assert missing.citation_check != empty.citation_check
    assert missing.model_dump()["citation_check"] is None
    assert empty.model_dump()["citation_check"] is not None

    missing_verdict = decision.fact_check(missing, config=None)
    empty_verdict = decision.fact_check(empty, config=None)
    _assert_verdict_shape(missing_verdict)
    _assert_verdict_shape(empty_verdict)
    missing_codes = {finding.code for finding in missing_verdict.findings}
    empty_codes = {finding.code for finding in empty_verdict.findings}
    assert FindingCode.citation_check_missing in missing_codes
    assert FindingCode.citation_check_missing not in empty_codes
    assert missing_codes != empty_codes


def test_input_has_no_gold_label_or_rewrite_used() -> None:
    """gold_label, rewrite_used, and rewrite_budget are not input fields."""

    assert "gold_label" not in FactCheckInput.model_fields
    assert "rewrite_used" not in FactCheckInput.model_fields
    assert "rewrite_budget" not in FactCheckInput.model_fields
    found = retrieval()
    for extra in (
        {"gold_label": "accept"},
        {"rewrite_used": True},
        {"rewrite_budget": 1},
    ):
        with pytest.raises(ValidationError):
            FactCheckInput(
                question="What is the monthly fee?",
                answer="The monthly fee is 100.",
                retrieval=found,
                **extra,
            )
    request = _matching_input(check=citation_check())
    verdict = decision.fact_check(request, config=None)
    _assert_verdict_shape(verdict)


def test_passed_number_claim_is_not_relabeled_as_a_rule() -> None:
    """A number claim stays a checked number when rule checking is off.

    An unchecked rule would be unverifiable and would not be checked.
    A matching number that lands in checked_claims was not relabeled.
    """

    request = _matching_input(check=citation_check())
    assert request.claims is not None
    assert request.claims[0].kind == ClaimKind.number
    verdict = decision.fact_check(request, config=None)
    _assert_verdict_shape(verdict)
    assert request.claims[0].kind == ClaimKind.number
    assert "c-fee" in verdict.checked_claims
    assert "c-fee" not in verdict.unverified_claims
    for finding in verdict.findings:
        if finding.claim_id == "c-fee":
            assert finding.code != FindingCode.unverifiable
            assert not finding.code.value.startswith("wrong_")
            assert finding.code != FindingCode.unsupported_claim
