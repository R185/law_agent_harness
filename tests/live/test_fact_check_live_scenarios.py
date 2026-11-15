"""Live fact-check scenarios at ``model_mode=on``."""

from __future__ import annotations

import decision
from decision import Claim, ClaimKind, DecisionConfig, FactCheckInput
from decision.factcheck import Action
from tests.live.live_helpers import assert_one_request, run_live, validate_fact_check
from tests.support import citation_check, document, evidence, retrieval, signals

_QUESTION = "What is the monthly fee?"


def _fact_check(answer, found, *, citation=None, claims=None, check_rules=None):
    return decision.fact_check(
        FactCheckInput(
            question=_QUESTION,
            answer=answer,
            retrieval=found,
            citation_check=citation,
            claims=claims,
            check_rules=check_rules,
        ),
        config=DecisionConfig(model_mode="on"),
    )


def test_matching_draft_not_refuse(monkeypatch, caplog):
    found = retrieval(
        documents=[document(parties=["North Warehouse LLC"])],
        evidence=[evidence(text="The monthly fee is 100.")],
    )
    citation = citation_check()

    def _call():
        result = _fact_check("The monthly fee is 100.", found, citation=citation)
        assert result.suggested_action != Action.refuse
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_fact_check)
    assert_one_request(sent)


def test_missing_citation_check_not_accept(monkeypatch, caplog):
    found = retrieval(evidence=[evidence(text="The monthly fee is 100.")])

    def _call():
        result = _fact_check("The monthly fee is 100.", found, citation=None)
        assert result.suggested_action != Action.accept
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_fact_check)
    assert_one_request(sent)


def test_unknown_reference_not_accept(monkeypatch, caplog):
    found = retrieval()
    citation = citation_check(
        valid=False,
        citations=[],
        unknown_docs=["doc-missing"],
        unknown_clauses=[],
        unsupported=[],
    )

    def _call():
        result = _fact_check("See doc-missing.", found, citation=citation)
        assert result.suggested_action != Action.accept
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_fact_check)
    assert_one_request(sent)


def test_rule_with_check_rules_false_not_checked(monkeypatch, caplog):
    found = retrieval(evidence=[evidence(text="Storage stays indoors.")])
    claims = [
        Claim(
            claim_id="rule-1",
            text="Cranes halt at dusk.",
            kind=ClaimKind.rule,
            quote="Cranes halt at dusk.",
        )
    ]

    def _call():
        result = _fact_check(
            "Cranes halt at dusk.",
            found,
            citation=citation_check(),
            claims=claims,
            check_rules=False,
        )
        assert "rule-1" not in result.checked_claims
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_fact_check)
    assert_one_request(sent)