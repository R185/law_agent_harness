"""Live degraded paths where the core already refused before the model."""

from __future__ import annotations

import decision
from decision import CoverageRequest, Decision, DecisionConfig, FactCheckInput, LayerStatus
from decision.factcheck import Action
from tests.live.live_helpers import assert_one_request, run_live, validate_fact_check, validate_recommendation
from tests.support import retrieval, signals

_QUESTION = "What is the monthly fee?"


def test_empty_retrieval_stays_not_clarify(monkeypatch, caplog):
    found = retrieval(documents=[], evidence=[], signals=signals(distinct_docs=0))

    def _call():
        result = decision.evaluate(
            CoverageRequest(question=_QUESTION, retrieval=found),
            config=DecisionConfig(model_mode="on"),
        )
        assert result.suggested_decision != Decision.clarify
        assert result.layer_status in {LayerStatus.ok, LayerStatus.degraded}
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_recommendation)
    assert_one_request(sent)


def test_fact_check_without_citation_not_accept(monkeypatch, caplog):
    def _call():
        result = decision.fact_check(
            FactCheckInput(
                question=_QUESTION,
                answer="The monthly fee is 100.",
                retrieval=retrieval(),
                citation_check=None,
            ),
            config=DecisionConfig(model_mode="on"),
        )
        assert result.suggested_action != Action.accept
        assert result.layer_status in {LayerStatus.ok, LayerStatus.degraded}
        return result

    sent = run_live(monkeypatch, caplog, call=_call, validate=validate_fact_check)
    assert_one_request(sent)