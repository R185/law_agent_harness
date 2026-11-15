"""Live request JSON shape for evaluate and fact_check."""

from __future__ import annotations

import logging

import decision
from decision import CoverageRequest, DecisionConfig, FactCheckInput
from tests.live import live_helpers
from tests.support import citation_check, retrieval

_QUESTION = "What is the monthly fee?"


def test_evaluate_and_fact_check_request_bodies(monkeypatch, caplog):
    api_key = live_helpers.read_key()
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="decision.llm")
    sent = live_helpers.hook_client_send(monkeypatch)
    decision.evaluate(
        CoverageRequest(question=_QUESTION, retrieval=retrieval()),
        config=DecisionConfig(model_mode="on"),
    )
    decision.fact_check(
        FactCheckInput(
            question=_QUESTION,
            answer="The monthly fee is 100.",
            retrieval=retrieval(),
            citation_check=citation_check(),
        ),
        config=DecisionConfig(model_mode="on"),
    )
    assert len(sent) == 2
    for request in sent:
        live_helpers.assert_request_target(request)
        live_helpers.assert_request_json_shape(request)
        live_helpers.assert_secret_absent(api_key, request.content.decode(), "request")