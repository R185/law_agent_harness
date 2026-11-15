"""Live classifier smoke test. Not part of the mandatory suite."""

from __future__ import annotations

import decision
from decision import CoverageRequest, DecisionConfig
from tests.live.live_helpers import (
    assert_one_request,
    run_live,
    validate_recommendation,
)
from tests.support import retrieval

_QUESTION = "What is the monthly fee?"


def test_enabled_mode_parses_recommendation(monkeypatch, caplog):
    sent = run_live(
        monkeypatch,
        caplog,
        call=lambda: decision.evaluate(
            CoverageRequest(question=_QUESTION, retrieval=retrieval()),
            config=DecisionConfig(model_mode="on"),
        ),
        validate=validate_recommendation,
    )
    assert_one_request(sent)