"""Same question and retrieval stay stable when the process calendar changes.

``evaluate`` is called from the test body. ``model_mode`` stays off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from decision import CoverageRequest, ModelMode
from tests.support import retrieval

_QUESTION = "What is the monthly fee?"
_OTHER_DAY = "1999-03-04"


@pytest.fixture(autouse=True)
def _model_off(monkeypatch):
    monkeypatch.setenv("DECISION_MODEL_MODE", "off")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def _shift_calendar(monkeypatch) -> None:
    """Point ``date.today`` and ``datetime.now`` at another calendar day.

    The builtin methods cannot be assigned on the C types, so the classes
    that own those methods are replaced, including bindings already imported
    by ``decision``.
    """

    import datetime as dt

    real_date = dt.date
    real_datetime = dt.datetime

    class OtherDate(real_date):
        @classmethod
        def today(cls):
            return cls(1999, 3, 4)

    class OtherDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls(1999, 3, 4, 15, 45, 0)
            return cls(1999, 3, 4, 15, 45, 0, tzinfo=tz)

        @classmethod
        def today(cls):
            return cls(1999, 3, 4)

    monkeypatch.setattr(dt, "date", OtherDate)
    monkeypatch.setattr(dt, "datetime", OtherDateTime)

    for name, module in list(sys.modules.items()):
        if module is None or (name != "decision" and not name.startswith("decision.")):
            continue
        for attr, value in list(vars(module).items()):
            if value is real_date:
                monkeypatch.setattr(module, attr, OtherDate, raising=False)
            elif value is real_datetime:
                monkeypatch.setattr(module, attr, OtherDateTime, raising=False)


def test_repeated_evaluate_ignores_calendar_changes(monkeypatch):
    import datetime as dt
    import decision

    request = CoverageRequest(question=_QUESTION, retrieval=retrieval())
    first = decision.evaluate(request, config=None)

    _shift_calendar(monkeypatch)
    assert dt.date.today().isoformat() == _OTHER_DAY
    assert dt.datetime.now().date().isoformat() == _OTHER_DAY

    second = decision.evaluate(request, config=None)
    assert second.suggested_decision == first.suggested_decision
    assert list(second.reason_codes) == list(first.reason_codes)
    assert second.event_date == first.event_date
    assert first.event_date == request.retrieval.signals.event_date
    assert first.event_date == "2024-06-01"
    assert first.event_date != _OTHER_DAY
    assert second.event_date != _OTHER_DAY
    assert first.model_mode == ModelMode.off
    assert second.model_mode == ModelMode.off


def test_decision_sources_do_not_call_today_or_now():
    root = Path(__file__).resolve().parents[2] / "decision"
    hits = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "date.today" in source or "datetime.now" in source:
            hits.append(str(path.relative_to(root.parent)))
    assert hits == []
