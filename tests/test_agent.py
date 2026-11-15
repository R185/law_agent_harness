from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tone  # noqa: E402
from tone import agent as tone_agent  # noqa: E402

GOOD_JSON = '{"violations": [{"code": "familiarity", "severity": "hard", "quote": "как другу", "hint": "убери фамильярность"}]}'


class ParseTest(unittest.TestCase):
    def test_valid_json(self) -> None:
        found = tone_agent.parse_violations(GOOD_JSON)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].code, "familiarity")
        self.assertEqual(found[0].severity, "hard")
        self.assertEqual(found[0].source, "agent")
        self.assertEqual(found[0].quote, "как другу")

    def test_json_inside_prose(self) -> None:
        raw = "Результат проверки:\n" + GOOD_JSON + "\nКонец."
        self.assertEqual(len(tone_agent.parse_violations(raw)), 1)

    def test_broken_json_does_not_raise(self) -> None:
        for raw in ["", "не json", "{нарушения}", '{"violations": ]']:
            with self.subTest(raw=raw):
                self.assertEqual(tone_agent.parse_violations(raw), [])

    def test_wrong_schema(self) -> None:
        for raw in ['{"items": []}', '["violations"]', '{"violations": "нет"}']:
            with self.subTest(raw=raw):
                self.assertEqual(tone_agent.parse_violations(raw), [])

    def test_unknown_severity_becomes_soft(self) -> None:
        raw = '{"violations": [{"code": "tone", "severity": "HIGH", "quote": "x"}]}'
        found = tone_agent.parse_violations(raw)
        self.assertEqual(found[0].severity, "soft")

    def test_items_without_quote_are_dropped(self) -> None:
        raw = '{"violations": [{"code": "tone", "severity": "hard"}, {"code": "ok", "quote": "y"}]}'
        found = tone_agent.parse_violations(raw)
        self.assertEqual([v.code for v in found], ["ok"])

    def test_cap_on_violations(self) -> None:
        items = ",".join(['{"code": "x", "severity": "hard", "quote": "q%d"}' % i for i in range(9)])
        found = tone_agent.parse_violations('{"violations": [' + items + "]}")
        self.assertEqual(len(found), tone_agent.MAX_VIOLATIONS)


class AgentCallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def spy(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return GOOD_JSON

    def test_agent_is_used_when_enabled(self) -> None:
        with patch.dict(os.environ, {"ENABLE_TONE_AGENT": "1"}):
            found = tone.check("Аренда 450 000 рублей.", agent=self.spy)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("familiarity", [v.code for v in found])
        self.assertEqual(found[0].source, "agent")

    def test_agent_is_ignored_when_disabled(self) -> None:
        with patch.dict(os.environ, {"ENABLE_TONE_AGENT": "0"}):
            found = tone.check("Аренда 450 000 рублей.", agent=self.spy)
        self.assertEqual(self.calls, [])
        self.assertEqual(found, [])

    def test_messages_carry_prompt_and_answer(self) -> None:
        with patch.dict(os.environ, {"ENABLE_TONE_AGENT": "1"}):
            tone.check("Аренда 450 000 рублей.", agent=self.spy, context="п. 3.1 договора")
        messages = self.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("JSON", messages[0]["content"])
        self.assertIn("Аренда 450 000 рублей.", messages[1]["content"])
        self.assertIn("п. 3.1 договора", messages[1]["content"])

    def test_agent_failure_is_skipped(self) -> None:
        def boom(messages):
            raise TimeoutError("network")

        with patch.dict(os.environ, {"ENABLE_TONE_AGENT": "1"}):
            found = tone.check("Аренда 450 000 рублей.", agent=boom)
        self.assertEqual(found, [])

    def test_agent_non_string_response(self) -> None:
        def silent(messages):
            return None

        with patch.dict(os.environ, {"ENABLE_TONE_AGENT": "1"}):
            found = tone.check("Аренда 450 000 рублей.", agent=silent)  # type: ignore[arg-type]
        self.assertEqual(found, [])

    def test_regex_and_agent_are_merged(self) -> None:
        with patch.dict(os.environ, {"ENABLE_TONE_AGENT": "1"}):
            found = tone.check("Конечно! Аренда 450 000 рублей.", agent=self.spy)
        codes = [v.code for v in found]
        self.assertIn("chatbot_opening", codes)
        self.assertIn("familiarity", codes)
        self.assertEqual(found[0].severity, tone.HARD)


if __name__ == "__main__":
    unittest.main()
