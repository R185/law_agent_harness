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


class BuildSystemPromptTest(unittest.TestCase):
    def test_contract_is_appended(self) -> None:
        base = "Ты — юридический ассистент."
        prompt = tone.build_system_prompt(base)
        self.assertTrue(prompt.startswith(base))
        self.assertIn(tone.STYLE_CONTRACT, prompt)
        self.assertEqual(prompt, base + "\n\n" + tone.STYLE_CONTRACT)

    def test_empty_base_returns_contract(self) -> None:
        self.assertEqual(tone.build_system_prompt(""), tone.STYLE_CONTRACT)

    def test_disabled_by_flag(self) -> None:
        base = "Ты — юридический ассистент."
        with patch.dict(os.environ, {"ENABLE_TONE": "0"}):
            self.assertEqual(tone.build_system_prompt(base), base)

    def test_idempotent(self) -> None:
        once = tone.build_system_prompt("База")
        twice = tone.build_system_prompt(once)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count(tone.STYLE_CONTRACT), 1)


class ContractContentTest(unittest.TestCase):
    def test_contract_states_opening_and_refusal(self) -> None:
        contract = tone.STYLE_CONTRACT
        self.assertIn("Начинай с существа", contract)
        self.assertIn("В предоставленной базе знаний отсутствуют документы", contract)

    def test_contract_forbids_chat_register(self) -> None:
        for marker in ["«ты»", "markdown", "эмодзи", "Конечно", "Надеюсь, это помогло"]:
            with self.subTest(marker=marker):
                self.assertIn(marker, tone.STYLE_CONTRACT)

    def test_contract_allows_legal_vocabulary(self) -> None:
        for marker in ["в соответствии с", "вправе", "подлежит"]:
            with self.subTest(marker=marker):
                self.assertIn(marker, tone.STYLE_CONTRACT)


if __name__ == "__main__":
    unittest.main()
