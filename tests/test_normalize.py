from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tone  # noqa: E402


class ArtifactRemovalTest(unittest.TestCase):
    def test_markdown_and_emoji(self) -> None:
        result = tone.normalize("**Аренда** — 450 000 ₽ 🙂 (п. 3.1)")
        self.assertEqual(result.text, "Аренда — 450 000 ₽ (п. 3.1)")
        kinds = {item.split(":")[0] for item in result.removed_artifacts}
        self.assertIn("markdown", kinds)
        self.assertIn("emoji", kinds)

    def test_invisible_and_cjk(self) -> None:
        result = tone.normalize("Аренда\u202f450\u00a0000\u200b рублей 合同.")
        self.assertEqual(result.text, "Аренда 450 000 рублей.")
        self.assertTrue(result.removed_artifacts)

    def test_service_tokens(self) -> None:
        result = tone.normalize("Аренда 450 000 рублей (п. 3.1).【источник】")
        self.assertEqual(result.text, "Аренда 450 000 рублей (п. 3.1).")
        self.assertTrue(any(item.startswith("service_token") for item in result.removed_artifacts))

    def test_multiline_collapses(self) -> None:
        result = tone.normalize("- пункт 3.1\n- 450 000 рублей")
        self.assertNotIn("\n", result.text)
        self.assertEqual(result.text, "пункт 3.1 450 000 рублей")


class ServicePhrasesTest(unittest.TestCase):
    def test_opening_is_removed(self) -> None:
        result = tone.normalize("Конечно! Вот ответ: аренда 450 000 рублей.")
        self.assertEqual(result.text, "аренда 450 000 рублей.")
        self.assertTrue(result.removed_artifacts)

    def test_closing_sentence_is_removed(self) -> None:
        result = tone.normalize("Срок — 30 дней. Надеюсь, это помогло!")
        self.assertEqual(result.text, "Срок — 30 дней.")

    def test_standalone_invitation_is_removed(self) -> None:
        result = tone.normalize("Срок — 30 дней. Обращайтесь…")
        self.assertEqual(result.text, "Срок — 30 дней.")

    def test_legal_instruction_is_preserved(self) -> None:
        text = "В соответствии с п. 5.4 Регламента обращайтесь в юридический отдел."
        result = tone.normalize(text)
        self.assertEqual(result.text, text)
        self.assertEqual(result.removed_artifacts, [])

    def test_meaning_is_not_truncated(self) -> None:
        text = "По вопросу санкционной проверки обращайтесь в комплаенс-службу (п. 5.4)."
        self.assertEqual(tone.normalize(text).text, text)


class ReferenceAnswersTest(unittest.TestCase):
    def test_gold_answers_are_unchanged(self) -> None:
        path = ROOT / "demo_output.csv"
        if not path.is_file():
            self.skipTest("demo_output.csv отсутствует")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(qid=row["id"]):
                result = tone.normalize(row["answer"])
                self.assertEqual(result.text, row["answer"])
                self.assertEqual(result.removed_artifacts, [])


class EdgeCaseTest(unittest.TestCase):
    def test_empty(self) -> None:
        result = tone.normalize("")
        self.assertEqual(result.text, "")
        self.assertEqual(result.removed_artifacts, [])

    def test_plain_text_untouched(self) -> None:
        text = "Аренда — 450 000 рублей в месяц (п. 3.1)."
        self.assertEqual(tone.normalize(text).text, text)


if __name__ == "__main__":
    unittest.main()
