from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import style  # noqa: E402


def hard_codes(text: str) -> set[str]:
    return {v.code for v in style.audit(text) if v.severity == style.HARD}


class GoldAnswersTest(unittest.TestCase):
    """Эталонные ответы команды не должны получать ни одного жёсткого флага."""

    def test_demo_output_is_clean(self) -> None:
        path = ROOT / "demo_output.csv"
        if not path.is_file():
            self.skipTest("demo_output.csv отсутствует")
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 0)
        for row in rows:
            with self.subTest(qid=row["id"]):
                self.assertEqual(hard_codes(row["answer"]), set(), row["answer"])

    def test_legal_register_is_not_penalised(self) -> None:
        legal = (
            "В соответствии с п. 2.4 Регламента в редакции от 01.03.2025 договор подлежит "
            "одобрению генеральным директором. Указанное требование является обязательным; "
            "общество вправе осуществлять согласование в СЭД."
        )
        self.assertEqual(hard_codes(legal), set())


class DetectionTest(unittest.TestCase):
    def test_chatbot_opening(self) -> None:
        self.assertIn("chatbot_opening", hard_codes("Конечно! Аренда 450 000 рублей."))
        self.assertIn("chatbot_opening", hard_codes("Отличный вопрос, сейчас разберёмся."))

    def test_chatbot_closing(self) -> None:
        self.assertIn("chatbot_closing", hard_codes("Аренда 450 000 рублей. Надеюсь, это помогло!"))
        self.assertIn("chatbot_closing", hard_codes("Аренда 450 000 рублей. Обращайтесь."))

    def test_thanks_and_apology(self) -> None:
        self.assertIn("gratitude", hard_codes("Спасибо за обращение. Аренда 450 000 рублей."))
        self.assertIn("apology", hard_codes("Извините, в базе нет такого документа."))

    def test_second_person(self) -> None:
        self.assertIn("second_person_singular", hard_codes("Тебе нужно проверить п. 3.1 договора."))
        self.assertIn("second_person_singular", hard_codes("Твой договор предусматривает штраф."))

    def test_colloquialism(self) -> None:
        self.assertIn("colloquialism", hard_codes("Кстати, аренда 450 000 рублей, ок?"))
        self.assertIn("colloquialism", hard_codes("Смотри, тут всё просто: 450 000 рублей."))

    def test_model_identity(self) -> None:
        self.assertIn("model_identity", hard_codes("Как ИИ, я не могу дать гарантию."))
        self.assertIn("model_identity", hard_codes("В моей базе знаний нет такого пункта."))

    def test_profanity_and_promise(self) -> None:
        self.assertIn("profanity", hard_codes("Это пиздец, а не договор."))
        self.assertIn("colloquialism", hard_codes("Это полная фигня, а не договор."))
        self.assertIn("false_promise", hard_codes("Поверь, штрафа не будет."))

    def test_soft_flags_do_not_block(self) -> None:
        text = "Возможно, договор подлежит согласованию."
        self.assertEqual(hard_codes(text), set())
        soft = {v.code for v in style.audit(text) if v.severity == style.SOFT}
        self.assertIn("hedge", soft)


class CleanupTest(unittest.TestCase):
    def test_markdown_and_emoji_are_removed(self) -> None:
        result = style.enforce_tone("**Аренда** — 450 000 ₽ 🙂 (п. 3.1)")
        self.assertEqual(result.clean, "Аренда — 450 000 ₽ (п. 3.1)")
        self.assertEqual(result.remaining, [])

    def test_chat_preamble_is_removed(self) -> None:
        result = style.enforce_tone("Конечно! Вот ответ: аренда 450 000 рублей.")
        self.assertEqual(result.clean, "аренда 450 000 рублей.")
        self.assertEqual(result.remaining, [])

    def test_invisible_characters_are_normalised(self) -> None:
        result = style.enforce_tone("Аренда\u202f450\u00a0000\u200b рублей.")
        self.assertEqual(result.clean, "Аренда 450 000 рублей.")

    def test_cjk_and_service_tokens_are_removed(self) -> None:
        result = style.enforce_tone("Аренда 450 000 рублей (п. 3.1).【источник】合同")
        self.assertEqual(result.clean, "Аренда 450 000 рублей (п. 3.1).")

    def test_newlines_collapse_to_one_line(self) -> None:
        result = style.enforce_tone("- пункт 3.1\n- 450 000 рублей")
        self.assertNotIn("\n", result.clean)
        self.assertEqual(result.clean, "пункт 3.1 450 000 рублей")

    def test_empty_input(self) -> None:
        self.assertEqual(style.enforce_tone("").clean, "")
        self.assertEqual(style.audit(""), [])


class ContractTest(unittest.TestCase):
    def test_contract_covers_register(self) -> None:
        for marker in ["«ты»", "markdown", "эмодзи", "Конечно", "официально-деловой"]:
            self.assertIn(marker, style.STYLE_CONTRACT)

    def test_repair_message_lists_issues(self) -> None:
        violations = style.audit_hard("Кстати, тебе стоит проверить п. 3.1.")
        message = style.build_repair_message(violations)
        self.assertIn("Перепиши ответ", message)
        self.assertIn("—", message)
        self.assertGreaterEqual(len(violations), 2)


if __name__ == "__main__":
    unittest.main()
