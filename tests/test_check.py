from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tone  # noqa: E402


def codes(text: str, context: str | None = None) -> list[str]:
    return [v.code for v in tone.check(text, context)]


def hard_codes(text: str, context: str | None = None) -> list[str]:
    return [v.code for v in tone.check(text, context) if v.severity == tone.HARD]


class ChatbotResidueTest(unittest.TestCase):
    def test_opening(self) -> None:
        self.assertIn("chatbot_opening", hard_codes("Конечно! Аренда 450 000 рублей."))
        self.assertIn("chatbot_opening", hard_codes("Отличный вопрос, сейчас разберёмся."))

    def test_greeting(self) -> None:
        self.assertIn("greeting", hard_codes("Здравствуйте! Аренда 450 000 рублей."))

    def test_closing(self) -> None:
        self.assertIn("chatbot_closing", hard_codes("Аренда 450 000 рублей. Надеюсь, это помогло."))
        self.assertIn("chatbot_closing", hard_codes("Аренда 450 000 рублей. Обращайтесь!"))

    def test_gratitude_and_apology(self) -> None:
        self.assertIn("gratitude", hard_codes("Спасибо за обращение. Аренда 450 000 рублей."))
        self.assertIn("apology", hard_codes("Извините, в базе нет такого документа."))


class RegisterTest(unittest.TestCase):
    def test_second_person_cases(self) -> None:
        for text in [
            "Тебе нужно проверить п. 3.1 договора.",
            "Твоей заявки в СЭД нет.",
            "Твой договор предусматривает штраф.",
            "В твоих интересах согласовать проект.",
        ]:
            with self.subTest(text=text):
                self.assertIn("second_person_singular", hard_codes(text))

    def test_model_identity(self) -> None:
        for text in [
            "Как ИИ, я не могу дать гарантию.",
            "Как ассистент, я не могу дать гарантию.",
            "Я языковая модель и не имею доступа.",
            "В моей базе знаний нет такого пункта.",
            "На момент моего обучения норма была иной.",
        ]:
            with self.subTest(text=text):
                self.assertIn("model_identity", hard_codes(text))

    def test_colloquialism(self) -> None:
        self.assertIn("colloquialism", hard_codes("Кстати, аренда 450 000 рублей, ок?"))
        self.assertIn("colloquialism", hard_codes("Смотри, тут всё просто: 450 000 рублей."))

    def test_profanity_and_promise(self) -> None:
        self.assertIn("profanity", hard_codes("Это пиздец, а не договор."))
        self.assertIn("colloquialism", hard_codes("Это полная фигня, а не договор."))
        self.assertIn("false_promise", hard_codes("Поверь, штрафа не будет."))

    def test_soft_does_not_become_hard(self) -> None:
        text = "Возможно, договор подлежит согласованию."
        self.assertEqual(hard_codes(text), [])
        self.assertIn("hedge", codes(text))


class LegalVocabularyTest(unittest.TestCase):
    """Юридическая лексика не должна давать срабатываний — ни hard, ни soft."""

    def test_no_false_positives(self) -> None:
        text = (
            "Арендная плата 27 093 рубля в месяц. Сделки с небанковскими кредитными "
            "организациями и в посёлках городского типа согласуются в соответствии с п. 5.4 "
            "Регламента; общество вправе осуществлять согласование в СЭД."
        )
        self.assertEqual(codes(text), [])

    def test_quoted_document_phrase_is_skipped(self) -> None:
        context = "Правила применяются в отношении кредитных организаций, небанковскими организациями и посёлках городского типа."
        answer = "В отношении небанковскими организациями действует общий порядок (п. 1.2)."
        self.assertEqual(codes(answer, context), [])


class FormatTest(unittest.TestCase):
    def test_format_artifacts(self) -> None:
        self.assertIn("format_markdown", hard_codes("**Аренда** 450 000 рублей."))
        self.assertIn("format_emoji", hard_codes("Аренда 450 000 рублей 🙂"))
        self.assertIn("format_cjk", hard_codes("Аренда 450 000 рублей 合同"))
        self.assertIn("format_service_token", hard_codes("Аренда 450 000 рублей 【источник】"))

    def test_nbsp_is_soft(self) -> None:
        found = tone.check("Аренда 450\u00a0000 рублей.")
        self.assertEqual([v.code for v in found if v.severity == tone.HARD], [])
        self.assertIn("format_nbsp", [v.code for v in found])

    def test_exclamation_is_soft(self) -> None:
        found = tone.check("Аренда 450 000 рублей!")
        self.assertEqual([v.code for v in found if v.severity == tone.HARD], [])
        self.assertIn("exclamation", [v.code for v in found])


class EmptyAnswerTest(unittest.TestCase):
    def test_empty_is_hard(self) -> None:
        for text in ["", "   ", "\n"]:
            with self.subTest(text=text):
                found = tone.check(text)
                self.assertEqual([v.code for v in found], ["empty_answer"])
                self.assertEqual(found[0].severity, tone.HARD)


class ResultShapeTest(unittest.TestCase):
    def test_violation_fields_and_order(self) -> None:
        found = tone.check("Конечно! Возможно, аренда 450 000 рублей!")
        self.assertTrue(found)
        for violation in found:
            self.assertIn(violation.severity, {tone.HARD, tone.SOFT})
            self.assertEqual(violation.source, "regex")
            self.assertTrue(violation.hint)
        severities = [v.severity for v in found]
        self.assertEqual(severities, sorted(severities, key=lambda s: 0 if s == tone.HARD else 1))

    def test_check_does_not_modify_answer(self) -> None:
        text = "Конечно! Аренда 450 000 рублей."
        tone.check(text)
        self.assertEqual(text, "Конечно! Аренда 450 000 рублей.")

    def test_helpers_split_by_severity(self) -> None:
        found = tone.check("Конечно! Возможно, аренда 450 000 рублей!")
        self.assertEqual(len(tone.hard_only(found)) + len(tone.soft_only(found)), len(found))


if __name__ == "__main__":
    unittest.main()
