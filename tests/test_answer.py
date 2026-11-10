from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import style  # noqa: E402

HAS_PIPELINE = importlib.util.find_spec("rag.answer") is not None

CONTEXT = """=== [D1] Договор аренды № А-08/2024
--- фрагмент (п. 3.1)
3.1. Арендная плата составляет 450 000 рублей в месяц, включая НДС.
"""

CHATTY = "Конечно! Смотри, **аренда** тут 450 000 рублей в месяц, кстати это выгодно тебе 🙂"
CLEAN = "Аренда по договору № А-08/2024 — 450 000 рублей в месяц, включая НДС (п. 3.1)."
HALLUCINATED = "Аренда по договору № А-08/2024 — 500 000 рублей в месяц (п. 3.1)."


class FakeClient:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    def chat_completion(self, messages, model=None, temperature=None, max_tokens=None) -> str:
        self.last_messages = messages
        index = min(self.calls, len(self.answers) - 1)
        self.calls += 1
        return self.answers[index]


class BrokenClient:
    def chat_completion(self, *args, **kwargs) -> str:
        raise RuntimeError("network down")


if not HAS_PIPELINE:

    class PipelinePendingTest(unittest.TestCase):
        @unittest.skip("тест включается после мёржа ветки RAG: rag.answer в этой ветке отсутствует")
        def test_pipeline_available(self) -> None:
            pass

else:
    from rag import answer
    from rag.query import ParsedDialog, QueryAnalysis

    def make_args():
        dialog = ParsedDialog(
            history="", last_user="Какая арендная плата по договору с «Северным Складом»?"
        )
        analysis = QueryAnalysis(
            standalone_question=dialog.last_user, search_queries=[dialog.last_user]
        )
        return dialog, analysis

    class GenerateAnswerTest(unittest.TestCase):
        def run_generation(self, client) -> str:
            dialog, analysis = make_args()
            return answer.generate_answer(
                client=client, dialog=dialog, analysis=analysis, context=CONTEXT
            )

        def test_professional_answer_is_not_rewritten(self) -> None:
            client = FakeClient([CLEAN])
            result = self.run_generation(client)
            self.assertEqual(result, CLEAN)
            self.assertEqual(client.calls, 1)

        def test_chatty_answer_triggers_one_rewrite(self) -> None:
            client = FakeClient([CHATTY, CLEAN])
            result = self.run_generation(client)
            self.assertEqual(result, CLEAN)
            self.assertEqual(client.calls, 2)
            self.assertEqual(style.audit_hard(result), [])

        def test_rewrite_request_lists_register_problems_only(self) -> None:
            client = FakeClient([CHATTY, CLEAN])
            self.run_generation(client)
            repair_prompt = client.last_messages[-1]["content"]
            self.assertIn("Перепиши ответ", repair_prompt)
            self.assertIn("«ты»", repair_prompt)
            self.assertNotIn("не удалось получить ответ", repair_prompt.lower())

        def test_number_hallucination_is_repaired_in_the_same_pass(self) -> None:
            client = FakeClient([HALLUCINATED, CLEAN])
            result = self.run_generation(client)
            self.assertEqual(result, CLEAN)
            self.assertEqual(client.calls, 2)
            self.assertIn("500000", client.last_messages[-1]["content"])

        def test_unfixed_answer_is_returned_without_chat_scaffolding(self) -> None:
            client = FakeClient([CHATTY, CHATTY])
            result = self.run_generation(client)
            self.assertEqual(client.calls, 2)
            self.assertNotIn("Конечно", result)
            self.assertNotIn("**", result)
            self.assertNotIn("🙂", result)
            self.assertIn("450 000", result)

        def test_service_only_answer_falls_back(self) -> None:
            client = FakeClient(["Конечно! Надеюсь, это помогло. Обращайтесь!"])
            result = self.run_generation(client)
            self.assertEqual(client.calls, 2)
            self.assertEqual(result, answer.config.FALLBACK)

        def test_broken_client_falls_back(self) -> None:
            self.assertEqual(self.run_generation(BrokenClient()), answer.config.FALLBACK)

        def test_system_prompt_contains_style_contract(self) -> None:
            client = FakeClient([CLEAN])
            self.run_generation(client)
            self.assertIn(style.STYLE_CONTRACT, client.last_messages[0]["content"])

        def test_answer_is_a_single_line(self) -> None:
            client = FakeClient(["Аренда 450 000 рублей.\nПункт 3.1."])
            result = self.run_generation(client)
            self.assertNotIn("\n", result)


if __name__ == "__main__":
    unittest.main()
