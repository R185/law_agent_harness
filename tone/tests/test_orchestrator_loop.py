from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tone  # noqa: E402

BASE_PROMPT = "Ты — юридический ассистент. Отвечай по фрагментам базы знаний."
CONTEXT = "3.1. Арендная плата составляет 450 000 рублей в месяц."
QUESTION = "Какая арендная плата по договору аренды?"

BAD = "Конечно! Смотри, аренда 450 000 рублей, кстати это выгодно тебе 🙂"
GOOD = "Арендная плата — 450 000 рублей в месяц, включая НДС (п. 3.1)."
FALLBACK = "Не удалось подготовить ответ, соответствующий служебному регистру."


def run_pipeline(responses: list[str], max_rewrites: int = 2):
    """Оркестраторский цикл из TONE.md: один список проблем на все валидаторы."""
    queue = list(responses)
    calls: list[list[dict[str, str]]] = []

    def llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    messages = [
        {"role": "system", "content": tone.build_system_prompt(BASE_PROMPT)},
        {"role": "user", "content": f"Фрагменты:\n{CONTEXT}\n\nВопрос: {QUESTION}"},
    ]

    answer = tone.normalize(llm(messages)).text
    issues = tone.check(answer, CONTEXT)
    attempts = 0

    while issues and attempts < max_rewrites:
        attempts += 1
        messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Перепиши, устранив: " + "; ".join(v.hint for v in issues)},
        ]
        answer = tone.normalize(llm(messages)).text
        issues = tone.check(answer, CONTEXT)

    if issues:
        answer = FALLBACK

    return answer, calls, attempts, issues


class LoopTest(unittest.TestCase):
    def test_bad_then_good(self) -> None:
        answer, calls, attempts, issues = run_pipeline([BAD, GOOD])
        self.assertEqual(answer, GOOD)
        self.assertEqual(attempts, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(issues, [])

    def test_rewrite_prompt_carries_hints(self) -> None:
        _, calls, _, _ = run_pipeline([BAD, GOOD])
        repair_prompt = calls[1][-1]["content"]
        self.assertIn("Перепиши", repair_prompt)
        self.assertIn("«ты»", repair_prompt)

    def test_good_answer_is_not_rewritten(self) -> None:
        answer, calls, attempts, _ = run_pipeline([GOOD])
        self.assertEqual(answer, GOOD)
        self.assertEqual(attempts, 0)
        self.assertEqual(len(calls), 1)

    def test_persistent_violation_falls_back(self) -> None:
        answer, _, attempts, issues = run_pipeline([BAD, BAD, BAD])
        self.assertEqual(answer, FALLBACK)
        self.assertEqual(attempts, 2)
        self.assertTrue(issues)

    def test_system_prompt_contains_contract(self) -> None:
        _, calls, _, _ = run_pipeline([GOOD])
        self.assertIn(tone.STYLE_CONTRACT, calls[0][0]["content"])

    def test_normalize_runs_before_check(self) -> None:
        answer, _, _, _ = run_pipeline(["**Аренда** — 450 000 рублей (п. 3.1)."])
        self.assertEqual(answer, "Аренда — 450 000 рублей (п. 3.1).")


if __name__ == "__main__":
    unittest.main()
