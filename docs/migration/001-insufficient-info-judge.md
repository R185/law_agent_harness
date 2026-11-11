# Миграция: Insufficient Info Judge (задачи 5–6)

## Зависимости

- Python-пакет `harness` в корне репозитория (без отдельной установки, путь в `PYTHONPATH`).
- После `../rag`: передайте `chunks` (`RetrievedChunk`) и опционально `rag_max_score` / `rag_mean_score`.

## Порядок в оркестраторе

```python
from harness import JudgeInput, evaluate_insufficient_info, build_clarification

inp = JudgeInput(
    question=user_message,
    dialogue=history,
    chunks=retrieved_chunks,
    rag_max_score=metrics.max_score,
    rag_mean_score=metrics.mean_score,
)
verdict = evaluate_insufficient_info(inp)

if verdict.decision == "clarify":
    clar = build_clarification(verdict)
    return endpoint_clarification(clar.questions)

if verdict.decision.startswith("reject"):
    clar = build_clarification(verdict)  # опционально вопросы при partial
    return endpoint_refusal(verdict.user_facing_text, clar.questions, verdict.analysis)

# allow_answer → генератор / RAG answer; передайте verdict.analysis.premise_errors в промпт
```

## Согласование с коллегами

| Модуль | Интерфейс |
|--------|-----------|
| `../rag` | `list[RetrievedChunk]` |
| `../guard` | санитизация до `JudgeInput.question` |
| `../tone` | пост-обработка `user_facing_text` и вопросов |
| Оркестратор | единый enum `JudgeDecision` — не дублировать в других ветках |

## Конфигурация

`JudgeConfig(min_max_score=0.22, min_mean_score=0.12)` — пороги подгоняются ПМ на валидации.

## Промпты для LLM-режима

- `harness/insufficient_info/prompts/judge_system.md`
- `harness/insufficient_info/prompts/clarify_system.md`

Детерминированный режим (по умолчанию): `evaluate_insufficient_info` без LLM.

## Тесты

```bash
pip install pytest
pytest benchmarks/insufficient_info/test_heuristics.py -q
python benchmarks/insufficient_info/llm_judge.py  # при наличии LLM_API_KEY
```
