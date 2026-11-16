# Оркестратор submission

Поток:

```text
questions.csv
-> guard G1: decline/refuse_assist/sanitize/allow
-> rag.retrieve -> judge.evaluate
-> judge-находки уходят модели как указания, а не как фильтр:
   premise_errors, gaps, event_date, coverage
-> guard G2 scan_context
-> guard G3 build_messages -> LLM (retries + time budget)
-> tone.normalize -> guard G4 + tone
-> один rewrite при замечаниях
-> submission.csv
```

`decision.evaluate` больше не отменяет ответ до генерации: `reject_*` и
`clarify` превращаются в подсказки модели. `refuse_assist` идёт в генерацию
с директивами guard: не помогать с нарушением, но разъяснить требования по
базе. `decline` — единственная ветка, где сразу возвращается текст отказа.

RAG отвечает только за знания и контекст. tone только проверяет регистр.
guard проверяет документы (G0), вопрос/историю (G1), контекст (G2),
промпт (G3) и ответ (G4).

## Запуск

```bash
python solution.py --data-dir data
python solution.py --data-dir data --limit 5
ORCHESTRATOR_OFFLINE=1 python solution.py --data-dir . --questions demo_questions.csv
python -m orchestrator.build_submission
```

`ORCHESTRATOR_OFFLINE=1` не вызывает LLM и собирает ответ из наиболее
близкого фрагмента RAG — только для локальной проверки CSV.

## Переменные

`DATA_DIR`, `LLM_BASE_URL`, `LLM_API_KEY`, `EMBEDDER_API_KEY`,
`RERANKER_API_KEY`, `LLM_MODEL`, `EMB_MODEL`, `RERANKER_MODEL`,
`CONTEXT_MAX_CHARS`, `TEMPERATURE`, `REWRITE_LIMIT`,
`ENABLE_DENSE`, `ENABLE_RERANK`, `GUARD_G1`, `GUARD_LLM`,
`GUARD_LLM_MODE`, `JUDGE`.

Свои лимиты генерации и поиска:

```text
ORCH_TOP_K          вместо платформенного TOP_K
ORCH_MAX_TOKENS     вместо платформенного MAX_TOKENS
ORCH_WORKERS        число параллельных вопросов (fallback: MAX_WORKERS)
LLM_RETRIES         дополнительные повторы вызова модели
TIME_BUDGET_SEC     бюджет всего прогона
REWRITE_CUTOFF      доля бюджета, после которой rewrite отключается
```

Платформенные `TOP_K` и `MAX_TOKENS` только логируются на старте и не
подменяют собственные значения. `MAX_TOKENS` по умолчанию `3000`: лимит
нужен и на рассуждение модели.

Локальный `.env` читается автоматически и в ZIP не попадает. Для локального
DeepSeek: `ENABLE_DENSE=0`, `ENABLE_RERANK=0`, `GUARD_LLM=1`.
`GUARD_LLM_MODE=auto` включает LLM-классификатор только для подозрительных
вопросов; `always` — для всех.

Без `LLM_BASE_URL`/`LLM_API_KEY` оркестратор автоматически работает в
offline-режиме и всё равно пишет валидный `submission.csv`.

`ENABLE_DENSE=1` (по умолчанию) считает embeddings из платформенной
`kb.zip` один раз на старте. `GUARD_G1` по умолчанию включён.
`JUDGE=1` (по умолчанию) включает `decision.evaluate`; `JUDGE=0` отключает
подсказки покрытия.

Вопросы обрабатываются параллельно в `ThreadPoolExecutor`; порядок строк в
CSV сохраняется по входному файлу. `forward.py` остаётся локальным
инструментом и в ZIP не включается.

## ZIP

`python -m orchestrator.build_submission` создаёт `dist/submission.zip`:
`solution.py`, `rag/`, `tone/`, `guard/`, `orchestrator/`, `decision/`,
`contracts/`, минимальный `requirements.txt` с `httpx`, `numpy` и
`pydantic`.
