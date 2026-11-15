# Слой решения (`decision`)

Импортируемая библиотека. Оркестратором не является: текст для юриста, один уточняющий вопрос, единственный цикл правки черновика и выпуск ответа остаются у оркестратора.

**До ответа** — по выдаче поиска, вопросу и диалогу возвращает один исход покрытия и разбор (`evaluate`).

**После черновика** — сверяет утверждения с фрагментами и документами выдачи, возвращает действие и находки (`fact_check`).

Оба шага читают контракт `contracts` (`RetrievalResult`, `CitationCheck` и связанные типы). Формы выдачи и проверки цитат слой из пакета поиска не берёт; сигналы заново не считает и `verify_citations` не реализует.

Импорт `decision` **не открывает сокет** и **не читает ключ API**. Сеть и ключи — только при `model_mode=on` внутри вызова `evaluate` / `fact_check`. При `model_mode=off` HTTP-клиент не создаётся.

---

## Публичные ручки

| Ручка | Когда | Возврат |
|---|---|---|
| `evaluate(request, config=None, transport=None)` | После поиска, до ответа | `Recommendation` |
| `fact_check(request, config=None, transport=None)` | После черновика; желательно с `CitationCheck` от поиска | `FactCheckVerdict` |

`request` — модель Pydantic или совместимый dict. Плохой порог или нечитаемый контракт **не бросают исключение**: `layer_status=failed`, см. раздел «Сбои».

`transport` — подмена HTTP для тестов/live; при `model_mode=off` игнорируется.

Экспорт: `from decision import evaluate, fact_check, CoverageRequest, FactCheckInput, …` — [`__init__.py`](./__init__.py).

### Закрытые исходы и действия

**Покрытие** — ровно четыре значения `suggested_decision`: `allow_answer`, `reject_insufficient_kb`, `reject_partial_coverage`, `clarify`. Других нет; проверка черновика этими полями не выражается.

**Черновик** — ровно три значения `suggested_action`: `accept`, `rewrite`, `refuse`. Исход покрытия действие не заменяет.

В обеих моделях **нет** полей: текста для юриста, готового вопроса, переписанного черновика, текста отказа. Поле `hint` в находке — только для оркестратора.

---

## Интеграция: что делает оркестратор

1. **После поиска** — собрать `RetrievalResult` из `contracts`, положить в `CoverageRequest` с `question` и `dialogue`, вызвать `evaluate`, читать `suggested_decision`.

2. **`clarify`** — оркестратор пишет **один** вопрос из `missing_fields` и `candidates` рекомендации.
   - `missing_fields`: только `contract`, `counterparty`, `period` (≤3, без повторов).
   - `candidates`: подмножество `retrieval.ambiguity`; у элемента есть `doc_id`, `title`, `doc_number`, `label`.
   - Текст вопроса слой не поставляет.

3. **`reject_insufficient_kb` / `reject_partial_coverage`** — текст отказа пишет оркестратор.
   - `layer_status` и `failure_reason` читать **отдельно** от кодов пробела в `gaps` / `reason_codes`.
   - Пустая читаемая выдача: `layer_status=ok`, `reject_insufficient_kb`, в разборе `missing_facts` — не технический сбой.
   - Технический сбой: `layer_status` `failed` или `degraded` и заданный `failure_reason` (это не `missing_facts`).

4. **`allow_answer`** — оркестратор составляет черновик, получает `CitationCheck` у поиска, вызывает `fact_check`.
   - `citation_check is None` — проверки не было (не то же самое, что пустой успешный `CitationCheck`).
   - Переданный пустой объект `CitationCheck` означает: проверка была, результат передан.

5. **Цикл правки по `suggested_action`**
   - `accept` — дальше к тону и защите.
   - `rewrite` — **единственный** цикл правки у оркестратора; бюджет правок слой не читает, число прошлых правок во входе нет.
   - `refuse` — выход из цикла сразу.
   - Повторный вызов с тем же входом → то же действие. Снова `citation_check is None` → снова `rewrite`; не выпускать черновик после правки решает оркестратор.
   - Текст правки и отказа, новый черновик — у оркестратора.

**Защита и цитаты:** существование цитаты — у поиска. Слой не дублирует проверку; `origin/guard` эта поставка не правит. Второй вердикт защиты о той же цитате (включая `_unsupported_refs`) — зона стыка с владельцем guard, не decision.

**Владение:** слой не владеет `solution.py` и не переписывает ответ банка.

---

## `evaluate` — рекомендация до ответа

```text
evaluate(request: CoverageRequest, config: DecisionConfig | None = None) -> Recommendation
```

### Вход: `CoverageRequest`

| Поле | Тип | Поведение |
|---|---|---|
| `question` | `str` | Текущий вопрос; важнее истории |
| `dialogue` | `list[DialogueTurn]` | `user` / `assistant`. Сущности, посылка, идентификаторы — из вопроса и реплик `user`; `assistant` поле не закрывает |
| `retrieval` | `RetrievalResult` | Готовая выдача; слой сигналы не пересчитывает |
| `question_kind` | `str \| null` | Ключ политики (`required_clauses`); `null` — списка пунктов нет |
| `policy` | `dict[str, list[str]] \| null` | Подмена на один вызов; `null` — `decision/coverage_policy.json` (по умолчанию `{}`) |

Эталонной метки в запросе нет.

### Выход: `Recommendation`

| Поле | Смысл |
|---|---|
| `suggested_decision` | Один из четырёх исходов покрытия |
| `reason_codes` | Закрытый список `ReasonCode`, без повторов |
| `premise_errors` | Посылка юриста vs материалы (возможно при `allow_answer`) |
| `ambiguities` | Копия `retrieval.ambiguity`, без новых кандидатов |
| `event_date` | Копия `retrieval.signals.event_date` |
| `applicable_editions` | Копия сигналов |
| `gaps` | Именованные пробелы (`code` + `detail`, не текст отказа) |
| `coverage` | `required` / `confirmed` / `missing` — id пунктов политики |
| `missing_fields` | До трёх: `contract`, `counterparty`, `period` |
| `candidates` | Кандидаты из `ambiguity` для уточнения |
| `trace` | Пояснения из выдачи |
| `layer_status` | `ok`, `degraded`, `failed` |
| `failure_reason` | `null` при `ok`, иначе причина |
| `model_mode` | `off` / `on` |

### Поведение (порядок правил)

Правила **последовательные**: первое сработавшее задаёт исход; поздние не перекрывают.

| Ситуация | `suggested_decision` | Коды (примеры) |
|---|---|---|
| `topic_match=false` | `reject_insufficient_kb` | `topic_mismatch` |
| `score_kind=order_only` | `reject_insufficient_kb` | `sufficiency_scale_unavailable` |
| Пустые `documents` и `evidence` | `reject_insufficient_kb` | `missing_facts` в `gaps`; **не** `clarify` |
| `reference_errors`, несколько применимых редакций или молчание о редакции | `reject_insufficient_kb` | `edition_missing`, `reference_error`, `ambiguous_edition` |
| Два случая уточнения | `clarify` | `ambiguous_entity`, `missing_identifier` |
| Политика не закрыта, уточнение не открыто | `reject_partial_coverage` | `partial_clause_coverage` |
| `unresolved`, имя не в `entity_match` | `reject_insufficient_kb` | `named_not_found` |
| `low_confidence`, низкая `max_relevance`, `date_conflict`, … | `reject_insufficient_kb` | соответствующие коды |
| Ворота пройдены | `allow_answer` | при расхождении посылки — `premise_error` |

`dense_used` и `rerank_used` сами исход не меняют. Слой **не извлекает** имена из текста вопроса: отказ по неподтверждённой сущности — только через `unresolved` / сигналы выдачи.

### Уточнение (`clarify`)

Слой не возвращает текст вопроса.

- **Случай 1:** не назван договор / контрагент / период; кандидатов в неоднозначности >1; `margin_top1_top2` **строго** < `config.max_clarify_margin`.
- **Случай 2:** в вопросе и репликах `user` нет идентификатора, без которого нельзя оставить ровно один документ (при >1 документа или >1 кандидата). Один оставшийся документ случай не открывает; пустая выдача — не `clarify`.

Год из четырёх цифр может закрыть период; в номер договора не пишется. Уже названное юристом поле в `missing_fields` не попадает.

---

## `fact_check` — проверка черновика

```text
fact_check(request: FactCheckInput, config: DecisionConfig | None = None) -> FactCheckVerdict
```

Не меняет рекомендацию покрытия, не переписывает черновик, не пишет отказ.

### Вход: `FactCheckInput`

| Поле | Тип | Поведение |
|---|---|---|
| `question` | `str` | Вопрос к черновику |
| `answer` | `str` | Черновик как есть |
| `retrieval` | `RetrievalResult` | Как у `evaluate` |
| `citation_check` | `CitationCheck \| null` | `null` = проверки **не было** |
| `claims` | `list[Claim] \| null` | `null` — слой извлекает виды сам; `rule`/`conclusion` только из переданного списка |
| `event_date` | `str \| null` | Для редакций; в ядре по умолчанию `retrieval.signals.event_date` |
| `check_rules` | `bool \| null` | `null` → `DecisionConfig.check_rules` |

### Выход: `FactCheckVerdict`

| Поле | Смысл |
|---|---|
| `suggested_action` | `accept` / `rewrite` / `refuse` |
| `findings` | Все находки, включая `low` |
| `checked_claims` / `unverified_claims` | `claim_id`; один id не в обоих списках |
| `layer_status` / `failure_reason` / `model_mode` | Как у рекомендации |

### Действия и находки

Агрегация: **`refuse` > `rewrite` > `accept`**.

| Условие | Код | Серьёзность | Действие (одна находка) |
|---|---|---|---|
| `unknown_docs` / `unknown_clauses` | `fabricated_citation` | high | `refuse` |
| `unsupported` (цитата) | `citation_not_in_context` | medium | `rewrite` (даже без замены) |
| `citation_check is None` | `citation_check_missing` | medium | `rewrite` |
| Неверное число/дата/сторона/документ, есть замена | `wrong_*` | high | `rewrite` |
| Редакция, сторона, `repealed` без замены | … | high | `refuse` |
| Прямое противоречие фрагмента | `contradicts_evidence` | high | `refuse` |
| Предмет в материалах, значения нет | `unverifiable` | low | не открывает `rewrite`; `rule`/`conclusion` при `check_rules=false` |
| Ни один фрагмент о предмете | `unsupported_claim` | low | не ложь; не смешивать с `unverifiable` |

`low` само по себе правку не требует: при переданной проверке цитат и без более строгих находок — `accept`. Неисправимое противоречие даёт `refuse`, даже если есть `citation_check_missing`.

Повтор с тем же `citation_check=null` → снова `rewrite`. Слой дату запуска не подставляет.

---

## Конфигурация: `DecisionConfig`

| Поле | По умолчанию | Назначение |
|---|---|---|
| `min_relevance_semantic` | `0.35` | Порог при `semantic` (0..1) |
| `min_relevance_lexical` | `0.20` | Порог при `lexical` |
| `max_clarify_margin` | `0.05` | Порог для случая 1 уточнения |
| `model_mode` | `off` | Опциональный LLM-классификатор |
| `model_timeout_seconds` | `20` | Таймаут модели |
| `check_rules` | `false` | Проверка `rule` / `conclusion` в `claims` |

Порог вне [0, 1] → `failed` / `config_invalid` (`evaluate`: `reject_insufficient_kb` без кодов; `fact_check`: `refuse` без находок).

### Переменные окружения

Секреты в репозиторий и логи не пишутся.

| Переменная | Поведение |
|---|---|
| `DECISION_MODEL_MODE` | `on` или `off`; незаданная и любое другое значение = `off`. Явный `DecisionConfig.model_mode` перекрывает. |
| Ключ (при `on`) | `DEEPSEEK_API_KEY` → `LLM_API_KEY` → `OPENAI_API_KEY` |
| База URL | `DEEPSEEK_BASE_URL` → `LLM_BASE_URL` → `OPENAI_BASE_URL`; иначе `https://api.deepseek.com` |
| Модель | `DEEPSEEK_MODEL` → `LLM_MODEL`; иначе `deepseek-flash` |

---

## Сбои: `layer_status` и `failure_reason`

| `layer_status` | Когда | Оркестратор |
|---|---|---|
| `ok` | Контракт прочитан; модель off или ответ прошёл ворота | Исход/действие и коды — результат слоя |
| `degraded` | Ядро посчитано; сбой/отбраковка модели | **Исход ядра сохранён**, есть `failure_reason` |
| `failed` | Нечитаемый контракт/запрос или `config_invalid` | `evaluate` → `reject_insufficient_kb` без кодов пробела; `fact_check` → `refuse` без находок о пробеле |

При `ok` поле `failure_reason` = `null`; при любом другом статусе — задано.

`failure_reason`: `contract_invalid`, `config_invalid`, `model_timeout`, `model_network`, `model_invalid_json`, `model_schema`, `model_unconfigured`.

Сбой модели **не** даёт тихий `allow_answer` / `accept`. Ответ модели при нарушении схемы отбрасывается **целиком** (частичное слияние полей нет). При `citation_check is None` ядро уже даёт `rewrite` — деградация модели его не снимает.

Исключения наружу не пробрасываются.

### Журнал (без ключа и без `Authorization`)

| Сбой | Статус | Логгер |
|---|---|---|
| `config_invalid`, `contract_invalid` | `failed` | `decision.coverage` / `decision.factcheck` — причина, без ключа |
| `model_unconfigured` | `degraded` | Запрос не уходит; ядро сохранено |
| `model_timeout`, `model_network` | `degraded` | `decision.llm` — тип/HTTP, без ключа |
| `model_invalid_json`, `model_schema` | `degraded` | `decision.llm`; ядро сохранено |

---

## Пример вызова

```python
from decision import CoverageRequest, DialogueTurn, evaluate, fact_check, FactCheckInput

rec = evaluate(
    CoverageRequest(
        question="…",
        dialogue=[DialogueTurn(role="user", content="…")],
        retrieval=retrieval,
        question_kind=None,
        policy=None,
    )
)

if rec.suggested_decision == "allow_answer":
    verdict = fact_check(
        FactCheckInput(
            question="…",
            answer=draft_text,
            retrieval=retrieval,
            citation_check=citation_check,
            claims=None,
        )
    )
elif rec.suggested_decision == "clarify":
    # missing_fields + candidates → один вопрос
    ...
```

---

## Тесты

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest --ignore=tests/live
```

Обязательный набор без сети и без чтения ключей. Зависимости тестов (`pytest`, `pydantic`, `httpx`) — в `requirements-dev.txt`, не в `requirements.txt`.

Живой набор: `pytest tests/live` (нужен `DEEPSEEK_API_KEY` в окружении процесса; значение не вписывать в репозиторий и команды). В `tests/live/` — несколько сценариев выдачи и черновиков, форма запроса к модели, деградация при отключённом ядре до LLM.

---

## Кратко: чего слой не делает

- Текст для юриста, вопрос, отказ, правка черновика.
- Повторная проверка существования цитаты; правки `origin/guard` и `solution.py`.
- Чтение бюджета правок и подсчёт уже сделанных правок.