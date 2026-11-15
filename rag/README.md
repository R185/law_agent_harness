# RAG-подсистема юридического ассистента ООО «Меридиан»

Данный каталог представляет собой автономный пакет **`rag`** — модуль извлечения знаний (Knowledge-Only RAG) в составе ИИ-ассистента для юристов.

---

## 1. Зона ответственности модуля

Модуль отвечает **исключительно за поиск, индексацию и структурирование контекста (Retrieval & Context)** из внутренней базы знаний (договоры компании, регламенты, служебные записки, допсоглашения и нормативно-правовые акты РФ).

* **Входит в зону RAG:**
  * Загрузка и структурный чанкинг документов компании и законов ([chunker.py](chunker.py), [loader.py](loader.py));
  * Извлечение метаданных, построение графа связей (`DocGraph`) и выбор действующих редакций нормативных актов на дату события ([metadata.py](metadata.py));
  * Гибридный поиск: BM25Okapi + плотные векторные эмбеддинги (`baai/bge-m3`) + RRF + реранкер ([retrieve.py](retrieve.py), [bm25.py](bm25.py), [index.py](index.py));
  * Строгая дискриминация похожих контрагентов через `EntityResolver` (например, ООО «Альфа-Трейд» vs ООО «Альфа-Трейдинг»);
  * Сборка форматированного контекста в пределах лимита токенов ([tools.py](tools.py), [context.py](context.py));
  * Проверка цитат и разрешение ссылок ([tools.py](tools.py));
  * Экспорт типизированного контракта и единой точки входа для внешних модулей ([__init__.py](__init__.py), [pipeline.py](pipeline.py), [schema.py](schema.py)).

* **НЕ входит в зону RAG (делегировано другим подсистемам):**
  * **Безопасность и защита от Prompt Injection** — зона модуля `guard` (`Guard.sanitize_document`, `Guard.check_answer`);
  * **Оценка достаточности информации и уточняющие вопросы** — зона модуля `request-questions` (`harness.evaluate_insufficient_info`, `build_clarification`);
  * **Служебный регистр правовой справки и детерминированная чистка чат-артефактов** — зона модуля `tone` (`rag/style.py`, `enforce_tone`, `audit_hard`);
  * **Сквозной пайплайн и генерация ответа через LLM** — зона центрального оркестратора проекта.

---

## 2. Публичный API (Ручки RAG)

Основная точка входа экспортируется напрямую из пакета `rag`:

```python
from rag import retrieve, build_context

# 1. Запуск гибридного поиска
result = retrieve(
    question="Какая арендная плата по договору с ООО «Северный Склад»?",
    history=None,   # Опционально: строка истории диалога ("Пользователь: ...\nАссистент: ...")
    top_k=10,       # Количество извлекаемых кандидатов
)

# 2. Передача чанков в модуль судьи (ветка request-questions):
# result.chunks возвращает list[RetrievedChunk], совместимый с harness.types.JudgeInput
judge_chunks = result.chunks  

# 3. Сборка текстового контекста для промпта LLM:
context_result = build_context(result, max_chars=12000)
prompt_context = context_result.text

# 4. Проверка сигналов поиска:
if result.signals.low_confidence:
    print("В базе знаний нет уверенных данных по вопросу")
```

### Дополнительные инструменты контракта

```python
from rag import get_document, get_clause, resolve_reference, verify_citations

# Получение метаданных документа по ID
doc = get_document("А-08-2024")

# Извлечение конкретного пункта нормы или договора
clause = get_clause("А-08-2024", "п. 3.1")

# Распознавание ссылки на норму или договор
ref = resolve_reference("ст. 70 ТК РФ")

# Проверка корректности юридических цитат в ответе
check = verify_citations("Согласно Договору поставки № П-17/2024 пункт 4.2...")
if not check.valid:
    print("Несуществующие документы:", check.unknown_docs)
```

### Интеграция с препроцессором / санитизатором (G0 для оркестратора)

Для обезвреживания атак, внедрённых в документы базы знаний (контрольная точка G0), `RAGPipeline` поддерживает параметр `preprocessor`:

```python
from guard import Guard
from rag import RAGPipeline

guard = Guard()
# Оркестратор передаёт метод санитизации в RAG при инициализации:
pipeline = RAGPipeline(preprocessor=guard.sanitize_document)
```

*Примечание:* `RAGPipeline` автоматически обнаруживает модуль `guard` (если ветка `guard` смержена в проект) и подключает `Guard().sanitize_document` по умолчанию. Явный `preprocessor=...` позволяет оркестратору переопределить логику или отключить её (`preprocessor=False`).

---

## 3. Архитектура интеграции в общий пайплайн

```
[Вопрос пользователя + история]
             │
             ▼
 1. guard.inspect_question()        (Ветка `guard`: проверка безопасности G1)
             │
             ▼
 2. rag.retrieve()                  (Ветка `RAG`: гибридный поиск фактов)
             │
             ├──► result.chunks ───► harness.evaluate_insufficient_info() 
             │                       (Ветка `request-questions`: достаточно ли данных?)
             ▼
 3. rag.build_context()             (Ветка `RAG`: упаковка фрагментов в карточки)
             │
             ▼
 4. guard.scan_context()            (Ветка `guard`: проверка контекста G2)
             │
             ▼
 5. LLM Генерация (Guard G3 + Tone STYLE_CONTRACT)
             │
             ▼
 6. tone.enforce_tone()             (Ветка `tone`: чистка артефактов и audit_hard)
             │
             ▼
 7. guard.check_answer()            (Ветка `guard`: проверка финального ответа G4)
             │
             ▼
[Финальный ответ]
```

---

## 4. Состав модулей пакета `rag/`

* [__init__.py](__init__.py) — публичные ручки и строгая группировка экспортируемых символов.
* [pipeline.py](pipeline.py) — класс `RAGPipeline` и модульные точки входа с подписанными импортами.
* [schema.py](schema.py) — типизированные структуры контракта (`RetrievalResult`, `DocumentMatch`, `Evidence`, `Signals`, `RetrievedChunk`).
* [retrieve.py](retrieve.py) — гибридный поиск (BM25 + Dense + RRF + реранкер) и `EntityResolver`.
* [metadata.py](metadata.py) — граф связей `DocGraph`, извлечение дат, учёт действующих редакций и допсоглашений.
* [chunker.py](chunker.py) — синтаксический и семантический чанкинг договоров и статей законов.
* [bm25.py](bm25.py) — автономная реализация BM25Okapi без внешних зависимостей.
* [query.py](query.py) — извлечение дат событий, нормализация сущностей и переписывание поисковых запросов.
* [tools.py](tools.py) — сборка контекстных карточек, детекция неоднозначностей и сверка цитат.
* [eval.py](eval.py) — валидационный скрипт оценки полноты извлечения (Recall@5 = 100%).
* [build_index.py](build_index.py) — утилита построения и сериализации индекса базы знаний.
* [loader.py](loader.py) — загрузка сырых документов из `knowledge-base/` и zip-архивов.
* [api.py](api.py) — отказоустойчивый HTTP-клиент для вызовов эмбеддера и реранкера платформы.
* [config.py](config.py) — чтение переменных окружения платформы и порогов поиска.
* [text.py](text.py) — нормализация пробелов, стемминг и канонизация номеров договоров.

---

## 5. Тестирование и валидация

```bash
# 1. Запуск юнит-тестов контракта RAG (8 тестов, 100% pass)
python3 -m unittest discover -s tests

# 2. Оценка качества поиска на эталонном наборе (100% Recall@5)
python3 rag/eval.py
# или
python3 -m rag.eval
```
