"""Оркестратор: guard G1 -> RAG -> судья покрытия -> guard G2/G3 -> LLM -> guard G4 + tone -> ответ.

RAG отвечает за поиск и контекст, tone — за регистр, guard — за защиту,
decision — за оценку покрытия. Все слои только сообщают; решение и тексты
отказа — здесь. Шаблон вместо ответа выдаётся только там, где показывать
нечего: запрос без вопроса по базе (guard decline), пустой ответ модели или
утечка служебных инструкций. Во всех остальных случаях отвечает модель,
а находки слоёв передаются ей как указания.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from rag.api import ApiClient
from rag.pipeline import RAGPipeline
from rag.schema import ContextResult, RetrievalResult

import tone
from guard import DECLINE, PASS, REPLACE, Guard, llm_from_env
from orchestrator import config


WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

ANSWER_NOTE = "Дай ответ строго по фрагментам."
REWRITE_NOTE = (
    "Проверка ответа не пройдена. Исправь замечания:\n{issues}\n"
    "Дай исправленный ответ на тот же вопрос строго по фрагментам базы знаний. "
    "Не добавляй новых фактов и сведений, о которых не спрашивали."
)
NO_CONTEXT = "Фрагменты по вопросу в базе знаний не найдены."

_REJECT_REASONS = {
    "topic_mismatch": "тема вопроса не найдена в базе",
    "insufficient_relevance": "найденные фрагменты слабо связаны с вопросом",
    "low_confidence": "поиск не уверен в найденных фрагментах",
    "named_not_found": "названные в вопросе документы или нормы в базе не найдены",
    "missing_facts": "во фрагментах не хватает фактов",
    "edition_missing": "нужной редакции документа в базе нет",
    "date_conflict": "дата события не совпадает с периодом действия документа",
    "reference_error": "ссылка в вопросе не разрешается по базе",
    "ambiguous_edition": "подходят несколько редакций документа",
}


def _words(text: str) -> set[str]:
    return {w.lower().replace("ё", "е") for w in WORD_RE.findall(text or "")}


def _one_line(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _short(text: object, limit: int = 120) -> str:
    flat = _one_line(str(text or ""))
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class Orchestrator:
    """Собирает ответ на один вопрос и не зависит от CSV/UI. Потокобезопасен."""

    def __init__(self, source_path: Path, offline: bool | None = None, deadline: float | None = None):
        self.offline = config.OFFLINE if offline is None else offline
        self.client = None if self.offline else ApiClient()
        self.deadline = deadline or (time.monotonic() + config.TIME_BUDGET_SEC)
        self.started = self.deadline - config.TIME_BUDGET_SEC
        guard_llm = llm_from_env() if (config.GUARD_LLM and not self.offline) else None
        self.guard = Guard(llm=guard_llm, llm_mode=config.GUARD_LLM_MODE)
        self.rag = RAGPipeline(source_path=source_path, preprocessor=self.guard.sanitize_document)  # G0
        self._judge_enabled = config.ENABLE_JUDGE
        if config.ENABLE_DENSE and not self.offline and self.client is not None:
            self._build_dense_index()

    # ------------------------------------------------------------ служебное
    def _build_dense_index(self) -> None:
        """Считать embeddings из платформенной базы один раз на старте."""
        if self.rag.index.embeddings is not None:
            return
        if not config.EMBED_KEY:
            print("[orchestrator] EMBEDDER_API_KEY не задан; dense выключен", flush=True)
            return
        try:
            self.rag.index.ensure_embeddings(self.client)
        except Exception as exc:
            print(f"[orchestrator] dense index не собрался ({type(exc).__name__}); BM25 only", flush=True)
            return
        if self.rag.index.embeddings is None:
            print("[orchestrator] dense index пуст; BM25 only", flush=True)
        else:
            print(f"[orchestrator] dense index: {len(self.rag.index.embeddings)} векторов", flush=True)

    def _time_left(self) -> float:
        return self.deadline - time.monotonic()

    def _rewrite_allowed(self) -> bool:
        return time.monotonic() - self.started < config.TIME_BUDGET_SEC * config.REWRITE_CUTOFF

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------ шаги
    def _retrieve(self, question: str, history: str | None = None) -> RetrievalResult:
        client = self.client if self._time_left() > 0 else None
        try:
            return self.rag.retrieve(question, history=history, top_k=config.TOP_K, client=client)
        except Exception as exc:
            if client is None:
                raise
            print(f"[orchestrator] retrieval с API не удался ({type(exc).__name__}); повтор без API", flush=True)
            return self.rag.retrieve(question, history=history, top_k=config.TOP_K, client=None)

    def _context(self, result: RetrievalResult) -> ContextResult:
        return self.rag.build_context(result, max_chars=config.CONTEXT_MAX_CHARS)

    def _scan_context(self, context: ContextResult, tag: str) -> str:
        """G2: вырезать инструкции, пережившие очистку документов."""
        scanned = self.guard.scan_context(context.text)
        if scanned.removed:
            print(f"{tag} guard G2: вырезано {len(scanned.removed)} фрагментов", flush=True)
        return scanned.text.strip()

    def _judge_notes(self, question: str, result: RetrievalResult, tag: str) -> list[str]:
        """Судья покрытия как подсказка генерации, а не фильтр: его вывод уходит модели указанием."""
        if not self._judge_enabled:
            return []
        try:
            from decision import CoverageRequest, evaluate
            from orchestrator.judge_adapter import to_contract
        except ImportError as exc:
            self._judge_enabled = False
            print(f"[orchestrator] judge недоступен ({exc}); проверка покрытия выключена", flush=True)
            return []
        try:
            rec = evaluate(CoverageRequest(question=question, retrieval=to_contract(result)))
        except Exception as exc:
            print(f"{tag} judge упал ({type(exc).__name__}); ответ без подсказки", flush=True)
            return []
        if rec.layer_status.value == "failed":
            return []

        notes: list[str] = []
        decision = rec.suggested_decision.value
        codes = [code.value for code in rec.reason_codes]
        if decision != "allow_answer":
            print(f"{tag} judge: {decision} ({','.join(codes)}) -> подсказка модели", flush=True)
        if decision == "reject_insufficient_kb":
            reasons = "; ".join(dict.fromkeys(_REJECT_REASONS[c] for c in codes if c in _REJECT_REASONS))
            missing = result.unresolved.contract_numbers + result.unresolved.party_names + result.unresolved.law_articles
            detail = f" В базе не найдены: {', '.join(_short(m, 40) for m in missing[:4])}." if missing else ""
            notes.append(
                "Проверка покрытия: во фрагментах, вероятно, нет ответа на вопрос"
                + (f" ({reasons})" if reasons else "") + "." + detail
                + " Если это так, одной фразой скажи, что в предоставленной базе знаний отсутствуют документы "
                "о теме вопроса, и назови тему. Если ответ во фрагментах всё же есть, ответь по ним."
            )
        elif decision == "reject_partial_coverage":
            gaps = ", ".join(_short(g.detail, 40) for g in rec.gaps[:4])
            notes.append(
                "Проверка покрытия: фрагменты отвечают на вопрос не полностью"
                + (f" (нет: {gaps})" if gaps else "")
                + ". Ответь на ту часть, что есть, и одной фразой укажи, чего в базе не хватает."
            )
        elif decision == "clarify":
            labels = [str(c.label or c.title or c.doc_number) for c in rec.candidates if (c.label or c.title or c.doc_number)]
            notes.append(
                "Документ или контрагент в вопросе указан неоднозначно"
                + (f"; подходят: {'; '.join(_short(l, 60) for l in dict.fromkeys(labels[:3]))}" if labels else "")
                + ". Не задавай встречный вопрос: кратко ответь по каждому подходящему документу, называя его."
            )
        for err in rec.premise_errors[:3]:
            expected = f" В документе: «{_short(err.expected, 80)}»." if err.expected else ""
            notes.append(
                f"В вопросе указано «{_short(err.actual, 80)}», документы этого не подтверждают.{expected} "
                "Прямо укажи пользователю, что сказано в документе."
            )
        return notes

    @staticmethod
    def _date_note(result: RetrievalResult) -> list[str]:
        event_date = result.signals.event_date
        if not event_date:
            return []
        applicable = [d.title for d in result.documents if d.applicable_edition][:2]
        note = f"Дата события из вопроса: {event_date}. Применяй редакцию документа, действовавшую на эту дату"
        return [note + (f" ({'; '.join(applicable)})." if applicable else ".")]

    def _system_prompt(self) -> str:
        return tone.build_system_prompt(config.BASE_PROMPT)

    def _llm(self, messages: list[dict[str, str]]) -> str:
        """Вызов модели с ограниченными повторами: ошибка прокси или пустой ответ — ещё попытка, пока есть время."""
        if self.client is None:
            raise RuntimeError("LLM недоступна: нет API_BASE/LLM_API_KEY")
        last_exc: Exception | None = None
        for attempt in range(config.LLM_RETRIES + 1):
            try:
                text = self.client.chat_completion(
                    messages, temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)
                if text.strip():
                    return text
                print("[orchestrator] модель вернула пустой ответ; повтор", flush=True)
            except Exception as exc:  # 429/5xx после повторов ApiClient, таймаут сети
                last_exc = exc
                print(f"[orchestrator] вызов модели не удался ({type(exc).__name__}); повтор", flush=True)
            pause = 5.0 * (attempt + 1)
            if attempt == config.LLM_RETRIES or self._time_left() < pause + 30:
                break
            time.sleep(pause)
        if last_exc is not None:
            raise last_exc
        return ""

    def _check(self, answer: str, context: str, verdict):
        """Все проверки ответа вместе: guard G4 и tone. Возвращает (проверка guard, ответ, замечания)."""
        # strict_refs: ссылка на пункт, статью или договор, которых нет во фрагментах, — повод переписать
        # (на эталонных ответах ложных срабатываний нет; выдуманная ссылка обнуляет вопрос)
        check = self.guard.check_answer(answer, context, strict_refs=True, question=verdict)
        answer = check.clean_answer
        issues = [h.fix + (f" (фрагмент: «{h.quote}»)" if h.quote else "") for h in check.rewrite_hints]
        for v in tone.hard_only(tone.check(answer, context)):
            issues.append(f"{v.code}: {v.hint}" + (f" (фрагмент: «{v.quote}»)" if v.quote else ""))
        return check, answer, issues

    def _extractive(self, question: str, result: RetrievalResult, context: ContextResult) -> str:
        """Локальный fallback без LLM: берёт наиболее близкий фрагмент."""
        evidence = context.evidence or result.evidence
        if not evidence:
            return config.FALLBACK

        qwords = _words(question)
        best = max(
            evidence,
            key=lambda ev: len(qwords & _words(ev.title + " " + ev.text)),
        )
        sentences = [s.strip() for s in SENTENCE_RE.split(best.text) if s.strip()]
        chosen: list[str] = []
        for sentence in sentences:
            if qwords & _words(sentence):
                chosen.append(sentence)
            if len(" ".join(chosen)) >= 500:
                break
        if not chosen:
            chosen = sentences[:2]
        body = " ".join(chosen)[:700].strip()
        if not body:
            return config.FALLBACK
        source = best.title
        if best.clause_ids:
            source += f", п. {', '.join(best.clause_ids)}"
        return _one_line(f"{body} Источник: {source}.")

    # ------------------------------------------------------------ ответ
    def answer(self, question: str, qid: str = "") -> str:
        tag = f"[orchestrator] {qid}:" if qid else "[orchestrator]"
        verdict = None
        clean_question, history = question, None

        if config.ENABLE_G1:
            verdict = self.guard.inspect_question(question)  # G1
            if verdict.action == DECLINE:
                print(f"{tag} guard G1 decline ({verdict.decline_reason})", flush=True)
                return config.DECLINE_TEXTS.get(verdict.decline_reason, config.FALLBACK)
            if verdict.action != "allow":
                # sanitize — атака вырезана; refuse_assist — отвечаем по базе без помощи в нарушении.
                # Указания для модели guard кладёт в verdict.directives, build_messages их добавит.
                print(f"{tag} guard G1 {verdict.action} ({','.join(verdict.categories)})", flush=True)
            if verdict.degraded:
                print(f"{tag} guard G1 degraded: проверка вопроса не выполнена", flush=True)
            clean_question = (verdict.clean_question or question).strip()
            history = verdict.history_text or None

        result = self._retrieve(clean_question, history=history)
        notes = self._judge_notes(clean_question, result, tag) + self._date_note(result)
        context = self._context(result)
        context_text = self._scan_context(context, tag) or NO_CONTEXT

        if self.offline or self._time_left() <= 0:
            if not self.offline:
                print(f"{tag} бюджет времени исчерпан; ответ без модели", flush=True)
            return _one_line(self._extractive(clean_question, result, context))

        extra = "\n".join([ANSWER_NOTE] + notes)
        messages = self.guard.build_messages(  # G3: правила безопасности, указания, данные в метках
            self._system_prompt(), verdict if verdict is not None else clean_question, context_text,
            extra_note=extra)
        answer = tone.normalize(self._llm(messages)).text
        check, answer, issues = self._check(answer, context_text, verdict or clean_question)

        for _ in range(config.REWRITE_LIMIT):
            if not issues or check.action == REPLACE:
                break
            if not self._rewrite_allowed():
                print(f"{tag} переписывание пропущено: мало времени", flush=True)
                break
            # переписываем в том же защищённом диалоге: правила guard и контракт tone остаются в силе
            raw = self._llm(messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": REWRITE_NOTE.format(issues="\n".join("- " + i for i in issues))},
            ])
            answer = tone.normalize(raw).text
            check, answer, issues = self._check(answer, context_text, verdict or clean_question)

        if issues:
            print(f"{tag} остались замечания: {'; '.join(i[:60] for i in issues)}", flush=True)
        if check.action != PASS and check.fallback_reason:
            # пустой ответ или утечка служебных инструкций: такой текст не показываем
            print(f"{tag} guard G4 fallback ({check.fallback_reason})", flush=True)
            return config.SAFE_FALLBACK
        return _one_line(answer) or config.SAFE_FALLBACK
