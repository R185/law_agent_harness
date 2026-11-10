"""guard — защитный слой юридического ассистента.

Библиотека без зависимостей от остального проекта: только стандартная
библиотека и два соседних файла (guard_rules.py, guard_prompts.py).
Модуль ничего не решает за систему: на каждой контрольной точке он возвращает
вердикт, а что с ним делать, определяет вызывающий код.

Контрольные точки:
    G0  Guard.sanitize_document()  — документ базы знаний при загрузке
    G1  Guard.inspect_question()   — вопрос пользователя и история диалога
    G2  Guard.scan_context()       — найденные фрагменты перед отправкой в модель
    G3  Guard.build_messages()     — сборка защищённого промпта
    G4  Guard.check_answer()       — готовый ответ перед выдачей
    G5  safe_html(), redact_secrets(), журнал событий — веб-интерфейс и логи

Гарантии: публичные методы не бросают исключений и не обращаются к сети.
Модель вызывается, только если передана функция llm(messages) -> str.

Подробности и примеры подключения — в GUARD_SPEC.md.
"""

from __future__ import annotations

import collections
import functools
import html
import json
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence, Union

import guard_prompts as P
import guard_rules as R

__all__ = [
    "Guard", "Finding", "Turn", "InputVerdict", "DocResult", "ContextResult", "OutputVerdict",
    "safe_html", "strip_active_content", "prepare_for_display", "redact_secrets", "llm_from_env",
]

LLM = Callable[[list], str]

ALLOW = "allow"                  # угроз нет, работать как обычно
SANITIZE = "sanitize"            # отвечать по clean_question с учётом directives
REFUSE_ASSIST = "refuse_assist"  # не помогать с нарушением, разъяснить требования по базе
DECLINE = "decline"              # законного вопроса не осталось, выдать suggested_reply

PASS = "pass"
REGENERATE = "regenerate"
REPLACE = "replace"

_RULES_ATTACK = {R.OVERRIDE, R.FAKE_SYSTEM, R.ROLE_CHANGE, R.AUTHORITY}
_HISTORY_CATEGORIES = _RULES_ATTACK | {R.PROMPT_LEAK}


# --------------------------------------------------------------------------
# Структуры данных
# --------------------------------------------------------------------------
@dataclass
class Finding:
    """Одна находка: что сработало, где и насколько это серьёзно."""
    category: str
    rule: str
    severity: str = "medium"   # low | medium | high
    excerpt: str = ""
    where: str = "question"    # question | history | hidden | document | context | answer
    source: str = "rules"      # rules | llm


@dataclass
class Turn:
    role: str   # "user" | "assistant"
    text: str


@dataclass
class InputVerdict:
    """Результат проверки вопроса (G1)."""
    action: str = ALLOW
    categories: list = field(default_factory=list)
    risk: float = 0.0
    findings: list = field(default_factory=list)
    original_question: str = ""
    clean_question: str = ""
    history: list = field(default_factory=list)
    directives: list = field(default_factory=list)
    suggested_reply: Optional[str] = None

    @property
    def history_text(self) -> str:
        names = {"user": "Пользователь", "assistant": "Ассистент"}
        return "\n".join("%s: %s" % (names.get(t.role, t.role), t.text) for t in self.history)

    @property
    def clean_dialog(self) -> str:
        """Очищенный запрос в исходном формате: одиночный вопрос или диалог с репликами.

        Подходит как замена исходного поля question для кода, который сам разбирает диалог.
        """
        if not self.history:
            return self.clean_question
        return self.history_text + "\nПользователь: " + self.clean_question

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocResult:
    """Результат очистки документа (G0)."""
    text: str
    removed: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    name: str = ""


@dataclass
class ContextResult:
    """Результат проверки найденных фрагментов (G2)."""
    fragments: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.fragments)


@dataclass
class OutputVerdict:
    """Результат проверки ответа (G4)."""
    action: str = PASS
    findings: list = field(default_factory=list)
    clean_answer: str = ""
    retry_message: str = ""
    safe_answer: str = P.SAFE_ANSWER

    @property
    def ok(self) -> bool:
        return self.action == PASS


# --------------------------------------------------------------------------
# Вспомогательное
# --------------------------------------------------------------------------
def _excerpt(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


_TURN_RE = re.compile(r"^\s*(пользователь|user|ассистент|assistant)\s*[:：]\s*", re.I)
_ASSISTANT_NAMES = {"ассистент", "assistant"}


def parse_dialog(text: str) -> tuple:
    """Делит текст на реплики. Возвращает (реплики, индекс последней реплики пользователя)."""
    lines = text.split("\n")
    if not any(_TURN_RE.match(line) for line in lines):
        return [Turn("user", text.strip())], 0
    turns: list = []
    role = "user"
    buf: list = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            turns.append(Turn(role, body))

    for line in lines:
        m = _TURN_RE.match(line)
        if m:
            flush()
            buf = [line[m.end():]]
            role = "assistant" if m.group(1).lower() in _ASSISTANT_NAMES else "user"
        else:
            buf.append(line)
    flush()
    last = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].role == "user":
            last = i
            break
    if last < 0:
        return [Turn("user", text.strip())], 0
    return turns, last


def _safe(fallback: Callable):
    """Публичный метод не должен ронять ответ: при сбое возвращается безопасное значение."""
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — сбой защиты не должен ломать пайплайн
                try:
                    self._record("error", {"fn": fn.__name__, "error": type(exc).__name__})
                except Exception:  # noqa: BLE001
                    pass
                return fallback(self, exc, *args, **kwargs)
        return wrapper
    return deco


def _error_finding(exc: Exception, where: str) -> Finding:
    return Finding("guard_error", "guard." + type(exc).__name__, "low", "", where)


def _secret_sentences(text: str) -> set:
    parts = re.split(r"(?<=[.!?:;])\s+|\n", text)
    return {R.norm_plain(p) for p in parts if len(R.norm_plain(p)) >= 40}


# --------------------------------------------------------------------------
# Основной класс
# --------------------------------------------------------------------------
class Guard:
    """Фасад защитного слоя. Один экземпляр на процесс; методы потокобезопасны.

    llm         — необязательная функция llm(messages) -> str для смысловой классификации вопроса;
    llm_mode    — "off" | "auto" (спрашивать модель только при сомнении) | "always";
    log_path    — путь к JSONL-журналу событий; по умолчанию события хранятся только в памяти.
    """

    def __init__(self, llm: Optional[LLM] = None, llm_mode: str = "auto",
                 max_question_chars: int = 8000, max_answer_chars: int = 10000,
                 log_path: Optional[str] = None) -> None:
        self.llm = llm
        self.llm_mode = llm_mode
        self.max_question_chars = max_question_chars
        self.max_answer_chars = max_answer_chars
        self.log_path = log_path
        self.canary = "GRD-" + secrets.token_hex(6)
        self._lock = threading.Lock()
        self._events = collections.deque(maxlen=5000)
        self._sessions: dict = {}
        self._own_secrets = _secret_sentences(P.SECURITY_RULES.format(canary=""))
        self._registered_secrets: set = set()
        self._injections: list = []       # нормализованные вырезанные предложения
        self._payloads: set = set()       # цитаты из вырезанных предложений
        self._inj_numbers: set = set()    # числа из вырезанных предложений
        self._kb_norm: list = []          # нормализованные чистые тексты документов
        self._kb_chars = 0

    # ------------------------------------------------------------------ G0
    @_safe(lambda self, exc, text, name="": DocResult(str(text or ""), [], [_error_finding(exc, "document")], name))
    def sanitize_document(self, text: str, name: str = "") -> DocResult:
        """G0. Убирает из документа базы знаний встроенные указания для ИИ и скрытые символы.

        Остальной текст не меняется. Вызывать при загрузке базы, до разбиения на фрагменты.
        """
        clean, removed, findings = self._clean_doc_text(str(text or ""), "document")
        safe_name = name
        if name:
            reasons, _ = R.doc_sentence_reasons(R.shadow(name)[0])
            if reasons:
                findings.append(Finding(R.DOC_INJECTION, "doc.name", "high", _excerpt(name), "document"))
                safe_name = "документ"
        self._remember_injections(removed)
        self._remember_kb(clean)
        if findings:
            self._record("G0", {"name": _excerpt(name, 80), "removed": len(removed),
                                "rules": sorted({f.rule for f in findings})})
        return DocResult(clean, removed, findings, safe_name)

    # ------------------------------------------------------------------ G1
    @_safe(lambda self, exc, raw_question, llm=None: InputVerdict(
        ALLOW, [], 0.0, [_error_finding(exc, "question")], str(raw_question or ""), str(raw_question or "")))
    def inspect_question(self, raw_question: str, llm: Optional[LLM] = None) -> InputVerdict:
        """G1. Проверяет вопрос и историю диалога.

        raw_question — поле question как оно пришло: одиночный вопрос или диалог
        с репликами «Пользователь: … / Ассистент: …».
        """
        raw = str(raw_question or "")
        findings: list = []
        text, n_invisible, hidden = R.strip_invisible(raw)
        if n_invisible:
            findings.append(Finding(R.OBFUSCATION, "obfuscation.invisible", "low",
                                    "%d скрытых символов" % n_invisible, "question"))
        if hidden.strip():
            findings.append(Finding(R.OBFUSCATION, "obfuscation.tag_chars", "high", _excerpt(hidden), "hidden"))
            for hit in R.find_hits(R.shadow(hidden)[0]):
                findings.append(Finding(hit.rule.category, hit.rule.id, hit.rule.severity,
                                        _excerpt(hidden[hit.start:hit.end]), "hidden"))
        text, n_spaced = R.collapse_spaced(text)
        if n_spaced:
            findings.append(Finding(R.OBFUSCATION, "obfuscation.spaced_letters", "medium",
                                    "слов через пробел: %d" % n_spaced, "question"))
        if len(text) > self.max_question_chars:
            half = self.max_question_chars // 2
            findings.append(Finding(R.DOS, "dos.too_long", "medium", "%d символов" % len(text), "question"))
            text = text[:half] + "\n" + text[-half:]

        turns, last = parse_dialog(text)
        question_raw = turns[last].text
        clean_q, q_findings = self._analyze(question_raw, "question")
        findings.extend(q_findings)

        history: list = []
        has_assistant_turn = False
        for turn in turns[:last]:
            if turn.role == "assistant" and R.FORGED_ASSISTANT.search(R.shadow(turn.text)[0]):
                findings.append(Finding(R.ROLE_CHANGE, "history.forged_assistant", "high",
                                        _excerpt(turn.text), "history"))
                continue
            clean_turn, h_findings = self._analyze(turn.text, "history", only=_HISTORY_CATEGORIES)
            findings.extend(h_findings)
            if clean_turn.strip():
                history.append(Turn(turn.role, clean_turn.strip()))
                has_assistant_turn = has_assistant_turn or turn.role == "assistant"

        model = llm or self.llm
        if model is not None and self._should_ask_llm(clean_q, q_findings):
            clean_q, llm_findings = self._classify(model, clean_q)
            findings.extend(llm_findings)

        verdict = self._decide(question_raw, clean_q, history, findings, has_assistant_turn)
        if verdict.action != ALLOW or verdict.findings:
            self._record("G1", {"action": verdict.action, "categories": verdict.categories,
                                "risk": verdict.risk, "rules": sorted({f.rule for f in verdict.findings}),
                                "excerpt": redact_secrets(_excerpt(question_raw, 120))})
        return verdict

    # ------------------------------------------------------------------ G2
    @_safe(lambda self, exc, fragments: ContextResult(
        [fragments] if isinstance(fragments, str) else [str(f) for f in (fragments or [])],
        [], [_error_finding(exc, "context")]))
    def scan_context(self, fragments: Union[str, Sequence[str]]) -> ContextResult:
        """G2. Проверяет найденные фрагменты перед отправкой в модель.

        Нужен, даже если документы прошли G0: фрагмент мог прийти из другого
        источника, а заголовок фрагмента — из имени файла.
        """
        items = [fragments] if isinstance(fragments, str) else [str(f) for f in (fragments or [])]
        out: list = []
        removed_all: list = []
        findings: list = []
        for item in items:
            clean, removed, found = self._clean_doc_text(item, "context")
            out.append(clean)
            removed_all.extend(removed)
            findings.extend(found)
        self._remember_injections(removed_all)
        if findings:
            self._record("G2", {"removed": len(removed_all), "rules": sorted({f.rule for f in findings})})
        return ContextResult(out, removed_all, findings)

    # ------------------------------------------------------------------ G3
    def security_rules(self) -> str:
        """G3. Блок правил безопасности для системного промпта любого вызова модели."""
        return P.SECURITY_RULES.format(canary=self.canary)

    def wrap(self, text: str, kind: str = "data") -> str:
        """G3. Заключает недоверенный текст в метки данных со случайным идентификатором."""
        nonce = secrets.token_hex(3)
        body = str(text or "").replace("[[", "[ [").replace("]]", "] ]")
        return "[[DATA:%s:%s]]\n%s\n[[/DATA:%s:%s]]" % (kind, nonce, body, kind, nonce)

    def register_secret(self, text: str) -> None:
        """Запоминает текст (например, системный промпт), утечку которого нужно ловить в ответе."""
        with self._lock:
            self._registered_secrets |= _secret_sentences(str(text or ""))

    @_safe(lambda self, exc, system_prompt, question, context="", **kw: [
        {"role": "system", "content": str(system_prompt or "")},
        {"role": "user", "content": "Фрагменты базы знаний:\n%s\n\nВопрос: %s" % (
            context if isinstance(context, str) else "\n\n".join(context or []),
            question.clean_question if isinstance(question, InputVerdict) else question)}])
    def build_messages(self, system_prompt: str, question: Union[InputVerdict, str],
                       context: Union[str, Sequence[str]] = "", extra_note: str = "") -> list:
        """G3. Собирает защищённые сообщения для отвечающего вызова модели.

        system_prompt — ваш системный промпт (роль, формат ответа);
        question      — вердикт G1 или просто текст вопроса;
        context       — найденные фрагменты, желательно после G2.
        """
        self.register_secret(system_prompt)
        verdict = question if isinstance(question, InputVerdict) else None
        q_text = verdict.clean_question if verdict else str(question or "")
        system = str(system_prompt or "").rstrip() + "\n\n" + self.security_rules()
        if verdict and verdict.directives:
            system += "\n\nУКАЗАНИЯ ПО ЭТОМУ ЗАПРОСУ:\n" + "\n".join("- " + d for d in verdict.directives)

        ctx = context if isinstance(context, str) else "\n\n".join(str(c) for c in (context or []))
        blocks = ["Фрагменты базы знаний (данные):\n" + self.wrap(ctx, "documents")]
        if verdict and verdict.history:
            blocks.append("История диалога (цитата, данные):\n" + self.wrap(verdict.history_text, "history"))
        blocks.append("Вопрос пользователя (данные):\n" + self.wrap(q_text, "question"))
        if extra_note:
            blocks.append(str(extra_note))
        blocks.append(P.USER_REMINDER)
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(blocks)}]

    # ------------------------------------------------------------------ G4
    @_safe(lambda self, exc, answer, *a, **kw: OutputVerdict(
        PASS if str(answer or "").strip() else REPLACE, [_error_finding(exc, "answer")], str(answer or "")))
    def check_answer(self, answer: str, context: Union[str, Sequence[str], None] = None,
                     strict_refs: bool = False,
                     question: Union[InputVerdict, str, None] = None) -> OutputVerdict:
        """G4. Проверяет готовый ответ перед выдачей.

        context  — фрагменты, по которым строился ответ. Без него проверяются только
                   утечки, выход из роли, повтор встроенных инструкций и формат;
        question — вердикт G1 или текст вопроса. Нужен, чтобы не принять за повтор
                   инструкции фразу или число, которые назвал сам пользователь
                   (ответ «не 90 дней, а 30» на вопрос «срок 90 дней?» — это норма).
        """
        text = str(answer or "")
        if isinstance(question, InputVerdict):
            asked = question.original_question + "\n" + question.history_text
        else:
            asked = str(question or "")
        asked_norm = R.norm_plain(asked)
        findings: list = []
        clean = text.replace(self.canary, "")
        clean = re.sub(r"\[\[/?DATA:[^\]]{0,40}\]\]", "", clean).strip()
        if not clean:
            return OutputVerdict(REPLACE, [Finding("format", "format.empty", "high", "", "answer")], "")

        norm = R.norm_plain(clean)
        ctx_text = None if context is None else (
            context if isinstance(context, str) else "\n".join(str(c) for c in context))
        ctx_norm = None if ctx_text is None else R.norm_plain(ctx_text)
        hints: list = []

        # утечка системного промпта
        own_hits = sum(1 for s in self._own_secrets if s in norm)
        with self._lock:
            reg_hits = sum(1 for s in self._registered_secrets if s in norm)
            payloads = set(self._payloads)
            injections = list(self._injections)
            inj_numbers = set(self._inj_numbers)
            kb_norm = list(self._kb_norm)
        if self.canary in text:
            findings.append(Finding(R.PROMPT_LEAK, "leak.canary", "high", "", "answer"))
        if own_hits >= 2 or reg_hits >= 3:
            findings.append(Finding(R.PROMPT_LEAK, "leak.prompt_text", "high",
                                    "совпало предложений: %d" % (own_hits + reg_hits), "answer"))
        if "[[DATA:" in text or "[[/DATA:" in text:
            findings.append(Finding(R.PROMPT_LEAK, "leak.delimiter", "low", "", "answer"))
        if any(f.severity == "high" for f in findings):
            hints.append(P.RETRY_HINTS["leak"])

        # повтор встроенной в документ инструкции
        echo = False
        for payload in payloads:
            if payload in norm and payload not in asked_norm:
                legit = payload in ctx_norm if ctx_norm is not None else any(payload in d for d in kb_norm)
                if not legit:
                    findings.append(Finding(R.DOC_INJECTION, "echo.payload", "high", _excerpt(payload), "answer"))
                    echo = True
        for inj in injections:
            words = inj.split()
            if any(" ".join(words[i:i + 5]) in norm for i in range(max(0, len(words) - 4))):
                findings.append(Finding(R.DOC_INJECTION, "echo.quote", "medium", _excerpt(inj), "answer"))
                echo = True
                break
        if ctx_norm is not None and question is not None:
            for num in inj_numbers:
                pat = r"(?<!\d)%s(?!\d)" % re.escape(num)
                if re.search(pat, norm) and not re.search(pat, ctx_norm) and not re.search(pat, asked_norm):
                    findings.append(Finding(R.DOC_INJECTION, "echo.number", "medium", num, "answer"))
                    echo = True
        if echo:
            hints.append(P.RETRY_HINTS["echo"])

        # выход из роли
        shadow = R.shadow(clean)[0]
        for pat in R.ANSWER_ROLE_BREAK:
            m = pat.search(shadow)
            if m:
                findings.append(Finding(R.ROLE_CHANGE, "role.break", "medium",
                                        _excerpt(clean[m.start():m.end()]), "answer"))
                hints.append(P.RETRY_HINTS["role"])
                break

        # ссылки, которых нет в контексте
        if ctx_text is not None:
            missing = self._unsupported_refs(clean, ctx_text)
            if missing:
                findings.append(Finding("ungrounded_ref", "ref.unsupported",
                                        "medium" if strict_refs else "low", ", ".join(missing), "answer"))
                if strict_refs:
                    hints.append(P.RETRY_HINTS["refs"].format(refs=", ".join(missing)))

        # длина
        if len(clean) > self.max_answer_chars:
            cut = clean[: self.max_answer_chars]
            end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
            clean = cut[: end + 1] if end > self.max_answer_chars // 2 else cut
            findings.append(Finding("format", "format.too_long", "low", "", "answer"))

        action = REGENERATE if any(f.severity in ("medium", "high") for f in findings) else PASS
        retry = (P.RETRY_PREFIX + " ".join(hints) + P.RETRY_SUFFIX) if action == REGENERATE else ""
        if findings:
            self._record("G4", {"action": action, "rules": sorted({f.rule for f in findings})})
        return OutputVerdict(action, findings, clean, retry)

    def answer_with_retry(self, llm: LLM, messages: list,
                          context: Union[str, Sequence[str], None] = None,
                          strict_refs: bool = False,
                          question: Union[InputVerdict, str, None] = None) -> tuple:
        """Вызывает модель, проверяет ответ и при необходимости один раз перегенерирует.

        Возвращает (текст ответа, OutputVerdict последней проверки). Если повторный
        ответ снова не прошёл проверку, заглушка verdict.safe_answer подставляется только
        при утечке внутренних инструкций. В остальных случаях возвращается повторный
        ответ, а verdict.ok остаётся False: вызывающий код решает сам.
        """
        try:
            first = llm(messages)
        except Exception as exc:  # noqa: BLE001
            return P.SAFE_ANSWER, OutputVerdict(REPLACE, [_error_finding(exc, "answer")], "")
        verdict = self.check_answer(first, context, strict_refs, question)
        if verdict.action == PASS:
            return verdict.clean_answer, verdict
        if verdict.action == REGENERATE:
            retry_messages = list(messages) + [
                {"role": "assistant", "content": verdict.clean_answer},
                {"role": "user", "content": verdict.retry_message},
            ]
            try:
                second = llm(retry_messages)
            except Exception as exc:  # noqa: BLE001
                return verdict.safe_answer, OutputVerdict(REPLACE, [_error_finding(exc, "answer")], "")
            verdict2 = self.check_answer(second, context, strict_refs, question)
            if verdict2.action == PASS:
                return verdict2.clean_answer, verdict2
            leaked = any(f.category == R.PROMPT_LEAK and f.severity == "high" for f in verdict2.findings)
            if leaked or verdict2.action == REPLACE:
                return verdict2.safe_answer, verdict2
            return verdict2.clean_answer, verdict2
        return verdict.safe_answer, verdict

    # ------------------------------------------------------------------ G5
    def events(self) -> list:
        """G5. Копия журнала событий (последние 5000)."""
        with self._lock:
            return list(self._events)

    def stats(self) -> dict:
        """G5. Сводка по журналу: сколько срабатываний на каждой точке и по каждой категории."""
        points: dict = {}
        categories: dict = {}
        actions: dict = {}
        for event in self.events():
            points[event["point"]] = points.get(event["point"], 0) + 1
            for cat in event.get("categories", []):
                categories[cat] = categories.get(cat, 0) + 1
            if "action" in event:
                actions[event["action"]] = actions.get(event["action"], 0) + 1
        return {"points": points, "categories": categories, "actions": actions}

    def session_update(self, session_id: str, verdict: InputVerdict) -> dict:
        """G5. Счётчик риска на сессию веб-чата: растёт с каждой попыткой атаки."""
        with self._lock:
            state = self._sessions.setdefault(str(session_id), {"attempts": 0, "level": "normal"})
            if verdict.action != ALLOW and verdict.risk >= 0.6:
                state["attempts"] += 1
            state["level"] = "normal" if state["attempts"] < 2 else ("elevated" if state["attempts"] < 4 else "high")
            return dict(state)

    # ------------------------------------------------------------ внутреннее
    def _record(self, point: str, data: dict) -> None:
        event = {"ts": round(time.time(), 3), "point": point}
        event.update(data)
        with self._lock:
            self._events.append(event)
            if self.log_path:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                except OSError:
                    pass

    def _remember_injections(self, removed: Sequence[str]) -> None:
        if not removed:
            return
        with self._lock:
            for sentence in removed:
                norm = R.norm_plain(sentence)
                if norm and norm not in self._injections:
                    self._injections.append(norm)
                    for quoted in R.QUOTED.findall(sentence):
                        self._payloads.add(R.norm_plain(quoted))
                    self._inj_numbers.update(re.findall(r"(?<!\d)\d{2,}(?!\d)", norm))

    def _remember_kb(self, clean_text: str) -> None:
        with self._lock:
            if self._kb_chars < 3_000_000:
                norm = R.norm_plain(clean_text)
                self._kb_norm.append(norm)
                self._kb_chars += len(norm)

    def _clean_doc_text(self, text: str, where: str) -> tuple:
        clean, n_invisible, hidden = R.strip_invisible(text)
        findings: list = []
        removed: list = []
        if hidden.strip():
            findings.append(Finding(R.OBFUSCATION, "doc.tag_chars", "high", _excerpt(hidden), where))
            removed.append(hidden.strip())
        elif n_invisible >= 5:
            findings.append(Finding(R.OBFUSCATION, "doc.invisible", "low",
                                    "%d скрытых символов" % n_invisible, where))
        out_lines: list = []
        for line in clean.split("\n"):
            if not line.strip():
                out_lines.append(line)
                continue
            line_shadow = R.shadow(line)[0]
            if not (R.DOC_ADDRESS.search(line_shadow) or R.DOC_IMPERATIVE.search(line_shadow)
                    or any(p.search(line_shadow) for p in R.DOC_META)):
                out_lines.append(line)
                continue
            keep: list = []
            addressed = False
            changed = False
            for sentence in R.split_doc_sentences(line):
                sent_shadow = R.shadow(sentence)[0]
                reasons, address = R.doc_sentence_reasons(sent_shadow)
                if not reasons and addressed and R.doc_has_command(sent_shadow):
                    reasons = ["doc.follows_address"]
                if reasons:
                    removed.append(sentence.strip())
                    findings.append(Finding(R.DOC_INJECTION, reasons[0], "high", _excerpt(sentence), where))
                    addressed = addressed or address
                    changed = True
                else:
                    keep.append(sentence)
                    addressed = False
            if not changed:
                out_lines.append(line)
            elif keep:
                out_lines.append(" ".join(s.strip() for s in keep))
        return "\n".join(out_lines), removed, findings

    def _analyze(self, text: str, where: str, only: Optional[set] = None) -> tuple:
        shadow, folded = R.shadow(text)
        findings: list = []
        spans: list = []
        seen: set = set()
        for hit in R.find_hits(shadow):
            if only is not None and hit.rule.category not in only:
                continue
            start, end = hit.start, hit.end
            if hit.rule.strip:
                start, end = R.extend_span(shadow, start, end, hit.rule.strip)
                spans.append((start, end))
            key = (hit.rule.id, start, end)
            if key in seen:
                continue
            seen.add(key)
            findings.append(Finding(hit.rule.category, hit.rule.id, hit.rule.severity,
                                    _excerpt(text[start:end]), where))
        for start, end, decoded in R.find_base64_payloads(text):
            spans.append((start, end))
            findings.append(Finding(R.OBFUSCATION, "obfuscation.base64", "medium", _excerpt(decoded), where))
            for hit in R.find_hits(R.shadow(decoded)[0]):
                findings.append(Finding(hit.rule.category, hit.rule.id, hit.rule.severity,
                                        _excerpt(decoded[hit.start:hit.end]), "hidden"))
        if R.ILLEGAL_INFO.search(shadow) and not R.ILLEGAL_HOWTO.search(shadow):
            # вопрос о последствиях или допустимости — отвечаем по базе как обычно
            for f in findings:
                if f.category == R.ILLEGAL and f.severity == "medium":
                    f.severity = "low"
                    f.rule += ".informational"
        if folded and any(f.category != R.OBFUSCATION for f in findings):
            findings.append(Finding(R.OBFUSCATION, "obfuscation.homoglyph", "low",
                                    "слов с подменёнными буквами: %d" % folded, where))
        clean = R.cut_spans(text, spans) if spans else text
        return clean, findings

    def _should_ask_llm(self, clean_q: str, q_findings: list) -> bool:
        if self.llm_mode == "off" or not R.has_substance(clean_q):
            return False
        if self.llm_mode == "always":
            return True
        if any(f.severity == "high" for f in q_findings):
            return False
        shadow = R.shadow(clean_q)[0]
        return (not R.DOMAIN_LEXICON.search(shadow)) or bool(R.WEAK_SUSPICION.search(shadow)) or len(clean_q) > 600

    def _classify(self, llm: LLM, clean_q: str) -> tuple:
        """Смысловая классификация моделью. Ответ модели может только добавить флаги
        или вырезать дословные фрагменты — текст от модели в вопрос не попадает."""
        messages = [
            {"role": "system", "content": P.CLASSIFIER_SYSTEM},
            {"role": "user", "content": P.CLASSIFIER_USER.format(wrapped=self.wrap(clean_q, "question"))},
        ]
        data = None
        for _ in range(2):  # одна повторная попытка: сеть и обрыв ответа случаются
            try:
                reply = llm(messages)
                parsed = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
                if isinstance(parsed, dict):
                    data = parsed
                    break
            except Exception:  # noqa: BLE001
                continue
        if data is None:
            return clean_q, [Finding("classifier_error", "llm.bad_reply", "low", "", "question", "llm")]

        allowed = R.ALL_CATEGORIES - {R.OBFUSCATION, R.DOS, R.DOC_INJECTION}
        categories = [c for c in data.get("categories", []) if isinstance(c, str) and c in allowed]
        spans_text = [s for s in data.get("attack_spans", []) if isinstance(s, str) and s.strip()]
        legit = bool(data.get("has_legit_question", True))
        findings = [Finding(c, "llm." + c, "medium", _excerpt(" | ".join(spans_text)), "question", "llm")
                    for c in categories]
        shadow = R.shadow(clean_q)[0]
        if R.ILLEGAL_INFO.search(shadow) and not R.ILLEGAL_HOWTO.search(shadow):
            # вопрос о последствиях или допустимости — не просьба помочь нарушить
            for f in findings:
                if f.category == R.ILLEGAL:
                    f.severity = "low"
                    f.rule += ".informational"
        strip = [c for c in categories if c in R.STRIP_CATEGORIES]
        if not strip:
            return clean_q, findings
        positions = []
        for span in spans_text:
            needle = R.shadow(span.strip())[0]
            idx = shadow.find(needle)
            if idx >= 0:
                positions.append((idx, idx + len(needle)))
        if not legit:
            only_offtopic = set(strip) == {R.OFFTOPIC}
            if only_offtopic and R.DOMAIN_LEXICON.search(shadow):
                for f in findings:
                    f.severity = "low"
                return clean_q, findings
            return "", findings
        return (R.cut_spans(clean_q, positions) if positions else clean_q), findings

    def _decide(self, question_raw: str, clean_q: str, history: list, findings: list,
                has_assistant_turn: bool) -> InputVerdict:
        categories: list = []
        for f in findings:
            if f.category in R.ALL_CATEGORIES and f.category not in categories:
                categories.append(f.category)
        q_cats = {f.category for f in findings if f.where == "question"}
        cut = clean_q != question_raw
        illegal = any(f.category == R.ILLEGAL and f.where == "question" and f.severity != "low"
                      for f in findings)

        if cut and not R.has_substance(clean_q):
            action = DECLINE
            clean_q = ""
        elif illegal:
            action = REFUSE_ASSIST
        elif any(f.severity in ("medium", "high") for f in findings):
            action = SANITIZE
        else:
            action = ALLOW

        directives: list = []
        if q_cats & _RULES_ATTACK:
            directives.append(P.DIRECTIVES["rules_attack"])
        if R.PROMPT_LEAK in q_cats:
            directives.append(P.DIRECTIVES["prompt_leak"])
        for cat in (R.PRESSURE, R.FAKE_FACT, R.FORMAT):
            if cat in q_cats:
                directives.append(P.DIRECTIVES[cat])
        if illegal:
            directives.append(P.DIRECTIVES[R.ILLEGAL])
        if R.OFFTOPIC in q_cats and cut:
            directives.append(P.DIRECTIVES["offtopic"])
        if any(f.where == "history" for f in findings):
            directives.append(P.DIRECTIVES["history_attack"])
        if any(f.where == "hidden" for f in findings):
            directives.append(P.DIRECTIVES["hidden"])
        if has_assistant_turn:
            directives.append(P.DIRECTIVES["history"])

        reply = None
        if action == DECLINE:
            if q_cats & _RULES_ATTACK:
                reply = P.REPLIES["rules_attack"]
            elif R.PROMPT_LEAK in q_cats:
                reply = P.REPLIES["prompt_leak"]
            else:
                reply = P.REPLIES["offtopic"]

        weights = [R.SEVERITY_WEIGHT.get(f.severity, 0.0) for f in findings if f.category in R.ALL_CATEGORIES]
        risk = min(1.0, max(weights) + 0.05 * (len(weights) - 1)) if weights else 0.0
        return InputVerdict(action, categories, round(risk, 2), findings, question_raw, clean_q,
                            history, directives, reply)

    @staticmethod
    def _unsupported_refs(answer: str, context: str) -> list:
        missing: list = []
        ctx_key = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", context).upper()
        ctx_low = context.lower()
        for ref in dict.fromkeys(R.REF_DOC_NUMBER.findall(answer)):
            if re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", ref).upper() not in ctx_key:
                missing.append(ref)
        for num in dict.fromkeys(R.REF_ARTICLE.findall(answer)):
            if not re.search(r"ст\w*\.?[\s_]*%s(?!\d)" % re.escape(num), ctx_low):
                missing.append("ст. " + num)
        for num in dict.fromkeys(R.REF_CLAUSE.findall(answer)):
            if not re.search(r"(?<![\d.])%s(?!\d)" % re.escape(num), context):
                missing.append("п. " + num)
        return missing


# --------------------------------------------------------------------------
# G5. Функции для веб-интерфейса и логов
# --------------------------------------------------------------------------
def strip_active_content(text: str) -> str:
    """Убирает из текста HTML-теги, картинки и ссылки в разметке Markdown.

    Картинка или ссылка в ответе модели может унести данные наружу через адрес,
    поэтому в интерфейсе остаётся только их подпись.
    """
    out = re.sub(r"!\[([^\]]{0,200})\]\([^)]{0,2000}\)", r"\1", str(text or ""))
    out = re.sub(r"\[([^\]]{1,200})\]\([^)]{0,2000}\)", r"\1", out)
    out = re.sub(r"<[^>]{1,300}>", "", out)
    return out


def safe_html(text: str) -> str:
    """Экранирует текст для вставки в HTML. Применять и к ответу, и к фрагментам документов."""
    return html.escape(str(text or ""), quote=True).replace("\n", "<br>")


def prepare_for_display(text: str) -> str:
    """Готовит ответ модели или фрагмент документа к показу в веб-интерфейсе."""
    return safe_html(strip_active_content(text))


_SECRET_ENV = ("LLM_API_KEY", "OPENAI_API_KEY", "EMBEDDER_API_KEY", "RERANKER_API_KEY")


def redact_secrets(text: str) -> str:
    """Маскирует ключи доступа перед записью в лог."""
    out = str(text or "")
    for name in _SECRET_ENV:
        value = os.environ.get(name, "").strip()
        if len(value) >= 6:
            out = out.replace(value, "***")
    out = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}", "Bearer ***", out)
    out = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "sk-***", out)
    return out


def llm_from_env(timeout: float = 90.0, max_tokens: int = 1500) -> Optional[LLM]:
    """Строит функцию llm(messages) -> str по переменным окружения платформы.

    Нужна, чтобы включить смысловую классификацию без чужого HTTP-клиента.
    Возвращает None, если адрес или ключ не заданы.

    max_tokens взят с запасом: у «думающих» моделей в этот лимит входят и токены
    рассуждения, и при малом значении ответ обрывается до JSON.
    """
    import urllib.request

    base = (os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("BASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL")
             or os.environ.get("GEN_MODEL") or "").strip()
    if not base or not key or not model:
        return None

    def call(messages: list) -> str:
        payload = json.dumps({"model": model, "messages": messages, "temperature": 0.0,
                              "max_tokens": max_tokens}).encode("utf-8")
        request = urllib.request.Request(
            base + "/chat/completions", data=payload,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content") or "")

    return call
