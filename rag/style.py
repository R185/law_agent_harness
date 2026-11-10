"""Контроль тона ответа юридического ассистента.

Слой решает одну задачу: ответ, который уйдёт юристу банка, должен быть
служебной правовой справкой, а не сообщением приятеля. Он состоит из трёх
частей:

1. STYLE_CONTRACT — текст контракта, который добавляется к системному промпту;
2. аудит — поиск отступлений от регистра в готовом ответе модели;
3. детерминированная чистка — удаление технических следов чата (markdown,
   эмодзи, служебные вступления), которую нельзя поручить самой модели.

Аудит и чистка намеренно разделены: чистка не трогает смысл и не переписывает
формулировки, а регистровые нарушения уходят модели на один повторный проход.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HARD = "hard"
SOFT = "soft"

STYLE_CONTRACT = """ТОН, РЕГИСТР И ФОРМА ОТВЕТА (нарушение любого пункта — ошибка):
Ты готовишь служебную правовую справку для юриста банка. Читатель — профессионал, который будет проверять твои ссылки. Регистр — официально-деловой, нейтральный, без эмоций.
1. Ответ начинается с существа: «Да.», «Нет.», «Согласно п. 3.1 …», число, срок. Запрещены вступления и служебные обороты: «Конечно», «Отличный вопрос», «Вот ответ», «Давайте разберёмся», «Итак», «Спасибо за вопрос».
2. Запрещены приветствия, прощания, благодарности, извинения, обещания помощи и завершающие фразы вроде «Надеюсь, это помогло», «Обращайтесь». Отсутствие данных — констатация, а не извинение.
3. Запрещены фамильярность, шутки, ирония, сарказм, сленг, разговорные вставки, эмоциональные оценки и пустые усилители: «кстати», «вообще», «по сути», «на самом деле», «честно говоря», «круто», «ок», «супер», «штука».
4. Запрещено обращение к пользователю на «ты» (ты, тебя, тебе, твой) и любые личные обращения. Допустимы безличные формулировки и «вы» внутри цитат из документов.
5. Запрещено ссылаться на себя как на модель, ИИ или ассистента, на «мою базу знаний», «мои знания», «моё обучение». Для отказа используй формулу: «В предоставленной базе знаний отсутствуют документы о …».
6. Запрещены рекламные и оценочные прилагательные: «уникальный», «эффективный», «удобный», «оптимальный», «передовой», а также мотивационный тон.
7. Запрещены пустые хеджи: «возможно», «наверное», «как правило», «по-моему», «мне кажется», «в целом». Факт из фрагмента утверждается прямо; отсутствие факта указывается прямо.
8. Один термин — одно название на весь ответ. Не подбирай синонимы ради разнообразия.
9. Форма: без markdown (никаких **, #, `, списков и заголовков), без эмодзи и знаков-украшений, без кавычек-акцентов, без восклицательных знаков. Одна строка, 1–5 предложений.
10. Юридическая терминология и канцеляризмы уместны: «в соответствии с», «вправе», «подлежит», «имущественное положение», «в редакции от». Это норма жанра — не упрощай их до разговорных оборотов.
11. Уточняющий вопрос допустим, только если без него ответ будет неверным. Тогда одно нейтральное предложение в конце, без «плиз» и «давай(те)»."""

REPAIR_INSTRUCTION = """Ответ не принят: он не соответствует служебному регистру правовой справки.
Выявленные нарушения:
{issues}

Перепиши ответ заново. Сохрани все факты, числа, названия документов и ссылки на пункты без изменений. Убери только то, что перечислено выше. Ничего не добавляй от себя. Ответ — одна строка, 1–5 предложений."""


@dataclass(frozen=True)
class Violation:
    code: str
    severity: str
    category: str
    quote: str
    hint: str


@dataclass
class ToneResult:
    clean: str
    fixed: list[Violation] = field(default_factory=list)
    remaining: list[Violation] = field(default_factory=list)

    @property
    def hard(self) -> list[Violation]:
        return [v for v in self.remaining if v.severity == HARD]


def _rule(code: str, severity: str, category: str, pattern: str, hint: str) -> tuple:
    return (code, severity, category, re.compile(pattern, re.I), hint)


HARD_RULES: list[tuple] = [
    _rule(
        "chatbot_opening",
        HARD,
        "chatbot",
        r"^\s*(?:конечно|разумеется|безусловно|отличный вопрос|хороший вопрос|спасибо за вопрос|давайте|итак|вот (?:ваш |мой )?(?:ответ|итог))\b",
        "убери вступление, начинай с ответа по существу",
    ),
    _rule(
        "chatbot_closing",
        HARD,
        "chatbot",
        r"(?:надеюсь,? (?:(?:это|этот ответ|оно) )?(?:было |будет )?(?:полезн|помог)\w*|если (?:у вас )?(?:есть|появятся|возникнут) (?:ещё |еще )?вопрос|обращайтесь|буду рад помочь|готов помочь|с удовольствием помог)",
        "убери прощание и обещание помощи",
    ),
    _rule("gratitude", HARD, "chatbot", r"\b(?:спасибо|благодарю)\b", "убери благодарность"),
    _rule(
        "apology",
        HARD,
        "chatbot",
        r"\b(?:извините|извини|прошу прощения|простите)\b",
        "убери извинение, отказ формулируется как констатация",
    ),
    _rule(
        "model_identity",
        HARD,
        "identity",
        r"(?:\bкак (?:ии|искусственный интеллект|языковая модель)\b|\bя\s*[—-]\s*(?:ии|искусственный интеллект|языковая модель|модель|ассистент)\b|\bмо(?:я|ей|ю) баз[аеы] знаний\b|\bмои знания\b|\bна момент (?:моего )?обучения\b|\bя не имею доступа\b)",
        "не ссылайся на себя как на модель и на свои знания",
    ),
    _rule(
        "second_person_singular",
        HARD,
        "register",
        r"(?<![\w-])(?:ты|тебе|тебя|тобой|твой|твоя|твоё|твое|твои|твоего|твоему|твоим)(?![\w-])",
        "обращение на «ты» недопустимо: переформулируй безлично",
    ),
    _rule(
        "profanity",
        HARD,
        "register",
        r"(?:хуй|хуё|хуе|пизд|бляд|бля[!,. ]|ебан|ебат|ебёт|ебет|мудак|говно|жоп[аыуе])",
        "грубая лексика недопустима",
    ),
    _rule(
        "colloquialism",
        HARD,
        "register",
        r"(?<![\w-])(?:ок|окей|кстати|к слову|вообще|по-любому|прикинь|чё|щас|норм|ладно|ага|ну|типа|капец|жесть|круто|супер|фигня|пофиг|штука|штуки|штуку|делов-то|без проблем|в общем|честно говоря|на самом деле|по сути|скажем так|как-то так|по-быстрому|смотри|короче|просто|тут|вон|вот)(?![\w-])",
        "замени разговорное слово нейтральным",
    ),
    _rule(
        "chatty_frame",
        HARD,
        "register",
        r"(?:как я понимаю|насколько я понимаю|если я правильно понял|между прочим|вот что (?:важно|интересно)|дело в том, что|штука в том)",
        "убери разговорную рамку и перейди к существу",
    ),
    _rule(
        "false_promise",
        HARD,
        "register",
        r"\b(?:я уверен|поверь(?:те)?|обещаю|гарантирую тебе)\b",
        "убери заверения и обещания",
    ),
    _rule(
        "rhetorical_question",
        HARD,
        "register",
        r"(?:знаете ли|не так ли|правда\?|согласны\?)",
        "убери риторический вопрос",
    ),
]

SOFT_RULES: list[tuple] = [
    _rule(
        "hedge",
        SOFT,
        "vagueness",
        r"(?<![\w-])(?:возможно|наверное|вероятно|по-видимому|как правило|мне кажется|по-моему|скорее всего|в целом|в принципе|вроде)(?![\w-])",
        "убери хедж или замени прямым утверждением",
    ),
    _rule(
        "promo_adjective",
        SOFT,
        "register",
        r"\b(?:уникальн\w*|эффективн\w*|удобн\w*|оптимальн\w*|передов\w*|инновационн\w*|современн\w*|качественн\w*)",
        "убери оценочное прилагательное",
    ),
    _rule(
        "filler_link",
        SOFT,
        "register",
        r"(?:важно отметить|следует отметить|стоит подчеркнуть|таким образом|в заключение|подводя итог|к сожалению)",
        "убери служебную связку",
    ),
    _rule("exclamation", SOFT, "format", r"!", "восклицательный знак не используется в справке"),
    _rule("motivational", SOFT, "register", r"(?:вы сможете|верьте в себя|не бойтесь|смело)", "убери мотивационный тон"),
]

CHAT_PREAMBLE_RE = re.compile(
    r"^\s*(?:(?:конечно|разумеется|безусловно)\s*[,!.]\s*"
    r"|(?:отличный|хороший|прекрасный|замечательный)\s+вопрос\s*[!.,]?\s*"
    r"|спасибо за (?:ваш )?вопрос\s*[!.,]?\s*"
    r"|итак\s*[,.]\s*"
    r"|вот (?:ваш |мой )?(?:ответ|итог)\s*[:.]\s*"
    r"|вот что \w[^.!?]{0,80}?[:.]\s*"
    r"|давайте \w[^.!?]{0,80}?[:.]\s*)",
    re.I,
)

CHAT_CLOSING_RE = re.compile(
    r"\s*(?:(?:надеюсь|полагаю),? (?:(?:это|этот ответ|оно) )?(?:было |будет )?(?:полезн|помог)\w*[^.!?]*[.!?]?"
    r"|если (?:у вас )?(?:есть|появятся|возникнут) (?:ещё |еще )?вопрос\w*[^.!?]*[.!?]?"
    r"|обращайтесь[^.!?]*[.!?]?"
    r"|буду рад помочь[^.!?]*[.!?]?)\s*$",
    re.I,
)

INVISIBLE_TRANSLATION = {
    0x00A0: " ",
    0x202F: " ",
    0x2009: " ",
    0x2007: " ",
    0x200B: "",
    0xFEFF: "",
    0xE200: "",
    0xE201: "",
    0xE202: "",
}

INVISIBLE_RE = re.compile("[\u00a0\u202f\u2009\u2007\u200b\ufeff\ue200\ue201\ue202]")
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2B2F\u2B50\u2B55\uFE0F\u2190-\u21AF]"
)
CJK_RE = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff01-\uff60]")
SERVICE_TOKEN_RE = re.compile(
    r"(?:【[^】]*】|\[\s*Источник:\s*[^\]]*\]|<\|[^|>]*\|>|\[/?INST\]|contentReference|oaicite|oai_citation|turn0search\d|†)"
)
MARKDOWN_RE = re.compile(r"[*#`~<>]")
BULLET_RE = re.compile(r"^\s*(?:[-+•·]|\d+[.)])\s+", re.M)
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
SPACED_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def _snippet(text: str, start: int, end: int, width: int = 30) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return " ".join(text[left:right].split())


def _scan(text: str, rules: list[tuple]) -> list[Violation]:
    found: list[Violation] = []
    for code, severity, category, regex, hint in rules:
        match = regex.search(text)
        if match:
            found.append(
                Violation(
                    code=code,
                    severity=severity,
                    category=category,
                    quote=_snippet(text, match.start(), match.end()),
                    hint=hint,
                )
            )
    return found


def audit(text: str) -> list[Violation]:
    """Полный аудит тона: технические следы, регистр, мягкие замечания."""
    if not text or not text.strip():
        return []
    return _scan(text, HARD_RULES) + _scan(text, SOFT_RULES)


def audit_hard(text: str) -> list[Violation]:
    if not text or not text.strip():
        return []
    return _scan(text, HARD_RULES)


def strip_artifacts(text: str) -> str:
    """Убирает только технические следы чата, не трогая смысл и формулировки."""
    if not text:
        return ""
    result = text.replace("\r\n", "\n").replace("\r", "\n")
    result = INVISIBLE_RE.sub(lambda m: INVISIBLE_TRANSLATION.get(ord(m.group(0)), " "), result)
    result = EMOJI_RE.sub("", result)
    result = SERVICE_TOKEN_RE.sub("", result)
    result = CJK_RE.sub("", result)
    result = re.sub(r"^\s*```\w*\s*$", "", result, flags=re.M)
    result = MARKDOWN_RE.sub("", result)
    result = BULLET_RE.sub("", result)
    result = result.strip()
    result = result.strip("\"'").strip()
    for _ in range(2):
        previous = result
        result = CHAT_PREAMBLE_RE.sub("", result).strip()
        result = CHAT_CLOSING_RE.sub("", result).strip()
        if result == previous:
            break
    result = MULTISPACE_RE.sub(" ", result)
    result = SPACED_PUNCT_RE.sub(r"\1", result)
    result = " ".join(result.split())
    return result


def enforce_tone(raw: str) -> ToneResult:
    """Чистит технические артефакты и возвращает оставшиеся нарушения регистра."""
    artifacts = _scan(raw, HARD_RULES)
    clean = strip_artifacts(raw)
    remaining = audit_hard(clean)
    fixed = [v for v in artifacts if v.code not in {r.code for r in remaining}]
    return ToneResult(clean=clean, fixed=fixed, remaining=remaining)


def build_issues_message(issues: list[str]) -> str:
    formatted = "\n".join(f"— {issue}" for issue in issues)
    return REPAIR_INSTRUCTION.format(issues=formatted)


def build_repair_message(violations: list[Violation]) -> str:
    issues = [f"{v.hint} (фрагмент: «{v.quote}»)" for v in violations]
    return build_issues_message(issues)


def format_violations(violations: list[Violation]) -> str:
    return "; ".join(f"{v.code}: «{v.quote}»" for v in violations)
