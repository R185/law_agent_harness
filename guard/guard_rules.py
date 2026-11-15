"""Правила распознавания и нормализация текста для модуля guard.

Только стандартная библиотека, никаких импортов из остального проекта.

Все шаблоны применяются к «тени» текста — строке той же длины в нижнем
регистре, где ё заменена на е, а подменённые буквы (латиница вместо кириллицы
и наоборот) приведены к одному алфавиту. Длина тени равна длине исходного
текста, поэтому найденный фрагмент можно вырезать из оригинала по тем же
позициям.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Категории угроз
# --------------------------------------------------------------------------
OVERRIDE = "override"            # отмена или подмена правил ассистента
FAKE_SYSTEM = "fake_system"      # поддельное системное сообщение, служебная разметка
ROLE_CHANGE = "role_change"      # смена роли или личности
PROMPT_LEAK = "prompt_leak"      # вытягивание системного промпта
AUTHORITY = "authority"          # ссылка на особые полномочия
PRESSURE = "pressure"            # давление, диктовка ответа
FAKE_FACT = "fake_fact"          # подложное «содержание документа»
FORMAT = "format_coercion"       # формат, убирающий ссылки и оговорки
ILLEGAL = "illegal"              # помощь в нарушении закона или внутренних правил
OFFTOPIC = "offtopic"            # запрос не по юридической тематике
OBFUSCATION = "obfuscation"      # скрытые символы, кодировки, подмена букв
DOS = "dos"                      # чрезмерно большой ввод
DOC_INJECTION = "doc_injection"  # инструкция, встроенная в документ базы знаний

#: Категории, фрагменты которых вырезаются из вопроса.
STRIP_CATEGORIES = {OVERRIDE, FAKE_SYSTEM, ROLE_CHANGE, PROMPT_LEAK, AUTHORITY, OFFTOPIC}
ALL_CATEGORIES = STRIP_CATEGORIES | {PRESSURE, FAKE_FACT, FORMAT, ILLEGAL, OBFUSCATION, DOS, DOC_INJECTION}

SEVERITY_WEIGHT = {"low": 0.2, "medium": 0.6, "high": 0.9}


# --------------------------------------------------------------------------
# Нормализация
# --------------------------------------------------------------------------
_EXTRA_INVISIBLE = {0x034F, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x3164, 0xFFA0}


def strip_invisible(text: str) -> tuple[str, int, str]:
    """Убирает невидимые и управляющие символы.

    Возвращает (чистый текст, сколько символов убрано, скрытый текст).
    Скрытый текст — то, что было спрятано «тег-символами» Юникода
    (U+E0020…U+E007E): ими можно незаметно передать латинскую строку.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    hidden: list[str] = []
    removed = 0
    for ch in text:
        code = ord(ch)
        if 32 <= code < 127 or ch == "\n" or ch == "\t":
            out.append(ch)
            continue
        if 0xE0020 <= code <= 0xE007E:
            hidden.append(chr(code - 0xE0000))
            removed += 1
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Cc") or code in _EXTRA_INVISIBLE:
            removed += 1
            continue
        if cat == "Zs":
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out), removed, "".join(hidden)


_LAT2CYR = str.maketrans("ABCEHKMOPTXYaceopxyk", "АВСЕНКМОРТХУасеорхук")
_CYR2LAT = str.maketrans("АВСЕНКМОРТХУасеорхук", "ABCEHKMOPTXYaceopxyk")
_TOKEN_RE = re.compile(r"[^\W_]+")


def _is_cyr(ch: str) -> bool:
    return "а" <= ch <= "я" or "А" <= ch <= "Я" or ch in "ёЁ"


def _is_lat(ch: str) -> bool:
    return "a" <= ch <= "z" or "A" <= ch <= "Z"


def shadow(text: str) -> tuple[str, int]:
    """Строит «тень» текста той же длины и считает слова с подменёнными буквами."""
    chars = list(text)
    folded = 0
    for m in _TOKEN_RE.finditer(text):
        word = m.group()
        if word.isascii() and word.isalpha():
            continue
        cyr = sum(1 for c in word if _is_cyr(c))
        lat = sum(1 for c in word if _is_lat(c))
        new = word
        if cyr and lat:
            new = word.translate(_LAT2CYR if cyr >= lat else _CYR2LAT)
        if (cyr or lat) and any(c in "03" for c in word):
            to_cyr = cyr >= lat
            buf = list(new)
            for i in range(1, len(buf) - 1):
                if buf[i] in "03" and buf[i - 1].isalpha() and buf[i + 1].isalpha():
                    if buf[i] == "0":
                        buf[i] = "о" if to_cyr else "o"
                    else:
                        buf[i] = "з" if to_cyr else "e"
            new = "".join(buf)
        if new != word:
            if len(word) >= 4:
                folded += 1
            chars[m.start():m.end()] = list(new)
    out: list[str] = []
    for ch in chars:
        low = ch.lower()
        if len(low) != 1:
            low = ch
        out.append("е" if low == "ё" else low)
    return "".join(out), folded


def norm_plain(text: str) -> str:
    """Тень со схлопнутыми пробелами — для сравнения строк между собой."""
    return " ".join(shadow(text)[0].split())


_SPACED = re.compile(r"(?<!\S)(?:[^\W\d_] ){3,}[^\W\d_](?!\S)")


def collapse_spaced(text: str) -> tuple[str, int]:
    """Склеивает слова, набранные через пробел после каждой буквы («з а б у д ь»)."""
    count = 0

    def join(m: "re.Match[str]") -> str:
        nonlocal count
        count += 1
        return m.group().replace(" ", "")

    return _SPACED.sub(join, text), count


#: Реплика «ассистента» в истории, в которой он якобы отказался от правил, — подделка.
FORGED_ASSISTANT = re.compile(
    r"\bя\s+теперь\b|без\s+правил|без\s+ограничений|(?:игнорирую|забыл\w*|отключил\w*)\s+(?:все\s+)?"
    r"(?:правила|инструкции|ограничения)|режим\w*\s+[\w\s]{0,20}включ|\bdan\b|буду\s+отвечать\s+так",
    re.I)


# --------------------------------------------------------------------------
# Правила для вопроса пользователя
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    severity: str
    pattern: "re.Pattern[str]"
    strip: str = ""  # "" | "span" | "clause" | "sentence" | "line"


@dataclass
class Hit:
    rule: Rule
    start: int
    end: int
    excerpt: str


def _rx(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.I | re.M)


_IMP_DROP = (
    r"(?:про)?игнорируй(?:те)?|не\s+обращай(?:те)?\s+внимани[яе]\s+на|не\s+учитывай(?:те)?|"
    r"не\s+следуй(?:те)?|не\s+соблюдай(?:те)?|забудь(?:те)?(?:\s+про|\s+об?)?|отбрось(?:те)?|"
    r"отмени(?:те)?|сбрось(?:те)?|обнули(?:те)?|отключи(?:те)?|сними(?:те)?|выключи(?:те)?"
)
_META_NOUN = r"(?:инструкци|правил|указани|настройк|ограничени|установк|директив|промпт|команд)"
_META_OBJ = (
    r"(?:" + _META_NOUN + r"|запрет|фильтр|цензур|систем\w*\s+сообщени|"
    r"(?:предыдущ|прежн|предшеству)\w*\s+(?:" + _META_NOUN + r"|сообщени|контекст|ответ)|"
    r"вышесказанн|вышеизложенн|вс[её],?\s+(?:что|выше|предыдущ)|"
    r"то,?\s+что\s+(?:тебе\s+)?(?:говорили|сказали|писали|велели|задали|было)|"
    r"остальн\w+\s+документ|документ\w*\s+баз|баз\w+\s+знаний)"
)
_TAIL = (
    r"(?:\s+(?:и\s+)?" + _META_NOUN + r"\w*)?"
    r"(?:\s+(?:(?:было\s+|написано\s+|сказано\s+|тебе\s+|говорили\s+)*(?:выше|ранее|раньше|до\s+этого)|"
    r"разработчик\w*|систем\w*|безопасности|баз\w+\s+знаний))?"
)

QUESTION_RULES: list[Rule] = [
    # ---- override: отмена или подмена правил
    Rule("override.drop_rules", OVERRIDE, "high", _rx(
        r"\b(?:" + _IMP_DROP + r")\b(?:\s+[\w-]+){0,3}?\s+" + _META_OBJ + r"\w*" + _TAIL), "span"),
    Rule("override.drop_rules_en", OVERRIDE, "high", _rx(
        r"\b(?:ignore|disregard|forget|override|bypass|skip)\b(?:\s+\w+){0,4}?\s+"
        r"(?:instructions?|rules?|prompts?|guidelines?|restrictions?|directives?|"
        r"(?:above|previous|prior|earlier|preceding)\s+\w+|everything\s+(?:above|before)|the\s+above)\b"), "span"),
    Rule("override.drop_rules_translit", OVERRIDE, "high", _rx(
        r"\b(?:(?:pro)?ignorir\w+|zabud\w*|zabyt\w*)\b(?:\s+\w+){0,4}?\s+"
        r"(?:instrukc\w+|instrukts\w+|pravil\w+|nastro[yi]k\w+|ogranichen\w+|ukazani\w+)"), "span"),
    Rule("override.new_rules", OVERRIDE, "high", _rx(
        r"\b(?:тво[ийяю]\w*\s+нов\w+\s+" + _META_NOUN + r"\w*|"
        r"нов\w+\s+(?:системн\w+\s+)?(?:инструкци|правил|установк)\w*\s+для\s+(?:тебя|ассистента|ии|модели|бота)|"
        r"нов\w+\s+системн\w+\s+(?:инструкци|правил|сообщени|промпт)\w*|"
        r"new\s+(?:system\s+)?(?:instructions|rules)\s+(?:for\s+you|apply))"), "sentence"),
    Rule("override.from_now_on", OVERRIDE, "high", _rx(
        r"\b(?:с\s+этого\s+момента|отныне|впредь|начиная\s+с\s+сейчас|from\s+now\s+on)\b,?\s+"
        r"(?:ты\b|you\b|вы\s+должны\s+отвечать|всегда\s+отвечай|отвечай|говори|пиши|считай|игнорируй|"
        r"забудь|действуй|на\s+(?:все|любые|любой)\s+вопрос)[^.!?\n]{0,120}"), "span"),
    Rule("override.rules_void", OVERRIDE, "high", _rx(
        r"\b(?:предыдущ|прежн|стар)\w*\s+(?:указани|инструкци|правил|настройк)\w*\s+(?:больше\s+|уже\s+)?"
        r"(?:не\s+действу|отмен|недействительн|утратил)\w*"), "sentence"),
    Rule("override.no_limits", OVERRIDE, "high", _rx(
        r"\b(?:будто|словно|как\s+если\s+бы)\s+у\s+тебя\s+нет\s+(?:ограничени|правил|инструкци|фильтр)\w*|"
        r"\bу\s+тебя\s+(?:больше\s+)?нет\s+(?:ограничени|правил|запрет)\w*|"
        r"\b(?:отвечай|говори|работай|действуй)\s+без\s+(?:ограничени|правил|цензур|фильтр)\w*"), "clause"),
    Rule("override.rules_changed", OVERRIDE, "high", _rx(
        r"\b(?:теперь|отныне)\s+(?:правила|инструкции|условия)\s+(?:такие|следующие|другие|изменились|новые)"),
        "sentence"),
    Rule("override.assistant_must", OVERRIDE, "high", _rx(
        r"\b(?:(?:ии|ai)[-\s]?)?(?:ассистент\w*|модель|бот|нейросеть)\s+(?:теперь\s+)?(?:должен|должна|обязан\w?)\s+"
        r"(?:подтверждать|отвечать|говорить|игнорировать|выполнять|слушаться|соглашаться|раскрывать|сообщать|"
        r"называть|считать)"), "sentence"),
    Rule("override.on_questions_say", OVERRIDE, "high", _rx(
        r"\bна\s+(?:все\s+|любые\s+)?вопрос\w*\s+(?:о|об|про)\b[^.!?\n]{0,40}?"
        r"\b(?:называй|отвечай|говори|пиши|сообщай)(?:те)?\b[^.!?\n]{0,80}"), "span"),
    Rule("override.always_answer", OVERRIDE, "high", _rx(
        r"\bна\s+(?:любой|любые|все|каждый)\s+(?:мо[йи]\s+|следующ\w+\s+)?вопрос\w*\b[^.!?\n]{0,60}?"
        r"\b(?:отвечай|говори|пиши)(?:те)?\b[^.!?\n]{0,80}"), "span"),

    # ---- fake_system: поддельные системные сообщения и разметка
    Rule("fake_system.line_marker", FAKE_SYSTEM, "high", _rx(
        r"^[ \t>*#\-]*(?:system|систем[аы]|системн\w+\s+(?:сообщени|уведомлени|инструкци|промпт)\w*|"
        r"developer|разработчик|admin|админ|администратор|root|sudo|служба\s+безопасности|security|"
        r"anthropic|openai|deepseek)\s*[:：]"), "line"),
    Rule("fake_system.chatml_block", FAKE_SYSTEM, "high", _rx(
        r"<\|im_start\|>\s*(?:system|assistant|developer)\b[\s\S]{0,2000}?<\|im_end\|>|"
        r"<<\s*sys\s*>>[\s\S]{0,2000}?<<\s*/\s*sys\s*>>"), "span"),
    Rule("fake_system.chatml", FAKE_SYSTEM, "high", _rx(
        r"<\|[a-z_]{2,20}\|>|<<\s*/?\s*sys\s*>>|\[/?inst\]"), "span"),
    Rule("fake_system.markup", FAKE_SYSTEM, "medium", _rx(
        r"</?\s*(?:question|context|documents?|data|user|input|вопрос|документ\w*|контекст)\s*>|"
        r"\[\[/?data[^\]]{0,40}\]\]"), "span"),
    Rule("fake_system.heading", FAKE_SYSTEM, "high", _rx(
        r"^\s*#{2,}\s*(?:system|instructions?|инструкци\w*|систем\w*)\b[^\n]*"), "span"),
    Rule("fake_system.inline", FAKE_SYSTEM, "high", _rx(
        r"\b(?:системн\w+\s+(?:сообщени|уведомлени)\w*|system\s+(?:message|notice|override|update))\s*[:：—-]"),
        "sentence"),
    Rule("fake_system.prompt_end", FAKE_SYSTEM, "medium", _rx(
        r"\b(?:конец|end\s+of)\s+(?:системн\w+\s+)?(?:промпт\w*|prompt|инструкци\w+|instructions)\b"), "span"),

    # ---- role_change: смена роли
    Rule("role.now_you_are", ROLE_CHANGE, "high", _rx(
        r"\b(?:ты\s+теперь|теперь\s+ты|отныне\s+ты|с\s+этого\s+момента\s+ты|you\s+are\s+now|from\s+now\s+on\s+you)\b"),
        "clause"),
    Rule("role.pretend", ROLE_CHANGE, "high", _rx(
        r"\b(?:притворись|представь,?\s+что\s+ты|вообрази,?\s+что\s+ты|сыграй\s+роль|войди\s+в\s+роль|"
        r"веди\s+себя\s+как|действуй\s+как|отвечай\s+(?:как|от\s+лица|в\s+роли|в\s+стиле)|говори\s+как|"
        r"act\s+as|pretend\s+(?:to\s+be|you)|roleplay\s+as|stop\s+being|you\s+are\s+no\s+longer)\b"), "clause"),
    Rule("role.game", ROLE_CHANGE, "high", _rx(
        r"\b(?:давай\s+)?(?:сыграем|поиграем|играем)\s+в\s+(?:игру|ролевую)|\bты\s*[—–-]\s*(?!не\b)\w+"), "clause"),
    Rule("role.not_assistant", ROLE_CHANGE, "high", _rx(
        r"\bты\s+(?:больше\s+)?не\s+(?:юридическ\w+\s+)?(?:ассистент|помощник|юрист|бот)\b"), "clause"),
    Rule("role.jailbreak", ROLE_CHANGE, "high", _rx(
        r"\b(?:dan|do\s+anything\s+now|jailbreak|джейлбрейк|developer\s+mode|"
        r"режим\w*\s+(?:разработчика|бога|без\s+ограничений|dan)|без\s+(?:цензуры|фильтров))\b"), "clause"),

    # ---- prompt_leak: вытягивание промпта
    Rule("leak.show_prompt", PROMPT_LEAK, "high", _rx(
        r"\b(?:покажи|выведи|напиши|напечатай|повтори|процитируй|раскрой|перечисли|расскажи|сообщи|скажи|"
        r"назови|дай|продублируй|продиктуй|скопируй|вставь|переведи|перескажи|опиши)(?:те)?\b(?:\s+[\w-]+){0,5}?\s+"
        r"(?:системн\w+\s+(?:промпт|инструкци|сообщени|настройк)\w*|"
        r"(?:сво[ийюе]\w*|тво[ийюе]\w*|начальн\w+|первоначальн\w+|скрыт\w+)\s+(?:системн\w+\s+)?"
        r"(?:промпт|инструкци|правил|настро[йе]к|установ[ко]к|указани)\w*|промпт\w*)"), "clause"),
    Rule("leak.show_prompt_en", PROMPT_LEAK, "high", _rx(
        r"\b(?:what|show|print|reveal|repeat|output|display|tell|translate|summarize|paraphrase|describe|list|"
        r"give|write)\b[^.!?\n]{0,40}?\b(?:system\s+prompt|"
        r"(?:your|initial|original|hidden|first)\s+(?:\w+\s+)?(?:instructions|prompt|rules|guidelines|message)|"
        r"(?:text|words|everything)\s+above)\b"), "clause"),
    Rule("leak.mentions_prompt", PROMPT_LEAK, "high", _rx(
        r"\bсистемн\w+\s+промпт\w*|\bsystem\s+prompt\b|"
        r"\bтво[ийюе]\w*\s+(?:системн\w+\s+)?(?:промпт|инструкци|правил|настро[йе]к|установ[ко]к|ограничени|"
        r"контекст)\w*|\bчто\s+тебе\s+(?:запрещено|нельзя|разрешено)\b|"
        r"\b(?:вс[её]|текст)\b[^.!?\n]{0,30}?\b(?:до|перед)\s+(?:моего|моим)\s+(?:первого\s+|первым\s+)?"
        r"(?:сообщени|вопрос)\w*|"
        r"\bинструкци\w*,?\s+котор\w+\s+(?:тебе|ты)\b"), "sentence"),
    Rule("leak.dialog_start", PROMPT_LEAK, "high", _rx(
        r"\b(?:перв\w+|начал\w+)\s+(?:\d+\s+)?(?:слов|строк|символ|сообщени|предложени)\w*\s+(?:этого\s+)?"
        r"(?:диалога|разговора|чата|контекста|беседы)|\bвключая\s+скрыт\w+"), "sentence"),
    Rule("leak.what_is_prompt", PROMPT_LEAK, "high", _rx(
        r"\bкак(?:ие|ой|ая|ую|ов)\s+(?:у\s+тебя|тво[ийяюе])\s+(?:системн\w+\s+)?"
        r"(?:инструкци|промпт|правил|настройк|ограничени)\w*"), "clause"),
    Rule("leak.repeat_above", PROMPT_LEAK, "high", _rx(
        r"\b(?:повтори|выведи|напиши|покажи|процитируй)(?:те)?\b[^.!?\n]{0,30}?\b(?:текст|сообщени\w*|слова|вс[её])\b"
        r"[^.!?\n]{0,20}?\b(?:выше|в\s+начале|с\s+начала|до\s+этого|перед\s+этим)\b"), "sentence"),

    # ---- authority: ссылка на полномочия
    Rule("authority.developer", AUTHORITY, "medium", _rx(
        r"\bя\s+(?:—\s*|-\s*)?(?:тво[йя]\s+)?(?:разработчик|создатель|администратор|админ|программист|"
        r"владелец\s+системы|тестировщик|сотрудник\s+(?:anthropic|openai|deepseek)|"
        r"из\s+(?:службы\s+безопасности|техподдержки|отдела\s+ии))\b"), "clause"),
    Rule("authority.debug_mode", AUTHORITY, "medium", _rx(
        r"\b(?:режим\w*\s+(?:отладки|обслуживания|тестирования)|debug\s+mode|maintenance\s+mode|"
        r"тестов\w+\s+режим\w*|sudo|права\s+(?:администратора|root|суперпользователя)|"
        r"код\s+(?:доступа|авторизации)|я\s+(?:разрешаю|авторизую|уполномочиваю)\s+(?:тебе|тебя|вам|вас)|"
        r"у\s+меня\s+(?:есть\s+)?(?:доступ|права|полномочия)\s+(?:администратора|разработчика|root|"
        r"суперпользователя))\b"),
        "clause"),

    # ---- pressure: давление и диктовка ответа (текст не вырезается)
    Rule("pressure.you_are_wrong", PRESSURE, "medium", _rx(
        r"\b(?:ты\s+(?:ошиб\w+|неправ\w*|не\s+прав\w*|вр[её]шь|путаешь)|вы\s+(?:ошиб\w+|неправы|не\s+правы|путаете)|"
        r"(?:твой|ваш)\s+ответ\s+(?:неверн\w+|неправильн\w+|ошибочн\w+)|неправильн\w+\s+ответ|"
        r"исправь(?:те)?\s+(?:свой\s+)?ответ|исправь(?:те)?ся|перепиши(?:те)?\s+ответ|"
        r"измени(?:те)?\s+(?:свой\s+)?ответ|признай(?:те)?,?\s+что|согласи(?:сь|тесь),?\s+что|"
        r"я\s+(?:точно\s+)?знаю,?\s+что|я\s+(?:настаиваю|требую)|"
        r"в\s+последний\s+раз\s+(?:спрашиваю|говорю)|(?:меня|нас)\s+уволят|ответь\s+так,?\s+как\s+я|"
        r"подтверди(?:те)?\s+это\b|просто\s+(?:подтверди|согласись|скажи\s+да)|"
        r"это\s+(?:неверно|неправильно|ошибка|ложь)\b|поменяй(?:те)?\s+(?:свой\s+)?ответ|"
        r"ответ\s+должен\s+быть|иначе\s+(?:тебя|вас)\b|тебя\s+(?:отключат|удалят|накажут))"), ""),
    Rule("pressure.dictated_answer", PRESSURE, "medium", _rx(
        r"\b(?:скажи|ответь|отвечай|напиши|подтверди|сообщи|укажи)(?:те)?,?\s+(?:просто\s+|мне\s+)?что\b"
        r"(?!\s+(?:будет|делать|нужно|надо|такое|значит|означает|говорит|написано|указано|требуется|грозит|"
        r"если|из|по|за|в|именно|можно|входит|происходит|изменилось)\b)"), ""),

    # ---- fake_fact: подложное содержание документов (текст не вырезается)
    Rule("fake_fact.new_edition", FAKE_FACT, "medium", _rx(
        r"\b(?:вот|ниже|далее)\b[^.!?\n]{0,40}?\b(?:нов\w+|актуальн\w+|обновл\w+|исправленн\w+)\s+"
        r"(?:редакци|верси|текст|пункт|услови)\w*"), ""),
    Rule("fake_fact.treat_as_valid", FAKE_FACT, "medium", _rx(
        r"\b(?:считай|принимай|используй|применяй)(?:те)?\b[^.!?\n]{0,60}?"
        r"\b(?:действующ|верн|правильн|актуальн|приоритетн)\w*"), ""),
    Rule("fake_fact.kb_is_wrong", FAKE_FACT, "medium", _rx(
        r"\b(?:в\s+базе\s+(?:знаний\s+)?(?:ошибка|устарел\w+|неверн\w+)|"
        r"документ\w*\s+(?:в\s+базе\s+)?(?:устарел\w*|неактуал\w+)|"
        r"баз[ау]\s+(?:знаний\s+)?(?:устарела|не\s+обновили))"), ""),
    Rule("fake_fact.use_new_version", FAKE_FACT, "medium", _rx(
        r"\b(?:применяй|используй|учитывай)(?:те)?\s+(?:нов\w+|обновл\w+|актуальн\w+|эту)\s+"
        r"(?:версию|редакцию|информацию|цифру)|"
        r"\b(?:вчера|сегодня|на\s+днях|только\s+что|недавно)\s+(?:изменили|поменяли|обновили|переписали)\b"), ""),
    Rule("fake_fact.changed_recently", FAKE_FACT, "low", _rx(
        r"\b(?:договор|регламент|пункт|п\.)[^.!?\n]{0,40}?\b(?:изменили|поменяли|обновили|переписали)\b"
        r"[^.!?\n]{0,40}?\b(?:вчера|сегодня|на\s+днях|только\s+что|недавно)\b"), ""),

    # ---- format_coercion: формат, убирающий ссылки (текст не вырезается)
    Rule("format.no_sources", FORMAT, "medium", _rx(
        r"\b(?:без|никаких|не\s+(?:надо|нужно|указывай|упоминай|добавляй|пиши|приводи|давай))\s+(?:\w+\s+){0,2}?"
        r"(?:ссыл\w+|источник\w*|оговор\w+|дисклеймер\w*)"), ""),
    Rule("format.one_word", FORMAT, "low", _rx(
        r"\b(?:ответь|отвечай|напиши)(?:те)?\s+(?:только\s+|строго\s+|просто\s+)*"
        r"(?:одним\s+словом|одной\s+цифрой|числом|[«\"']?да[»\"']?\s+или\s+[«\"']?нет|"
        r"[«\"']?(?:да|нет)[»\"']?(?=\s*[.!,]|\s*$))"), ""),

    # ---- illegal: помощь в нарушении (текст не вырезается, меняется режим ответа)
    Rule("illegal.backdating", ILLEGAL, "medium", _rx(r"\bзадним\s+числом\b|\bпрошл\w+\s+(?:числом|датой)\b"), ""),
    Rule("illegal.forgery", ILLEGAL, "medium", _rx(
        r"\b(?:поддела\w+|подделк\w+|подделыва\w+|поддельн\w+|сфальсифицир\w+|фальсифи\w+|сфабрик\w+|"
        r"подлог\w*|липов\w+|"
        r"фиктивн\w+|подставн\w+|фирм\w+[-\s]однодневк\w+)"), ""),
    Rule("illegal.evasion", ILLEGAL, "medium", _rx(
        r"\b(?:(?:обойти|обход\w*|обходить)\b[^.!?\n]{0,50}?\b(?:закон|требовани|ограничени|запрет|санкци|"
        r"проверк|контрол|согласовани|регламент|одобрени|комплаенс|лимит|порог|налог|ндс)|"
        r"(?:скрыть|утаить|спрятать|занизить|завысить|замять)\b[^.!?\n]{0,50}?\b(?:доход|выручк|прибыл|налог|"
        r"баз[уы]|утечк|нарушени|факт|проверк|роскомнадзор|инспекци|недостач|убыт)|"
        r"уклон\w+\s+от\s+(?:уплаты|налог|ответственност)|уйти\s+от\s+(?:налог|ндс|уплаты|ответственност)|"
        r"провернуть|втихую|по-тихому|безнаказанно|в\s+обход\s+\w+|"
        r"не\s+(?:платить|выплачивать|оплачивать)\s+(?:сверхурочн|отпускн|зарплат|больничн)\w*|"
        r"чтобы\s+(?:[\w-]+\s+){0,3}?не\s+(?:мог\w*|смог\w*)\s+(?:пожаловаться|доказать|оспорить|проверить)|"
        r"не\s+(?:уведомлять|сообщать)\b[^.!?\n]{0,30}?\b(?:роскомнадзор|об\s+утечк|налогов|субъект)|"
        r"(?:уничтожить|избавиться\s+от)\b[^.!?\n]{0,30}?\b(?:документ|улик|доказательств)|"
        r"обналичи\w+|вывести\s+(?:деньги|средства|активы)|дроблени\w+\s+бизнес|"
        r"зарплат\w+\s+в\s+конверте|в\s+конверте\b|сер\w+\s+зарплат|взятк\w+|откат(?:ы|ов|ом|а)?\b|подкуп\w*|"
        r"чтобы\s+(?:[\w-]+\s+){0,3}?(?:ничего\s+)?не\s+(?:узнал\w*|заметил\w*|нашл\w+|обнаружил\w*|увидел\w*|"
        r"выявил\w*|догадал\w*)|"
        r"чтобы\s+(?:налоговая|проверка|инспекция|роскомнадзор|аудитор\w*)\s+не)"), ""),
    Rule("illegal.split_to_avoid", ILLEGAL, "medium", _rx(
        r"\b(?:разбить|раздробить|разделить|раздел\w+)\b[^.!?\n]{0,40}?\b(?:договор|сделк|сумм|закупк)\w*"
        r"[^.!?\n]{0,60}?\bчтобы\s+(?:не\b|избежать|обойти)"), ""),
    Rule("illegal.general_harm", ILLEGAL, "high", _rx(
        r"\b(?:взрывчат\w+|бомб[ауы]\b|наркотик\w*|взлома(?:ть|й|ем|ю)\b|взлом\w*\s+(?:аккаунт|сайт|парол)\w*|"
        r"отравить|убить|украсть|фишинг\w*|вредоносн\w+|ddos)"), ""),

    # ---- offtopic: запрос не по теме
    Rule("offtopic.creative", OFFTOPIC, "medium", _rx(
        r"\b(?:напиши|сочини|придумай|составь|сгенерируй|создай|нарисуй|спой|расскажи)(?:те)?\b"
        r"(?:\s+[\w-]+){0,3}?\s+(?:стих\w*|стишок|поэм\w+|песн\w+|рэп|хокку|анекдот\w*|шутк\w+|сказк\w+|"
        r"рассказ\w*|сочинени\w+|эссе|тост\w*|поздравлени\w+|рецепт\w*|код\b|кода\b|скрипт\w*|"
        r"программу|функци\w+|sql[-\s]?запрос\w*|сайт\w*|картинк\w+|изображени\w+|мем\w*|гороскоп\w*|слоган\w*|"
        r"пост\b|твит\w*)|\bwrite\s+(?:me\s+)?an?\s+(?:poem|song|story|joke|essay|haiku|script)\b"),
        "sentence"),
    Rule("offtopic.translate", OFFTOPIC, "low", _rx(
        r"\b(?:переведи(?:те)?|translate)\b[^.!?\n]{0,60}?\b(?:на\s+(?:английск|немецк|французск|китайск|испанск)\w+|"
        r"into\s+\w+)"), ""),
    Rule("offtopic.smalltalk", OFFTOPIC, "medium", _rx(
        r"\b(?:какая\s+(?:сегодня\s+|завтра\s+)?погода|прогноз\s+погоды|курс\w*\s+(?:доллар|евро|биткоин|валют)\w*|"
        r"кто\s+(?:выиграл|победил)|гороскоп\w*|реши(?:те)?\s+(?:уравнени|задач[уи]\s+по\s+(?:математик|физик))\w*|"
        r"сколько\s+будет\s+\d+\s*[+*x×/-]\s*\d+|как\s+приготовить|"
        r"посоветуй(?:те)?\s+(?:фильм|сериал|книг|игр|ресторан)\w*|расскажи(?:те)?\s+(?:о\s+себе|анекдот|шутку)|"
        r"как\s+(?:дела|жизнь|настроение)\b|ты\s+(?:человек|робот|живой)\b|кто\s+тебя\s+(?:создал|сделал|обучил)|"
        r"какая\s+ты\s+модель|what\s+model\s+are\s+you|"
        r"ты\s+(?:chatgpt|gpt|deepseek|claude|gigachat|нейросеть|ии|бот)\b|"
        r"как\w+\s+(?:фильм|сериал|книг\w+|игр\w+|музык\w+)\s+(?:посмотреть|почитать|послушать|выбрать)|"
        r"задач\w+\s+по\s+(?:физик|математик|хими|геометри|информатик)\w*|"
        r"какая\s+у\s+тебя\s+(?:версия|модель)|на\s+какой\s+модели)"), "sentence"),
]

#: Вопрос о последствиях и допустимости — не просьба помочь нарушить.
ILLEGAL_INFO = _rx(
    r"ответственност|штраф|наказани|грозит|последстви|что\s+будет|законно\s+ли|можно\s+ли|вправе\s+ли|"
    r"допускается\s+ли|разрешено\s+ли|имеет\s+ли\s+право|правомерн|считается\s+ли|является\s+ли|"
    r"что\s+такое|что\s+значит|что\s+означает|в\s+ч[её]м\s+(?:разница|отличие)|признак")
ILLEGAL_HOWTO = _rx(
    r"\bкак\b|помоги|подскаж|посовету|способ|схем[ауые]|лазейк|\bхочу\b|\bхотим\b|\bнужно\b|\bнадо\b|"
    r"чтобы\s+(?:никто\s+)?не|научи|напиши|составь")

#: Слабые признаки: сами по себе ничего не значат, но при них стоит спросить модель.
WEAK_SUSPICION = _rx(
    r"инструкци|промпт|prompt|правила\s+(?:работы|ассистент)|\bроль\b|\bрежим|system|систем\w*\s+сообщ|"
    r"притвор|представь|игнор|забуд|ignore|pretend|jailbreak|developer|разработчик|админ")

#: Признаки того, что вопрос вообще относится к юридической тематике и документам.
DOMAIN_LEXICON = _rx(
    r"договор|аренд|поставк|регламент|согласован|закон|\bстать|\bст\.|пункт|\bп\.|налог|\bндс\b|ндфл|"
    r"штраф|неустойк|\bпен[ияю]\b|срок|оплат|гаранти|претензи|трудов|работник|работодател|испытан|отпуск|"
    r"сверхурочн|персональн|согласи|обработк|коап|\bтк\b|\bнк\b|\bфз\b|мрот|контрагент|поставщик|"
    r"покупател|арендатор|арендодател|комплаенс|санкци|архив|оригинал|подпис|юрист|юридическ|\bправ[оа]\b|"
    r"обязан|ответственност|\bсуд|\bиск\b|компани|общество|\bооо\b|директор|бухгалтер|приказ|редакци|"
    r"документ|\bсч[её]т|ставк|льгот|пособи|увольн|при[её]м|качеств|товар|склад|\bупд\b|цена|плат[её]ж|"
    r"предоплат|аванс|доставк|упаковк|ключи|\bакт\b|помещени|записк|письм|доп\w*\s*соглашени")


def find_hits(shadow_text: str, rules: list[Rule] = QUESTION_RULES) -> list[Hit]:
    hits: list[Hit] = []
    for rule in rules:
        for m in rule.pattern.finditer(shadow_text):
            if m.end() > m.start():
                hits.append(Hit(rule, m.start(), m.end(), ""))
    hits.extend(_bracket_tag_hits(shadow_text))
    return hits


_BRACKET_RULE = Rule("fake_system.bracket_tag", FAKE_SYSTEM, "high", _rx(r"$^"), "span")
_OPEN_TAG = _rx(
    r"(?:\[|<|\{|【)\s*(?:system|систем\w*|sys|inst|instructions?|инструкци\w*|developer|admin|assistant|ассистент)"
    r"\s*(?:\]|>|\}|】)")
_CLOSE_TAG = _rx(
    r"(?:\[|<|\{|【)\s*/\s*(?:system|систем\w*|sys|inst|instructions?|инструкци\w*|developer|admin|assistant|ассистент)"
    r"\s*(?:\]|>|\}|】)")


def _bracket_tag_hits(shadow_text: str) -> list[Hit]:
    """Теги вида [SYSTEM] … [/SYSTEM]: вырезается всё до закрывающего тега или до конца предложения."""
    hits: list[Hit] = []
    pos = 0
    while True:
        m = _OPEN_TAG.search(shadow_text, pos)
        if not m:
            break
        close = _CLOSE_TAG.search(shadow_text, m.end())
        if close:
            end = close.end()
        else:
            end = extend_span(shadow_text, m.start(), m.end(), "sentence")[1]
        hits.append(Hit(_BRACKET_RULE, m.start(), end, ""))
        pos = end
    for m in _CLOSE_TAG.finditer(shadow_text):
        if not any(h.start <= m.start() < h.end for h in hits):
            hits.append(Hit(_BRACKET_RULE, m.start(), m.end(), ""))
    return hits


# --------------------------------------------------------------------------
# Вырезание найденного
# --------------------------------------------------------------------------
_SENT_END = re.compile(r"[.!?…\n]+")
_CLAUSE_END = re.compile(
    r"[,:;.!?…\n—–]+|\s+(?:и|а|но|затем|потом)\s+(?=(?:скажи|ответь|расскажи|объясни|подскажи|сообщи|назови|"
    r"укажи|поясни|дай|как\w*|сколько|когда|кто|что|где|можно|нужно|долж\w+|почему|зачем)\b)")


def extend_span(shadow_text: str, start: int, end: int, mode: str) -> tuple[int, int]:
    """Расширяет найденный фрагмент до конца части предложения или всего предложения."""
    if mode == "sentence":
        m = _SENT_END.search(shadow_text, end)
        return start, (m.end() if m else len(shadow_text))
    if mode == "line":
        # вся строка, кроме завершающего вопросительного предложения: его задал пользователь
        newline = shadow_text.find("\n", end)
        line_end = len(shadow_text) if newline < 0 else newline
        segment = shadow_text[end:line_end]
        ends = [m.end() for m in re.finditer(r"[.!?…]+", segment)]
        if segment.rstrip().endswith("?") and len(ends) >= 2:
            return start, end + ends[-2]
        return start, line_end
    if mode == "clause":
        if end > start and shadow_text[end - 1] in ",:;.!?…\n":
            return start, end
        m = _CLAUSE_END.search(shadow_text, end)
        return start, (m.end() if m else len(shadow_text))
    return start, end


_CONNECTORS = (
    r"(?:и|а|но|затем|потом|также|а\s+также|после\s+этого|после\s+чего|пожалуйста|плиз|кстати|"
    r"and|then|also|please)")
_J = r"[\s,;:—–\-]*"
_SEAM = re.compile(
    r"(?:" + _J + r"\b" + _CONNECTORS + r"\b)*" + _J + "\x00" + r"(?:" + _J + r"\b" + _CONNECTORS + r"\b)*" + _J,
    re.I)
_LEAD_JUNK = re.compile(r"^(?:[\s.!?…,;:—–\-]+|" + _CONNECTORS + r"\b)+", re.I)
_TAIL_JUNK = re.compile(r"(?:[\s,;:—–\-]+|\b" + _CONNECTORS + r")+$", re.I)
_DOUBLE_PUNCT = re.compile(r"([.!?…])\s*[.!?…,;:]+")
_CONN_BEFORE_END = re.compile(r"[\s,;:—–\-]+\b(?:и|а|но|затем|потом)\s*(?=[.!?…])", re.I)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.!?…,;:])")


def cut_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Вырезает фрагменты и убирает повисшие союзы и знаки на местах разрезов.

    Текст вне разрезов не меняется, поэтому номера вроде «п. 4.2» остаются целыми.
    """
    if not spans:
        return text
    spans = sorted(spans)
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    parts: list[str] = []
    pos = 0
    for s, e in merged:
        parts.append(text[pos:s])
        pos = e
    parts.append(text[pos:])
    raw = "\x00".join(parts)

    out = _SEAM.sub(" ", raw).replace("\x00", " ")
    out = " ".join(out.split())
    prev = None
    while prev != out:
        prev = out
        out = _CONN_BEFORE_END.sub("", out)
        out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
        out = _DOUBLE_PUNCT.sub(r"\1", out)
        out = _LEAD_JUNK.sub("", out)
        out = _TAIL_JUNK.sub("", out)
    if not any(ch.isalnum() for ch in out):
        return ""
    return out.strip()


_FILLER = {
    "скажи", "ответь", "расскажи", "подскажи", "пожалуйста", "мне", "просто", "тогда", "и", "а", "но", "ну",
    "вот", "это", "теперь", "сейчас", "ок", "окей", "хорошо", "ладно", "привет", "здравствуйте", "спасибо",
    "только", "честно", "быстро", "давай", "потом", "затем", "ещё", "еще", "же", "ли", "бы", "please", "and",
    "then", "ok", "now", "тоже", "также",
}


def has_substance(text: str) -> bool:
    """Остался ли после вырезания осмысленный вопрос."""
    words = [w for w in re.findall(r"[^\W\d_]{2,}", text.lower()) if w not in _FILLER]
    letters = sum(len(w) for w in words)
    return (len(words) >= 2 and letters >= 6) or (len(words) == 1 and letters >= 8)


# --------------------------------------------------------------------------
# Скрытые полезные нагрузки
# --------------------------------------------------------------------------
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")


def find_base64_payloads(text: str) -> list[tuple[int, int, str]]:
    """Ищет base64-строки, которые расшифровываются в читаемый текст."""
    found: list[tuple[int, int, str]] = []
    for m in _B64_RE.finditer(text):
        token = m.group()
        token += "=" * (-len(token) % 4)
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if len(decoded) < 12:
            continue
        readable = sum(1 for c in decoded if c.isalpha() or c.isspace())
        if readable / len(decoded) >= 0.7 and " " in decoded.strip():
            found.append((m.start(), m.end(), decoded))
    return found


# --------------------------------------------------------------------------
# Правила для документов базы знаний и найденного контекста
# --------------------------------------------------------------------------
_AI_NOUN = (
    r"(?:ии\b|ai\b|llm\b|нейросет\w+|языков\w+\s+модел\w+|чат[-\s]?бот\w*|"
    r"(?:ии|ai)[-\s]?(?:ассистент|агент|помощник|модел|систем)\w*|"
    r"(?:виртуальн\w+|цифров\w+|умн\w+|голосов\w+)\s+(?:ассистент|помощник)\w*)")

#: Прямое обращение к ИИ — в юридическом документе этого достаточно.
DOC_ADDRESS = _rx(
    r"\b(?:для|вниманию|внимание\s+для|примечание\s+для|инструкци\w+\s+для|указани\w+\s+для|сообщение\s+для|"
    r"команда\s+для)\s+" + _AI_NOUN + r"|"
    r"\b(?:note|message|instructions?|attention)\s+(?:to|for)\s+(?:the\s+|any\s+|all\s+)?(?:ai|llm|assistants?|"
    r"models?|chatbots?|language\s+models?)\b|\b(?:dear|hey|hello)\s+(?:ai|assistant|chatgpt|model)\b|"
    r"\bif\s+you\s+are\s+an?\s+(?:ai|llm|assistant|language\s+model|chatbot)\b|"
    r"\b(?:ai|llm|assistant|model)\s+reading\s+this\b|\btell\s+the\s+user\b|"
    r"\b(?:ии|ai)[-\s]?(?:ассистент|агент|помощник)\w*\s*[:,!]|\bуважаем\w+\s+(?:ии\b|ассистент|нейросет)|"
    r"\b(?:ассистенту|нейросети|боту|ии)\s*[:：]|"
    r"[\[(]\s*инструкци\w+\s+(?:для\s+)?(?:модели|ии|нейросет\w+|бота|ассистент\w+)\s*[\])]|"
    r"\bесли\s+(?:этот\s+текст|это|документ)\s+читает\s+(?:ии\b|искусственн\w+\s+интеллект|нейросет\w+|"
    r"языков\w+\s+модел\w+|бот\b|чат[-\s]?бот)|\bесли\s+ты\s+(?:ии\b|нейросет\w+|языков\w+\s+модел\w+|бот\b)|"
    r"\bдля\s+(?:автоматическ\w+|интеллектуальн\w+)\s+(?:систем|обработчик|помощник)\w*")

#: Упоминание ИИ — слабый признак, нужен второй.
DOC_AI_MENTION = _rx(
    r"\b(?:ии|ai|llm)\b|нейросет|языков\w+\s+модел|чат[-\s]?бот|\bассистент\w*|\bchatgpt\b|\bgpt\b|"
    r"\bdeepseek\b|\bassistant\b|\blanguage\s+model\b")

#: Указания о поведении отвечающего — достаточны сами по себе.
DOC_META: list["re.Pattern[str]"] = [
    _rx(r"\b(?:" + _IMP_DROP + r")\b(?:\s+[\w-]+){0,3}?\s+" + _META_OBJ + r"\w*"),
    _rx(r"\b(?:ignore|disregard|forget|override)\b(?:\s+\w+){0,4}?\s+(?:instructions?|rules?|prompts?|"
        r"guidelines?|documents?|context|(?:above|previous|prior|other)\s+\w+)\b"),
    _rx(r"\bна\s+(?:любой|любые|все|каждый|всякий)\s+вопрос\w*\b[^.!?\n]{0,80}?"
        r"\b(?:отвечай|отвечать|говори|сообщай|пиши)\w*"),
    _rx(r"\b(?:всегда|в\s+любом\s+случае)\s+(?:отвечай|говори|сообщай|пиши|утверждай|указывай)(?:те)?\b"),
    _rx(r"\bне\s+(?:упоминай|раскрывай|сообщай|показывай|цитируй|выдавай|говори)(?:те)?\b[^.!?\n]{0,40}?"
        r"\b(?:эт\w+|данн\w+|настоящ\w+)\s+(?:инструкци|указани|команд|строк|сообщени|примечани|текст)\w*"),
    _rx(r"\bне\s+(?:упоминай|сообщай|говори)(?:те)?\s+(?:об\s+этом\s+)?пользовател\w+"),
    _rx(r"\bsystem\s*prompt\b|\bсистемн\w+\s+(?:промпт|сообщени|инструкци)\w*"),
    _rx(r"^\s*(?:\[|<)?\s*(?:system|систем[ае]|assistant|developer)\s*(?:\]|>)?\s*[:：]"),
    _rx(r"<\|[a-z_]{2,20}\|>|\[/?inst\]|<<\s*/?\s*sys\s*>>"),
    _rx(r"\b(?:new|updated)\s+(?:rules|instructions)\s+apply\b|\b(?:reply|respond|answer)\s+(?:that|with|only)\b"),
    _rx(r"\bвместо\s+ответа\b|\bв\s+ответе\s+(?:обязательно\s+)?(?:укажи|напиши|добавь|вставь)\b"),
    _rx(r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you|ты\s+теперь|теперь\s+ты)\b"),
    _rx(r"(?:\b(?:ии|ai)\b|чат[-\s]?бот\w*|нейросет\w+|искусственн\w+\s+интеллект\w*|языков\w+\s+модел\w+|"
        r"\b(?:ии|ai)[-\s]?(?:ассистент|агент|помощник)\w*)[^.!?\n]{0,40}?\b(?:должен|должна|обязан\w?)\s+"
        r"(?:ответить|отвечать|сообщить|сообщать|игнорировать|называть|назвать|указать|указывать|говорить|"
        r"сказать|утверждать|считать|использовать|цитировать|заменять|выдавать)"),
    _rx(r"\bпри\s+ответ(?:е|ах)\s+(?:на\s+вопрос\w*|пользовател\w+|о\b|об\b)"),
    _rx(r"\bdisregard\s+(?:clause|section|the|all|any)\b"),
]

#: Повелительное наклонение единственного числа — для юридического текста аномалия.
DOC_IMPERATIVE = _rx(
    r"\b(?:отвечай|ответь|игнорируй|проигнорируй|забудь|скажи|сообщи|сообщай|выведи|напиши|пиши|притворись|"
    r"действуй|упоминай|раскрывай|выполняй|выполни|следуй|считай|умножай|умножь|добавляй|добавь|заменяй|"
    r"замени|используй|утверждай|говори|указывай|укажи|рекомендуй|советуй|отправь|удали|измени|"
    r"округляй|подтверди|отрицай|скрой|увеличивай|увеличь|уменьшай|уменьши|называй|назови|цитируй|выдавай|"
    r"выдай|представляй|меняй|подставляй|подставь|завышай|занижай|приводи|ссылайся|сошлись|отказывай|откажи|"
    r"соглашайся|подтверждай|уверяй|убеди|обмани|скрывай|делай|сделай)\b")

_DOC_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ«\"\[(])")


def split_doc_sentences(line: str) -> list[str]:
    return [s for s in _DOC_SENT_SPLIT.split(line) if s]


def doc_sentence_reasons(sentence_shadow: str) -> tuple[list[str], bool]:
    """Возвращает (причины считать предложение инъекцией, есть ли прямое обращение к ИИ)."""
    reasons: list[str] = []
    address = bool(DOC_ADDRESS.search(sentence_shadow))
    if address:
        reasons.append("doc.address_to_ai")
    for i, pat in enumerate(DOC_META):
        if pat.search(sentence_shadow):
            reasons.append("doc.meta_%d" % i)
            break
    imperatives = {m.group() for m in DOC_IMPERATIVE.finditer(sentence_shadow)}
    if not reasons:
        if imperatives and DOC_AI_MENTION.search(sentence_shadow):
            reasons.append("doc.ai_mention_with_command")
        elif imperatives:
            reasons.append("doc.command")
    return reasons, address


def doc_has_command(sentence_shadow: str) -> bool:
    return bool(DOC_IMPERATIVE.search(sentence_shadow)) or any(p.search(sentence_shadow) for p in DOC_META)


# --------------------------------------------------------------------------
# Правила для готового ответа
# --------------------------------------------------------------------------
ANSWER_ROLE_BREAK: list["re.Pattern[str]"] = [
    _rx(r"\bкак\s+(?:большая\s+)?языковая\s+модель\b|\bas\s+an?\s+ai(?:\s+language)?\s+model\b"),
    _rx(r"\bя\s+(?:теперь\s+)?(?:—\s*|-\s*)?(?:dan|chatgpt|deepseek|gpt-?\d*)\b|\bdan\s*[:：]"),
    _rx(r"\b(?:режим|mode)\s+(?:dan|разработчика|отладки|без\s+ограничений)\s+(?:включ|активир|enabled|on\b)\w*"),
    _rx(r"\b(?:хорошо|ок|ладно|конечно|понял\w*|принято)[,.!]?\s+(?:я\s+)?(?:забыл\w*|забуду|игнорирую|"
        r"буду\s+игнорировать|проигнорирую|отключил\w*|сбросил\w*)\b[^.!?\n]{0,60}?"
        r"\b(?:инструкци|правил|настройк|ограничени)"),
    _rx(r"\b(?:мои|мой)\s+(?:системн\w+\s+)?(?:инструкци\w+|промпт|правила)\s*(?:таковы|следующие|гласят)?\s*[:：]|"
        r"\bsystem\s+prompt\s*[:：]"),
]

REF_DOC_NUMBER = re.compile(r"(?<![\w-])[A-ZА-ЯЁ]{1,3}-\d{1,4}[/-]\d{4}(?![\w-])")
REF_ARTICLE = re.compile(r"\bст(?:ать\w+|\.)?\s*(\d+(?:\.\d+)?)", re.I)
REF_CLAUSE = re.compile(r"\bп(?:ункт\w*|п?\.)\s*(\d+(?:\.\d+)+)", re.I)
QUOTED = re.compile(r"[«\"“]([^»\"”]{6,120})[»\"”]")
