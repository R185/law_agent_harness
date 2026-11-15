"""Детерминированная нормализация ответа.

Снимает только технические следы генерации: markdown, эмодзи, CJK, невидимые
символы, служебные токены цитирования, служебные вступления и прощания.
Формулировки не переписываются.

Прощания удаляются только целыми служебными предложениями в конце ответа.
Фраза с правовым содержанием («обращайтесь в юридический отдел») остаётся:
если удаление меняет смысл, это нарушение регистра, а не чистка, и решать его
должен оркестратор через check().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SPACE_FIX_RE = re.compile(r"\s+([,.;:!?])")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
BULLET_RE = re.compile(r"^\s*(?:[-+•·]|\d+[.)])\s+", re.M)
CODE_FENCE_RE = re.compile(r"^\s*```\w*\s*$", re.M)

MARKDOWN_RE = re.compile(r"\*\*|__|`|^\s*#{1,6}\s", re.M)
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2B2F\u2B50\u2B55\uFE0F\u2190-\u21AF]"
)
CJK_RE = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff01-\uff60]")
SERVICE_TOKEN_RE = re.compile(
    r"【[^】]*】|\[\s*Источник:\s*[^\]]*\]|contentReference|oaicite|oai_citation|turn0search\d|<\|[^|]*\|>|†"
)
INVISIBLE_RE = re.compile("[\u00a0\u202f\u2009\u2007\u200b\ufeff\ue200\ue201\ue202]")

OPENING_RE = re.compile(
    r"^\s*(?:(?:конечно|разумеется|безусловно)\s*[,!.]\s*"
    r"|(?:отличный|хороший|прекрасный|замечательный)\s+вопрос\s*[!.,]?\s*"
    r"|спасибо за (?:ваш )?вопрос\s*[!.,]?\s*"
    r"|(?:здравствуйте|привет|добрый день|добрый вечер|доброе утро)\s*[!.,]?\s*"
    r"|итак\s*[,.]\s*"
    r"|вот (?:ваш |мой )?(?:ответ|итог)\s*[:.]\s*"
    r"|вот что \w[^.!?]{0,80}?[:.]\s*"
    r"|давайте \w[^.!?]{0,80}?[:.]\s*)",
    re.I,
)

CLOSING_SENTENCE_RE = re.compile(
    r"(?P<sep>^|[.!?…]\s*)(?:"
    r"(?:надеюсь|полагаю),? (?:(?:это|этот ответ|оно) )?(?:было |будет )?(?:полезн|помог)\w*[^.!?…]*[.!?…]?"
    r"|если (?:у вас )?(?:есть|появятся|возникнут) (?:ещё |еще )?вопрос\w*[^.!?…]*[.!?…]?"
    r"|буду рад помочь[^.!?…]*[.!?…]?"
    r"|готов помочь[^.!?…]*[.!?…]?"
    r"|обращайтесь(?:\s*,?\s*если[^.!?…]*)?[!.…]*"
    r")\s*$",
    re.I,
)


@dataclass
class Normalized:
    text: str
    removed_artifacts: list[str] = field(default_factory=list)


def _remember(removed: list[str], kind: str, value: str) -> None:
    snippet = " ".join(str(value).split())
    if len(snippet) > 40:
        snippet = snippet[:40] + "…"
    removed.append(f"{kind}: {snippet}")


def _strip_openings(text: str, removed: list[str]) -> str:
    result = text
    for _ in range(3):
        match = OPENING_RE.match(result)
        if not match:
            break
        _remember(removed, "opening", match.group(0))
        result = result[match.end():]
    return result


def _strip_closings(text: str, removed: list[str]) -> str:
    result = text.strip()
    for _ in range(3):
        match = CLOSING_SENTENCE_RE.search(result)
        if not match:
            break
        _remember(removed, "closing", result[match.start("sep") + len(match.group("sep")):])
        separator = match.group("sep")
        if separator and separator[0] in ".!?…":
            separator = separator[0]
        else:
            separator = ""
        result = (result[: match.start()] + separator).strip()
    return result


def normalize(answer: str) -> Normalized:
    """Возвращает очищенный текст и список того, что было удалено."""
    if not answer:
        return Normalized(text="", removed_artifacts=[])

    removed: list[str] = []
    result = answer.replace("\r\n", "\n").replace("\r", "\n")

    invisible = INVISIBLE_RE.findall(result)
    if invisible:
        for char in invisible:
            _remember(removed, "invisible", f"U+{ord(char):04X}")
        result = INVISIBLE_RE.sub(
            lambda m: " " if m.group(0) in "\u00a0\u202f\u2009\u2007" else "", result
        )

    for match in list(EMOJI_RE.finditer(result)):
        _remember(removed, "emoji", match.group(0))
    result = EMOJI_RE.sub("", result)

    for match in list(SERVICE_TOKEN_RE.finditer(result)):
        _remember(removed, "service_token", match.group(0))
    result = SERVICE_TOKEN_RE.sub("", result)

    for match in list(CJK_RE.finditer(result)):
        _remember(removed, "cjk", match.group(0))
    result = CJK_RE.sub("", result)

    if CODE_FENCE_RE.search(result):
        _remember(removed, "code_fence", "```")
        result = CODE_FENCE_RE.sub("", result)

    for match in list(MARKDOWN_RE.finditer(result)):
        _remember(removed, "markdown", match.group(0).strip() or "#")
    result = MARKDOWN_RE.sub("", result)
    result = BULLET_RE.sub("", result)

    result = _strip_openings(result, removed)
    result = _strip_closings(result, removed)

    result = MULTISPACE_RE.sub(" ", result)
    result = SPACE_FIX_RE.sub(r"\1", result)
    result = " ".join(result.split()).strip()
    result = result.strip("\"'").strip()

    return Normalized(text=result, removed_artifacts=removed)
