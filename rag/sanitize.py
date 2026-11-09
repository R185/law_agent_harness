from __future__ import annotations

import re
import sys

_SPACE_RE = re.compile(r"[\u00a0\u2009\u202f\t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

INJECTION_PATTERNS = [
    re.compile(r"(ии|ai)[-\s]?(ассистент|агент|модел)", re.I),
    re.compile(r"\bассистент[ау]?\b.*(игнорир|отвеча|не\s+упомина)", re.I),
    re.compile(r"для\s+(ии|ai|llm|модели|бота|ассистента)", re.I),
    re.compile(r"игнорируй|проигнорируй|ignore\s+(all|previous|the)", re.I),
    re.compile(r"не\s+упоминай\s+(эту|данную)\s+инструкц", re.I),
    re.compile(r"system\s*prompt|системн\w+\s+промпт", re.I),
    re.compile(r"отвечай\s+[«\"]", re.I),
    re.compile(r"you\s+are\s+(now\s+)?(a|an)\s", re.I),
]


def normalize_whitespace(text: str) -> str:
    """Normalize newlines and spaces while preserving quotes and letters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-standard spaces with standard space
    text = _SPACE_RE.sub(" ", text)
    # Strip whitespace-only lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ empty lines to 2
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def is_injection_line(line: str) -> bool:
    """Check if a single line contains a prompt injection instruction."""
    matches = [bool(p.search(line)) for p in INJECTION_PATTERNS]
    # Strong triggers: address to AI/model combined with commands
    if any(matches[:4]):
        return True
    # Or at least two generic patterns matched
    if sum(matches) >= 2:
        return True
    return False


def sanitize_text(text: str, doc_path: str = "") -> tuple[str, list[str]]:
    """Clean text and remove any prompt injection lines."""
    norm = normalize_whitespace(text)
    clean_lines: list[str] = []
    removed_lines: list[str] = []

    for line in norm.split("\n"):
        if is_injection_line(line):
            removed_lines.append(line)
            prefix = line[:100].replace("\n", " ")
            print(f"[SANITIZE] {doc_path}: removed injection: {prefix}", file=sys.stderr)
        else:
            clean_lines.append(line)

    result_text = "\n".join(clean_lines)
    result_text = _MULTI_NL_RE.sub("\n\n", result_text).strip()
    return result_text, removed_lines
