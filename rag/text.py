from __future__ import annotations

import re
from typing import Callable

_RU_ENDINGS = (
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ов", "ев", "ей",
    "ий", "ый", "ой", "ым", "им", "ах", "ях", "ом", "ем", "ам", "ям",
    "ы", "и", "а", "я", "у", "ю", "е", "о"
)
_RU_SUFFIXES = ("ательн", "ительн", "ани", "ени", "тельн", "тель", "ован", "ирован")


def _deeper_stem(s: str) -> str:
    for suf in _RU_SUFFIXES:
        if s.endswith(suf) and len(s) - len(suf) >= 3:
            return s[:-len(suf)]
    return s


# Try importing snowballstemmer, fallback to rule-based stemmer
try:
    import snowballstemmer  # type: ignore

    _snowball = snowballstemmer.stemmer("russian")
    _stem_fn: Callable[[str], str] = lambda word: _snowball.stemWord(word)
except Exception:
    def _fallback_stem(word: str) -> str:
        if len(word) <= 3:
            return word
        for end in _RU_ENDINGS:
            if word.endswith(end) and len(word) - len(end) >= 3:
                word = word[:-len(end)]
                break
        word = _deeper_stem(word)
        return word

    _stem_fn = _fallback_stem


STOP_WORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "о", "об", "от", "до", "за", "из",
    "у", "не", "ли", "же", "а", "но", "что", "как", "это", "для", "или", "при",
    "то", "бы", "мы", "нас", "наш", "нам", "вы", "я", "он", "она", "они", "его",
    "ее", "их", "так", "все", "еще", "уже", "можно", "нужно", "быть", "был", "была", "были"
}

TOKEN_RE = re.compile(r"[a-zа-я0-9]+(?:[-./][a-zа-я0-9]+)*")
DASH_RE = re.compile(r"[‐-‒–—−]")
QUOTE_RE = re.compile(r"[«»“”„\"']")
DOCNUM_RE = re.compile(r"(?:№\s*)?([A-ZА-ЯЁ]{1,4})[\s-]?(\d{1,4})(?:[/-](\d{2,4}))?", re.I)


def norm_str(s: str) -> str:
    """Normalize characters, lowercasing and replacing ё, quotes, dashes."""
    s = s.lower().replace("ё", "е")
    s = QUOTE_RE.sub(" ", s)
    s = DASH_RE.sub("-", s)
    return s


def canonical_doc_num(prefix: str, num: str, year: str | None = None) -> str:
    """Produce normalized canonical doc number, e.g. п-17/2024."""
    p = prefix.lower().strip()
    n = f"{int(num):02d}"
    if year:
        y = year.strip()
        if len(y) == 2:
            y = "20" + y
        return f"{p}-{n}/{y}"
    return f"{p}-{n}"


def extract_doc_numbers(text: str) -> list[str]:
    """Find and normalize document numbers in text or filename."""
    results: list[str] = []
    # Search for № XX-NN/YYYY or XX-NN-YYYY or explicit № ...
    for match in re.finditer(r"(?:№\s*([A-Za-zА-Яа-яЁё]{1,4})[\s-]?(\d{1,4})(?:[/-](\d{2,4}))?)|(?:\b([A-Za-zА-Яа-яЁё]{1,4})[-_](\d{1,4})(?:[/-](\d{2,4}))\b)", text):
        m = match.groups()
        if m[0] is not None:
            prefix, num, year = m[0], m[1], m[2]
        else:
            prefix, num, year = m[3], m[4], m[5]
        # Ignore common non-document abbreviations like ст (статья), п (пункт без года), г (год)
        p_norm = prefix.lower()
        if p_norm in {"ст", "ч", "пп"} and not year:
            continue
        canon = canonical_doc_num(prefix, num, year)
        if canon not in results:
            results.append(canon)
    return results


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Tokenize text into words, compounds, sub-tokens, and stems."""
    normalized = norm_str(text)
    raw_tokens = TOKEN_RE.findall(normalized)
    tokens: list[str] = []

    for t in raw_tokens:
        if t in STOP_WORDS:
            continue

        # Keep compound token (e.g. 'альфа-трейд', 'п-17/2024')
        tokens.append(t)

        has_digits = any(c.isdigit() for c in t)
        is_compound = ("-" in t or "/" in t or "." in t)

        if is_compound:
            # Also add constituent parts
            parts = re.split(r"[-/.]", t)
            for part in parts:
                if len(part) >= 2 and part not in STOP_WORDS:
                    tokens.append(part)
                    if stem and not any(c.isdigit() for c in part):
                        st = _stem_fn(part)
                        if st and st != part:
                            tokens.append(st)

        # Apply stemming if not compound and does not contain digits
        if stem and not is_compound and not has_digits and len(t) >= 3:
            st = _stem_fn(t)
            if st and st != t:
                tokens.append(st)
            deep = _deeper_stem(st)
            if deep and deep != st and deep != t:
                tokens.append(deep)

    return tokens


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance without external dependencies."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
