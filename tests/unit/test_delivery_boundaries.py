"""T039: delivery boundaries for ``decision/`` and ``contracts/``.

Sources are read as UTF-8. The repository root is
``Path(__file__).resolve().parents[2]``. These tests do not skip, do not
open a socket, and do not read an API key.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Exact spellings. ``from openai.types`` contains ``from openai``. The field
# name ``question`` and a question mark are not in these lists.
_OPENAI_IMPORTS = ("import openai", "from openai")
_PACKAGE_IMPORTS = (
    "import search",
    "from search",
    "import guard",
    "from guard",
    "import tone",
    "from tone",
)
_ORIGIN_IMPORTS = (
    "import origin.guard",
    "from origin.guard",
    "import origin.tone",
    "from origin.tone",
    "import origin.RAG",
    "from origin.RAG",
    "import origin/guard",
    "from origin/guard",
)
_IMPORT_START = re.compile(r"[ \t]*(?:import|from)\b")

_CLOCK = ("date.today", "datetime.now")
_RETIRED = (
    "no_chunks",
    "low_retrieval_score",
    "topic_not_in_kb",
    "ambiguous_counterparty",
    "missing_contract_reference",
    "partial_legal_basis",
    "sufficient_context",
)
_TEMPLATES = ("Уточните", "Какой договор", "Назовите контрагента")
_BANNED_TOP = frozenset({"openai", "search", "guard", "tone"})
_BANNED_ORIGIN = ("origin.guard", "origin.tone", "origin.RAG")


def _sources() -> list[tuple[Path, str]]:
    loaded: list[tuple[Path, str]] = []
    for folder in ("decision", "contracts"):
        root = REPO / folder
        assert root.is_dir(), folder
        paths = sorted(path for path in root.rglob("*.py") if path.is_file())
        assert paths, f"{folder}/ has no Python files"
        for path in paths:
            loaded.append((path, path.read_text(encoding="utf-8")))
    return loaded


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _text_hits(sources: list[tuple[Path, str]], needles: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    for path, source in sources:
        relative = _relative(path)
        for lineno, line in enumerate(source.splitlines(), start=1):
            for needle in needles:
                if needle in line:
                    problems.append(f"{relative}:{lineno}: {line.strip()}")
    return problems


def _import_modules(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module:
                found.append((node.lineno, module))
            for alias in node.names:
                if alias.name == "*":
                    continue
                qualified = f"{module}.{alias.name}" if module else alias.name
                found.append((alias.lineno, qualified))
    return found


def _banned_module(module: str, *, openai: bool, packages: bool) -> str | None:
    if not module:
        return None
    top = module.split(".", 1)[0]
    if openai and top == "openai":
        return "openai"
    if packages and top in _BANNED_TOP and top != "openai":
        return top
    if packages:
        for origin in _BANNED_ORIGIN:
            if module == origin or module.startswith(origin + "."):
                return origin
    return None


def _ast_import_hits(
    sources: list[tuple[Path, str]],
    *,
    openai: bool,
    packages: bool,
) -> list[str]:
    problems: list[str] = []
    for path, source in sources:
        relative = _relative(path)
        try:
            modules = _import_modules(source)
        except SyntaxError:
            continue
        lines = source.splitlines()
        seen: set[tuple[int, str]] = set()
        for lineno, module in modules:
            banned = _banned_module(module, openai=openai, packages=packages)
            if banned is None or (lineno, banned) in seen:
                continue
            seen.add((lineno, banned))
            text = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else module
            problems.append(f"{relative}:{lineno}: {text}")
    return problems


def _origin_guard_path_hits(sources: list[tuple[Path, str]]) -> list[str]:
    """Import statements whose text contains the path ``origin/guard``."""

    problems: list[str] = []
    for path, source in sources:
        relative = _relative(path)
        lines = source.splitlines()
        index = 0
        while index < len(lines):
            code = lines[index].split("#", 1)[0]
            if _IMPORT_START.match(code) is None:
                index += 1
                continue
            start = index
            parts = [code]
            depth = code.count("(") - code.count(")")
            continued = depth > 0 or code.rstrip().endswith("\\")
            index += 1
            while continued and index < len(lines):
                code = lines[index].split("#", 1)[0]
                parts.append(code)
                depth += code.count("(") - code.count(")")
                continued = depth > 0 or code.rstrip().endswith("\\")
                index += 1
            if "origin/guard" not in "\n".join(parts):
                continue
            for offset, part in enumerate(parts):
                if "origin/guard" not in part:
                    continue
                lineno = start + offset + 1
                problems.append(f"{relative}:{lineno}: {lines[lineno - 1].strip()}")
    return problems


def _assert_clean(problems: list[str]) -> None:
    assert not problems, "boundary violation:\n" + "\n".join(problems)


def test_decision_and_contracts_do_not_import_openai() -> None:
    sources = _sources()
    problems = _text_hits(sources, _OPENAI_IMPORTS)
    problems.extend(_ast_import_hits(sources, openai=True, packages=False))
    _assert_clean(problems)


def test_decision_and_contracts_do_not_import_search_guard_or_tone() -> None:
    sources = _sources()
    problems = _text_hits(sources, _PACKAGE_IMPORTS)
    problems.extend(_text_hits(sources, _ORIGIN_IMPORTS))
    problems.extend(_ast_import_hits(sources, openai=False, packages=True))
    _assert_clean(problems)


def test_decision_and_contracts_do_not_call_today_or_now() -> None:
    _assert_clean(_text_hits(_sources(), _CLOCK))


def test_decision_and_contracts_omit_retired_reason_codes() -> None:
    _assert_clean(_text_hits(_sources(), _RETIRED))


def test_decision_and_contracts_omit_user_facing_text() -> None:
    _assert_clean(_text_hits(_sources(), ("user_facing_text",)))


def test_decision_and_contracts_omit_ready_made_questions() -> None:
    """The catalog fragments are absent.

    ``question`` and ``?`` stay allowed: they belong to the request models.
    """

    _assert_clean(_text_hits(_sources(), _TEMPLATES))


def test_origin_guard_is_not_imported() -> None:
    sources = _sources()
    problems = _text_hits(
        sources,
        ("import origin/guard", "from origin/guard"),
    )
    problems.extend(_origin_guard_path_hits(sources))
    _assert_clean(problems)


def test_origin_guard_is_unchanged_against_fixing_llm_judge() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--", "origin/guard"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert tracked.returncode == 0, tracked.stderr
    if not tracked.stdout.strip():
        return

    diff = subprocess.run(
        ["git", "diff", "--name-only", "fixing_llm-judge", "--", "origin/guard"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.strip() == "", diff.stdout

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "origin/guard"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert status.returncode == 0, status.stderr
    assert status.stdout.strip() == "", status.stdout


def test_requirements_txt_does_not_contain_pytest() -> None:
    text = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "pytest" not in text


def test_requirements_dev_txt_contains_pytest() -> None:
    text = (REPO / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest" in text
    assert "pydantic" in text
    assert "httpx" in text
