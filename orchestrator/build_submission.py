"""Сборка ZIP для платформы: solution.py + rag + tone + guard + judge."""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "submission.zip"

REQUIREMENTS = """httpx==0.28.1
numpy==2.2.6
pydantic==2.14.2
"""


def _package_files(package: str, patterns: tuple[str, ...] = ("*.py",)) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        for path in sorted((ROOT / package).rglob(pattern)):
            if "__pycache__" in path.parts or "tests" in path.parts:
                continue
            if package == "guard" and path.name in {"guard_cases.py", "guard_eval.py", "guard_e2e.py"}:
                continue
            if package == "orchestrator" and path.name == "forward.py":
                continue
            files.append(path)
    return files


def build(output: Path = OUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = (
        [ROOT / "solution.py"]
        + _package_files("rag")
        + _package_files("tone")
        + _package_files("guard")
        + _package_files("orchestrator")
        + _package_files("decision", ("*.py", "*.json"))
        + _package_files("contracts")
    )
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("requirements.txt", REQUIREMENTS)
    return output


if __name__ == "__main__":
    path = build()
    print(path)
