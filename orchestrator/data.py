"""Чтение вопросов и поиск файлов платформы."""

from __future__ import annotations

import csv
from pathlib import Path

from orchestrator import config


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = [
            (str(row["id"]).strip(), str(row["question"]).strip())
            for row in reader
        ]
    if not rows or any(not qid or not question for qid, question in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


def find_questions(data_dir: Path) -> Path:
    roots = [data_dir, data_dir / "data", Path.cwd(), Path.cwd() / "data"]
    for root in roots:
        path = root / "questions.csv"
        if path.is_file():
            return path
    raise FileNotFoundError(f"questions.csv не найден в {data_dir}")


def find_kb(data_dir: Path) -> Path:
    names = (
        "kb.zip",
        "knowledge-base.zip",
        "knowledge_base.zip",
        "knowledge-base",
        "knowledge_base",
    )
    roots = [data_dir, data_dir / "data", Path.cwd(), Path.cwd() / "data"]
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file() or path.is_dir():
                return path
    raise FileNotFoundError(f"База знаний (kb.zip / knowledge-base.zip) не найдена в {data_dir}")
