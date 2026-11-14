from __future__ import annotations

import csv
import io
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from rag import config


@dataclass
class RawDocument:
    path: str
    text: str


def find_file(*names: str) -> Path:
    candidates = []
    roots = [
        config.DATA_DIR,
        Path(__file__).resolve().parent.parent / config.DATA_DIR,
        Path.cwd(),
    ]
    for root in roots:
        for name in names:
            p = root / name
            candidates.append(p)
            if p.is_file() or p.is_dir():
                return p
    raise FileNotFoundError(f"Не найден ни один из файлов/директорий: {', '.join(names)}; проверены пути: {[str(c) for c in candidates]}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv требуются колонки id и question")
        rows = [(str(r["id"]).strip(), str(r["question"]).strip()) for r in reader]
    if not rows or any(not qid for qid, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


def _fix_zip_filename(name: str) -> str:
    # Handle cp437 encoding issue on cyrillic filenames inside standard zip
    if any(c in name for c in ("╨", "╤", "Р", "С")):
        try:
            return name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def read_docx(raw: bytes) -> str:
    """Extract text from docx bytes without external dependencies."""
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    parts: list[str] = []
    for elem in root.iter():
        if elem.tag == f"{ns}p":
            p_text = "".join(t.text or "" for t in elem.iter(f"{ns}t")).strip()
            if p_text:
                parts.append(p_text)
        elif elem.tag == f"{ns}tbl":
            # Extract tables row by row
            for row in elem.iter(f"{ns}tr"):
                cells = []
                for cell in row.iter(f"{ns}tc"):
                    cell_text = "".join(t.text or "" for t in cell.iter(f"{ns}t")).strip()
                    cells.append(cell_text)
                if any(cells):
                    parts.append(" | ".join(cells))
    return "\n".join(parts)


def _decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_raw_documents(source_path: Path | None = None) -> list[RawDocument]:
    """Load raw documents from zip archive or local folder."""
    if source_path is None:
        try:
            source_path = find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge-base")
        except FileNotFoundError:
            # Fallback to local knowledge-base in project root if exists
            local_kb = Path("knowledge-base")
            if local_kb.is_dir():
                source_path = local_kb
            else:
                raise

    documents: list[RawDocument] = []

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            for item in sorted(archive.namelist()):
                if item.endswith("/") or item.startswith("__MACOSX/") or "/." in item or item.startswith("."):
                    continue
                clean_name = _fix_zip_filename(item)
                suffix = Path(clean_name).suffix.lower()
                if suffix not in {".txt", ".md", ".docx"}:
                    continue
                raw = archive.read(item)
                try:
                    text = read_docx(raw) if suffix == ".docx" else _decode_text(raw)
                except Exception as exc:
                    print(f"[LOADER] Пропущен нечитаемый файл {clean_name}: {exc}", file=sys.stderr)
                    continue
                if text.strip():
                    documents.append(RawDocument(path=clean_name, text=text.strip()))

    elif source_path.is_dir():
        for root, _, files in os.walk(source_path):
            for file_name in sorted(files):
                if file_name.startswith(".") or file_name.startswith("~"):
                    continue
                file_path = Path(root) / file_name
                suffix = file_path.suffix.lower()
                if suffix not in {".txt", ".md", ".docx"}:
                    continue
                rel_path = str(file_path.relative_to(source_path))
                try:
                    raw = file_path.read_bytes()
                    text = read_docx(raw) if suffix == ".docx" else _decode_text(raw)
                except Exception as exc:
                    print(f"[LOADER] Пропущен файл {rel_path}: {exc}", file=sys.stderr)
                    continue
                if text.strip():
                    documents.append(RawDocument(path=rel_path, text=text.strip()))
    else:
        raise ValueError(f"Неподдерживаемый путь базы знаний: {source_path}")

    if not documents:
        raise ValueError(f"В источнике {source_path} нет читаемых документов (.txt, .md, .docx)")

    return documents
