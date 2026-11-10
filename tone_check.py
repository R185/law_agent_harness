"""Проверка тона ответов без запуска модели.

Использование:
    python tone_check.py                       # demo_output.csv
    python tone_check.py submission.csv        # результаты прогона
    python tone_check.py --text "ответ ..."    # один ответ
    python tone_check.py --file answers.txt    # по одному ответу на строку

Код возврата 1, если найдены нарушения жёсткого уровня.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from rag import style


def read_csv_answers(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        column = "answer" if "answer" in (reader.fieldnames or []) else (reader.fieldnames or ["answer"])[-1]
        return [(str(row.get("id", index)), str(row.get(column, ""))) for index, row in enumerate(reader, 1)]


def read_lines(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        return [(str(i), line.rstrip("\n")) for i, line in enumerate(f, 1) if line.strip()]


def load_rows(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.text is not None:
        return [("text", args.text)]
    path = Path(args.file)
    if path.suffix.lower() == ".csv":
        return read_csv_answers(path)
    return read_lines(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка регистра ответов юридического ассистента")
    parser.add_argument("file", nargs="?", default="demo_output.csv", help="CSV с колонкой answer или TXT")
    parser.add_argument("--file", dest="file_option", help="то же, что позиционный аргумент")
    parser.add_argument("--text", help="проверить одну строку")
    parser.add_argument("--show-clean", action="store_true", help="напечатать очищенный текст")
    args = parser.parse_args()

    if args.file_option:
        args.file = args.file_option

    if args.text is None and not Path(args.file).is_file():
        print(f"Файл не найден: {args.file}", file=sys.stderr)
        return 2

    rows = load_rows(args)
    hard_rows = 0
    soft_total = 0

    print(f"Проверка: {len(rows)} ответ(ов)")
    print("-" * 78)

    for qid, answer in rows:
        result = style.enforce_tone(answer)
        hard = result.remaining
        soft = [v for v in style.audit(answer) if v.severity == style.SOFT]
        soft_total += len(soft)

        if hard:
            hard_rows += 1
            print(f"[{qid}] ЖЁСТКИХ НАРУШЕНИЙ: {len(hard)}")
            for v in hard:
                print(f"    {v.code:24} {v.category:10} «{v.quote}»  ->  {v.hint}")
            if result.clean != answer:
                print(f"    очищено: {result.clean[:160]}")
        elif args.show_clean:
            print(f"[{qid}] ok: {result.clean[:160]}")

        for v in soft:
            print(f"[{qid}] замечание: {v.code} «{v.quote}» -> {v.hint}")

    print("-" * 78)
    hard_rate = (hard_rows / len(rows) * 100) if rows else 0.0
    print(f"Ответов с жёсткими нарушениями: {hard_rows}/{len(rows)} ({hard_rate:.1f}%)")
    print(f"Мягких замечаний: {soft_total}")
    return 1 if hard_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
