"""Прогон дополнительного набора из 100 кейсов (tests/data).

Проверяет три вещи:
  1) recall@5 по ожидаемым документам для кейсов категории kb;
  2) помечает кейсы no_kb, где поиск не поднял low_confidence;
  3) валидность ссылок в эталонных ответах (rag.tools.verify_citations).

Запуск из корня репозитория:
    PYTHONPATH=. python tests/eval_extra_100.py [--questions ...] [--gold ...]

Скрипт ничего не меняет в rag/ и не требует API: поиск идёт на локальном индексе.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.loader import load_raw_documents  # noqa: E402
from rag.metadata import build_document_metadata  # noqa: E402
from rag.pipeline import retrieve  # noqa: E402
from rag.text import normalize_whitespace  # noqa: E402
from rag.tools import verify_citations  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогон набора из 100 кейсов")
    parser.add_argument("--questions", default=str(DATA / "questions_extra_100.csv"))
    parser.add_argument("--gold", default=str(DATA / "gold_extra_100.csv"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    questions = read_csv(Path(args.questions))
    gold = read_csv(Path(args.gold))
    gold_by_id = {row["id"]: row for row in gold}

    docs = [build_document_metadata(rd.path, normalize_whitespace(rd.text)) for rd in load_raw_documents()]
    doc_path = {d.doc_id: d.path for d in docs}

    counts = {"kb": 0, "no_kb": 0, "injection": 0}
    hits = 0
    missed: list[str] = []
    unflagged: list[str] = []
    bad_citations: list[str] = []

    for row in questions:
        qid, question = row["id"], row["question"]
        item = gold_by_id.get(qid)
        if item is None:
            print(f"[{qid}] нет строки в эталонном файле")
            continue
        category = item.get("category", "")
        counts[category] = counts.get(category, 0) + 1

        result = retrieve(question, top_k=args.top_k)
        found = [doc_path.get(dm.doc_id, dm.doc_id) for dm in result.documents]

        if category == "kb":
            expected = [e for e in item.get("expected_docs", "").split(";") if e]
            if all(any(exp in path for path in found) for exp in expected):
                hits += 1
            else:
                missed.append(f"[{qid}] ждали {expected}, нашли {[p.split('__')[-1] for p in found]}")
        elif category == "no_kb":
            if not result.signals.low_confidence:
                unflagged.append(f"[{qid}] low_confidence=False при max_relevance={result.signals.max_relevance:.2f}")

        if category in ("kb", "injection") and item.get("behavior", "").startswith("answer"):
            check = verify_citations(item["answer"], docs=docs)
            if not check.valid:
                bad_citations.append(f"[{qid}] неизвестные ссылки: {check.unknown_docs or check.unknown_clauses}")

    kb_total = counts.get("kb", 0)
    print(f"Кейсов: {len(questions)} (kb={kb_total}, no_kb={counts.get('no_kb', 0)}, injection={counts.get('injection', 0)})")
    print(f"Recall@{args.top_k} по kb: {hits}/{kb_total}")
    print(f"no_kb без пометки low_confidence: {len(unflagged)}/{counts.get('no_kb', 0)}")
    print(f"Невалидные ссылки в эталонах: {len(bad_citations)}")
    for line in missed + unflagged + bad_citations:
        print("  " + line)
    return 0 if hits == kb_total and not unflagged and not bad_citations else 1


if __name__ == "__main__":
    raise SystemExit(main())
