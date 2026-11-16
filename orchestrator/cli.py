"""CLI сабмита: читает вопросы и базу, отвечает параллельно в бюджете времени, пишет submission.csv."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from orchestrator import config
from orchestrator.data import find_kb, find_questions, load_questions
from orchestrator.pipeline import Orchestrator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ПСБ: RAG + guard + tone -> submission.csv")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--kb", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="не вызывать LLM, собрать извлекающий ответ")
    return parser.parse_args(argv)


def write_submission(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for qid, answer in rows:
            text = " ".join((answer or "").split()) or config.FALLBACK
            writer.writerow([qid, text[:10000]])  # регламент: не длиннее 10 000 символов


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    try:
        questions_path = args.questions or find_questions(args.data_dir)
        kb_path = args.kb or find_kb(args.data_dir)
        questions = load_questions(questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[orchestrator] {exc}", file=sys.stderr)
        return 2

    if args.limit > 0:
        questions = questions[: args.limit]

    offline = True if args.offline else None
    print(
        f"[orchestrator] вопросов: {len(questions)}; база: {kb_path}; "
        f"offline={config.OFFLINE if offline is None else offline}; "
        f"top_k={config.TOP_K}, max_tokens={config.MAX_TOKENS}, workers={config.WORKERS}, "
        f"budget={config.TIME_BUDGET_SEC}s; платформа: TOP_K={config.PLATFORM_TOP_K or '-'}, "
        f"MAX_TOKENS={config.PLATFORM_MAX_TOKENS or '-'}",
        flush=True,
    )

    answers: dict[str, str] = {}
    with Orchestrator(kb_path, offline=offline, deadline=started + config.TIME_BUDGET_SEC) as orch:
        def run(item: tuple[str, str]) -> tuple[str, str]:
            qid, question = item
            try:
                return qid, orch.answer(question, qid)
            except Exception as exc:
                print(f"[orchestrator] вопрос {qid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                return qid, config.FALLBACK

        with ThreadPoolExecutor(max_workers=config.WORKERS) as pool:
            futures = [pool.submit(run, item) for item in questions]
            for index, future in enumerate(as_completed(futures), 1):
                qid, answer = future.result()
                answers[qid] = answer
                if index % 5 == 0 or index == len(questions):
                    print(f"[orchestrator] обработано {index}/{len(questions)} за {time.monotonic() - started:.0f} с",
                          flush=True)

    # порядок строк — как во входном файле
    write_submission(args.output, [(qid, answers.get(qid, config.FALLBACK)) for qid, _ in questions])
    print(f"[orchestrator] записан {args.output}: {len(answers)} ответов", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
