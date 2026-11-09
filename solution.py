from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rag import config
from rag.answer import generate_answer
from rag.api import ApiClient
from rag.context import build_context_cards
from rag.index import KBIndex
from rag.loader import find_file, load_questions, load_raw_documents
from rag.metadata import build_doc_graph, build_document_metadata
from rag.query import analyze_query
from rag.retrieve import EntityResolver, retrieve_candidates
from rag.sanitize import sanitize_text


def process_question(
    position: int,
    qid: str,
    raw_question: str,
    kb_index: KBIndex,
    graph,
    resolver: EntityResolver,
) -> tuple[int, str]:
    """Process a single question end-to-end and return (position, answer_line)."""
    t0 = time.time()
    try:
        with ApiClient(timeout=90.0) as client:
            # 1. Parse dialogue & query analysis
            dialog, analysis = analyze_query(raw_question, client=client)

            # 2. Retrieve candidates
            dense_fn = (
                (lambda q, top_k: kb_index.dense_search(q, client, top_k=top_k))
                if kb_index.embeddings is not None and config.EMBED_KEY
                else None
            )

            ranked_chunks, low_conf = retrieve_candidates(
                analysis=analysis,
                chunks=kb_index.chunks,
                docs=kb_index.docs,
                graph=graph,
                bm25=kb_index.bm25,
                resolver=resolver,
                dense_scores_fn=dense_fn,
                client=client,
            )

            # 3. Build context cards
            context = build_context_cards(
                ranked_chunks=ranked_chunks,
                docs=kb_index.docs,
                graph=graph,
                analysis=analysis,
                is_ambiguous=bool(analysis.ambiguous_entities),
            )

            # 4. Generate answer
            answer = generate_answer(
                client=client,
                dialog=dialog,
                analysis=analysis,
                context=context,
                low_confidence=low_conf,
            )

            elapsed = time.time() - t0
            top_info = (
                f"{ranked_chunks[0][0].header[:40]}... (score: {ranked_chunks[0][1]:.2f})"
                if ranked_chunks
                else "no chunks"
            )
            print(
                f"[SOLVE] Q{qid} done in {elapsed:.2f}s | Top: {top_info}",
                file=sys.stderr,
                flush=True,
            )
            return position, answer
    except Exception as exc:
        print(f"[SOLVE] Error on Q{qid}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return position, config.FALLBACK


def main() -> None:
    t_start = time.time()

    # 1. Load questions
    try:
        q_path = find_file("questions.csv", "demo_questions.csv")
    except FileNotFoundError:
        q_path = Path("demo_questions.csv")

    questions = load_questions(q_path)
    print(f"[MAIN] Загружено вопросов: {len(questions)} из {q_path}", flush=True)

    # 2. Load and sanitize documents
    raw_docs = load_raw_documents()
    docs = []
    for rd in raw_docs:
        clean_text, removed = sanitize_text(rd.text, rd.path)
        doc = build_document_metadata(rd.path, clean_text, removed)
        docs.append(doc)

    print(f"[MAIN] Загружено документов базы знаний: {len(docs)}", flush=True)

    # 3. Build graph & entity resolver
    graph = build_doc_graph(docs)
    resolver = EntityResolver(docs, graph)

    # 4. Build index
    kb_index = KBIndex()
    kb_index.build_chunks_and_bm25(docs)
    print(f"[MAIN] Построено чанков: {len(kb_index.chunks)}, BM25 готов", flush=True)

    # 5. Check precomputed embeddings in index/ folder
    index_dir = Path(__file__).resolve().parent / "index"
    if index_dir.is_dir():
        if kb_index.load_precomputed_embeddings(index_dir):
            print(f"[MAIN] Загружены предрассчитанные векторы из {index_dir}", flush=True)
        else:
            print("[MAIN] Предрассчитанные векторы не совпали или отсутствуют", flush=True)

    # 6. Execute questions in ThreadPool
    answers = [config.FALLBACK] * len(questions)

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [
            pool.submit(
                process_question,
                pos,
                qid,
                q_text,
                kb_index,
                graph,
                resolver,
            )
            for pos, (qid, q_text) in enumerate(questions)
        ]

        for finished, fut in enumerate(as_completed(futures), 1):
            idx, ans = fut.result()
            answers[idx] = ans
            if finished % 5 == 0 or finished == len(questions):
                print(f"[MAIN] Обработано {finished}/{len(questions)}", flush=True)

    # 7. Write submission.csv
    out_path = Path("submission.csv")
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i]])

    print(
        f"[MAIN] Успешно создан submission.csv ({len(questions)} ответов) за {time.time() - t_start:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
