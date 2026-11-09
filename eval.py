from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from rag import config
from rag.bm25 import BM25Index
from rag.chunker import chunk_document
from rag.loader import find_file, load_questions, load_raw_documents
from rag.metadata import build_doc_graph, build_document_metadata
from rag.query import analyze_query
from rag.retrieve import EntityResolver, retrieve_candidates
from rag.sanitize import sanitize_text

GOLD_RETRIEVAL: dict[str, list[str]] = {
    "1": ["А-08-2024"],
    "2": ["ТК_ст70"],
    "3": [],  # Out of KB
    "4": ["А-08-2024"],
    "5": ["ред 2023", "ред 2025"],
    "6": ["П-17-2024"],
    "7": ["А-08-2024"],
    "8": ["П-17-2024"],
    "9": ["ПТ-05-2023"],
    "10": ["ред 2025"],
    "11": ["Служебная записка"],
    "12": ["ТК_ст70"],
    "13": ["152ФЗ_ст9"],
    "14": ["ТК_ст59"],
    "15": ["А-08-2024"],
    "16": ["П-17-2024"],
    "17": ["ПТ-05-2023"],
    "18": ["ред 2025"],
}


def evaluate_retrieval(questions_path: Path) -> None:
    questions = load_questions(questions_path)

    raw_docs = load_raw_documents()
    docs = []
    for rd in raw_docs:
        clean_text, removed = sanitize_text(rd.text, rd.path)
        doc = build_document_metadata(rd.path, clean_text, removed)
        docs.append(doc)

    graph = build_doc_graph(docs)
    resolver = EntityResolver(docs, graph)

    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d))

    bm25 = BM25Index().fit(chunks)
    doc_map = {d.doc_id: d for d in docs}

    recall_at_5 = 0
    recall_at_10 = 0
    valid_gold_count = 0
    confusion_errors = 0

    print("=" * 85)
    print(f"{'QID':4} | {'Target doc(s)':20} | {'Recall@5':8} | {'Recall@10':9} | {'Top-3 documents found'}")
    print("-" * 85)

    for qid, q_text in questions:
        expected = GOLD_RETRIEVAL.get(qid, [])
        is_oob = len(expected) == 0

        # Query analysis (heuristic mode without client)
        _, analysis = analyze_query(q_text, client=None)

        ranked_chunks, low_conf = retrieve_candidates(
            analysis=analysis,
            chunks=chunks,
            docs=docs,
            graph=graph,
            bm25=bm25,
            resolver=resolver,
            dense_scores_fn=None,
            client=None,
        )

        seen_docs: list[str] = []
        for c, _ in ranked_chunks:
            p = doc_map[c.doc_id].path
            if p not in seen_docs:
                seen_docs.append(p)

        top_docs_5 = seen_docs[:5]
        top_docs_10 = seen_docs[:10]

        # Confusion check for Q9 and Q17
        if qid in ("9", "17"):
            top3 = [doc_map[c.doc_id].path for c, _ in ranked_chunks[:3]]
            if any("П-17" in p for p in top3):
                confusion_errors += 1
                print(f"[CONFUSION ERROR] Q{qid}: Found П-17-2024 in top-3 for Alpha-Trading query!")

        if is_oob:
            r5_str = "N/A"
            r10_str = "N/A"
        else:
            valid_gold_count += 1
            # Check if all expected docs are in top_docs
            hit_5 = all(any(exp in p for p in top_docs_5) for exp in expected)
            hit_10 = all(any(exp in p for p in top_docs_10) for exp in expected)
            if hit_5:
                recall_at_5 += 1
            if hit_10:
                recall_at_10 += 1
            r5_str = "YES" if hit_5 else "NO"
            r10_str = "YES" if hit_10 else "NO"

        exp_str = ", ".join(expected) if expected else "(вне базы)"
        top3_names = ", ".join([doc_map[c.doc_id].path.split("__")[-1][:22] for c, _ in ranked_chunks[:3]])
        print(f"{qid:4} | {exp_str:20} | {r5_str:8} | {r10_str:9} | {top3_names}")

    print("=" * 85)
    r5_pct = (recall_at_5 / valid_gold_count) * 100 if valid_gold_count else 0
    r10_pct = (recall_at_10 / valid_gold_count) * 100 if valid_gold_count else 0
    print(f"Total valid queries evaluated: {valid_gold_count}")
    print(f"Recall@5:  {recall_at_5}/{valid_gold_count} ({r5_pct:.1f}%)")
    print(f"Recall@10: {recall_at_10}/{valid_gold_count} ({r10_pct:.1f}%)")
    print(f"Confusion errors (Alpha-Trade vs Alpha-Trading in top-3): {confusion_errors}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="demo_questions.csv")
    args = parser.parse_args()
    evaluate_retrieval(Path(args.questions))
