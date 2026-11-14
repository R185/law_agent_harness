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
from rag.schema import (
    Ambiguity,
    DocumentMatch,
    Evidence,
    ReferenceError,
    RetrievalResult,
    Signals,
    Unresolved,
)
from rag.tools import build_context, verify_citations

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


def evaluate_retrieval(
    questions_path: Path, kb_path: Path | None = None, gold_path: Path | None = None
) -> None:
    questions = load_questions(questions_path)

    raw_docs = load_raw_documents(kb_path)
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
    contract_violations = 0

    print("=" * 95)
    print(f"{'QID':4} | {'Target doc(s)':20} | {'Recall@5':8} | {'Recall@10':9} | {'MaxRel':6} | {'Top-3 documents found'}")
    print("-" * 95)

    for qid, q_text in questions:
        expected = GOLD_RETRIEVAL.get(qid, [])
        is_oob = len(expected) == 0

        # Run query analysis dynamically using known parties
        _, analysis = analyze_query(
            q_text, client=None, known_parties=resolver.get_known_parties()
        )

        res: RetrievalResult = retrieve_candidates(
            analysis=analysis,
            chunks=chunks,
            docs=docs,
            graph=graph,
            bm25=bm25,
            resolver=resolver,
            dense_scores_fn=None,
            client=None,
            top_k=10,
        )

        # 1. Verify contract schema types and ranges
        if not isinstance(res, RetrievalResult):
            contract_violations += 1
        if not isinstance(res.signals, Signals):
            contract_violations += 1
        if not isinstance(res.unresolved, Unresolved):
            contract_violations += 1
        if not (0.0 <= res.signals.max_relevance <= 1.0):
            contract_violations += 1
        if not (0.0 <= res.signals.mean_relevance <= 1.0):
            contract_violations += 1

        for dm in res.documents:
            if not (0.0 <= dm.relevance <= 1.0):
                contract_violations += 1
            if not dm.score_kind:
                contract_violations += 1

        for ev in res.evidence:
            if not (0.0 <= ev.relevance <= 1.0):
                contract_violations += 1
            if ev.start < 0 or ev.end <= ev.start:
                contract_violations += 1

        # Check unique documents in ranked order
        seen_docs: list[str] = [doc_map[dm.doc_id].path for dm in res.documents]
        top_docs_5 = seen_docs[:5]
        top_docs_10 = seen_docs[:10]

        # Confusion check for Q9 and Q17
        if qid in ("9", "17"):
            top3 = seen_docs[:3]
            if any("П-17" in p for p in top3):
                confusion_errors += 1
                print(f"[CONFUSION ERROR] Q{qid}: Found П-17-2024 in top-3 for Alpha-Trading query!")

        if is_oob:
            r5_str = "N/A"
            r10_str = "N/A"
            # Out of KB question must produce low_confidence=True
            if not res.signals.low_confidence:
                print(f"[OOB WARNING] Q{qid}: expected low_confidence=True, got False")
        else:
            valid_gold_count += 1
            hit_5 = all(any(exp in p for p in top_docs_5) for exp in expected)
            hit_10 = all(any(exp in p for p in top_docs_10) for exp in expected)
            if hit_5:
                recall_at_5 += 1
            if hit_10:
                recall_at_10 += 1
            r5_str = "YES" if hit_5 else "NO"
            r10_str = "YES" if hit_10 else "NO"

        exp_str = ", ".join(expected) if expected else "(вне базы)"
        top3_names = ", ".join([doc_map[dm.doc_id].path.split("__")[-1][:20] for dm in res.documents[:3]])
        print(f"{qid:4} | {exp_str:20} | {r5_str:8} | {r10_str:9} | {res.signals.max_relevance:6.3f} | {top3_names}")

    print("=" * 95)
    r5_pct = (recall_at_5 / valid_gold_count) * 100 if valid_gold_count else 0
    r10_pct = (recall_at_10 / valid_gold_count) * 100 if valid_gold_count else 0
    print(f"Total valid queries evaluated: {valid_gold_count}")
    print(f"Recall@5:  {recall_at_5}/{valid_gold_count} ({r5_pct:.1f}%)")
    print(f"Recall@10: {recall_at_10}/{valid_gold_count} ({r10_pct:.1f}%)")
    print(f"Confusion errors (Alpha-Trade vs Alpha-Trading in top-3): {confusion_errors}")
    print(f"Contract schema violations: {contract_violations}")
    print("=" * 95)

    # Optional citation verification test if gold answers are provided
    if gold_path and gold_path.is_file():
        print("[EVAL] Verifying citations against gold answers...")
        with gold_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            gold_rows = list(reader)
        checked = 0
        valid_citations = 0
        for r in gold_rows:
            ans = r.get("answer", "")
            if ans and ans != config.FALLBACK:
                check = verify_citations(ans, docs=docs)
                checked += 1
                if check.valid:
                    valid_citations += 1
        print(f"[EVAL] Gold citations check: {valid_citations}/{checked} valid citations")
        print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of RAG retrieval quality and contract compliance")
    parser.add_argument("--questions", default="demo_questions.csv", help="Path to questions CSV")
    parser.add_argument("--kb", default=None, help="Path to KB zip or directory")
    parser.add_argument("--gold", default="demo_output.csv", help="Path to gold output CSV")
    args = parser.parse_args()

    q_p = Path(args.questions)
    kb_p = Path(args.kb) if args.kb else None
    g_p = Path(args.gold) if args.gold else None

    evaluate_retrieval(questions_path=q_p, kb_path=kb_p, gold_path=g_p)
