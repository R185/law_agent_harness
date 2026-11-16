from __future__ import annotations

from rag.metadata import DocGraph, Document
from rag.query import QueryAnalysis, _law_known_in_kb
from rag.retrieve import EntityResolver


def _law_doc() -> Document:
    return Document(
        doc_id="law164",
        path="",
        title="НК РФ ст. 164",
        category="law",
        doc_type="law_article",
        text="",
        meta={"law_code": "НК РФ", "article": "164"},
    )


def test_llm_law_is_accepted_only_when_kb_has_it() -> None:
    known = {"нк рф ст. 164", "ст. 164"}

    assert _law_known_in_kb("НК РФ ст. 164", known)
    assert not _law_known_in_kb("ТК РФ ст. 293", known)


def test_matched_law_becomes_target_document() -> None:
    doc = _law_doc()
    resolver = EntityResolver([doc], DocGraph())
    analysis = QueryAnalysis(
        standalone_question="Какая ставка НДС?",
        search_queries=["ставка НДС"],
        entities={"laws": ["НК РФ ст. 164"]},
    )

    targets, _, _, unresolved, _, trace = resolver.resolve(analysis)

    assert doc.doc_id in targets
    assert unresolved.law_articles == []
    assert any("law 'НК РФ ст. 164'" in line for line in trace)


def test_invented_law_stays_unresolved() -> None:
    doc = _law_doc()
    resolver = EntityResolver([doc], DocGraph())
    analysis = QueryAnalysis(
        standalone_question="Какая ставка?",
        search_queries=["ставка"],
        entities={"laws": ["ТК РФ ст. 293"]},
    )

    targets, _, _, unresolved, _, _ = resolver.resolve(analysis)

    assert doc.doc_id not in targets
    assert unresolved.law_articles == ["ТК РФ ст. 293"]
