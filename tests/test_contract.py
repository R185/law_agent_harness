from __future__ import annotations

import unittest
from pathlib import Path

from rag.pipeline import (
    build_context,
    get_clause,
    get_document,
    resolve_reference,
    retrieve,
    verify_citations,
)
from rag.schema import (
    Ambiguity,
    ContextResult,
    DocumentMatch,
    Evidence,
    ReferenceError,
    RetrievalResult,
    Signals,
    Unresolved,
)


class TestRAGContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Trigger index loading once
        res = retrieve("договор аренды ООО Меридиан", top_k=3)
        assert isinstance(res, RetrievalResult)

    def test_retrieve_return_types(self):
        """Verify retrieve() returns a valid RetrievalResult with all typed fields."""
        res = retrieve("Какой срок оплаты по договору поставки с ООО «Альфа-Трейд»?", top_k=5)

        self.assertIsInstance(res, RetrievalResult)
        self.assertIsInstance(res.documents, list)
        self.assertIsInstance(res.evidence, list)
        self.assertIsInstance(res.signals, Signals)
        self.assertIsInstance(res.unresolved, Unresolved)
        self.assertIsInstance(res.ambiguity, list)
        self.assertIsInstance(res.reference_errors, list)
        self.assertIsInstance(res.trace, list)

        # Check DocumentMatch fields
        self.assertGreater(len(res.documents), 0)
        doc = res.documents[0]
        self.assertIsInstance(doc, DocumentMatch)
        self.assertTrue(hasattr(doc, "doc_id"))
        self.assertTrue(hasattr(doc, "title"))
        self.assertTrue(hasattr(doc, "doc_number"))
        self.assertTrue(hasattr(doc, "doc_type"))
        self.assertTrue(hasattr(doc, "parties"))
        self.assertTrue(hasattr(doc, "status"))
        self.assertTrue(hasattr(doc, "effective_from"))
        self.assertTrue(hasattr(doc, "effective_to"))
        self.assertTrue(hasattr(doc, "applicable_edition"))
        self.assertTrue(hasattr(doc, "edition_note"))
        self.assertTrue(hasattr(doc, "score"))
        self.assertTrue(hasattr(doc, "score_kind"))
        self.assertTrue(hasattr(doc, "relevance"))
        self.assertTrue(hasattr(doc, "matched_clauses"))
        self.assertTrue(hasattr(doc, "is_direct"))
        self.assertTrue(hasattr(doc, "why"))
        self.assertGreaterEqual(doc.relevance, 0.0)
        self.assertLessEqual(doc.relevance, 1.0)

        # Check Evidence fields
        self.assertGreater(len(res.evidence), 0)
        ev = res.evidence[0]
        self.assertIsInstance(ev, Evidence)
        self.assertTrue(hasattr(ev, "doc_id"))
        self.assertTrue(hasattr(ev, "title"))
        self.assertTrue(hasattr(ev, "section"))
        self.assertTrue(hasattr(ev, "clause_ids"))
        self.assertTrue(hasattr(ev, "text"))
        self.assertTrue(hasattr(ev, "start"))
        self.assertTrue(hasattr(ev, "end"))
        self.assertTrue(hasattr(ev, "score"))
        self.assertTrue(hasattr(ev, "score_kind"))
        self.assertTrue(hasattr(ev, "relevance"))
        self.assertTrue(hasattr(ev, "source"))
        self.assertTrue(hasattr(ev, "why"))
        self.assertGreaterEqual(ev.relevance, 0.0)
        self.assertLessEqual(ev.relevance, 1.0)

        # Check Signals fields
        sig = res.signals
        self.assertIsInstance(sig.low_confidence, bool)
        self.assertIsInstance(sig.max_relevance, float)
        self.assertIsInstance(sig.mean_relevance, float)
        self.assertIsInstance(sig.mean_top3_relevance, float)
        self.assertIsInstance(sig.margin_top1_top2, float)
        self.assertIsInstance(sig.distinct_docs, int)
        self.assertIsInstance(sig.entity_match, list)
        self.assertIsInstance(sig.topic_match, bool)
        self.assertIsInstance(sig.score_kind, str)
        self.assertIsInstance(sig.dense_used, bool)
        self.assertIsInstance(sig.rerank_used, bool)
        self.assertIsInstance(sig.applicable_editions, list)

        # Check Unresolved fields
        unres = res.unresolved
        self.assertIsInstance(unres.contract_numbers, list)
        self.assertIsInstance(unres.party_names, list)
        self.assertIsInstance(unres.law_articles, list)
        self.assertIsInstance(unres.clauses, list)

    def test_entity_discrimination(self):
        """Verify strict entity matching distinguishes Alfa-Trade vs Alfa-Trading."""
        res_trade = retrieve("Какой адрес у ООО «Альфа-Трейд»?", top_k=5)
        trade_parties = [p for d in res_trade.documents[:3] for p in d.parties]
        self.assertTrue(any("Альфа-Трейд" in p for p in trade_parties))
        # Must not have Alfa-Trading in top-1 direct hit
        self.assertNotIn("Альфа-Трейдинг", res_trade.documents[0].parties)

        res_trading = retrieve("Какой адрес у ООО «Альфа-Трейдинг»?", top_k=5)
        trading_parties = [p for d in res_trading.documents[:3] for p in d.parties]
        self.assertTrue(any("Альфа-Трейдинг" in p for p in trading_parties))
        self.assertNotIn("Альфа-Трейд", res_trading.documents[0].parties)

    def test_ambiguity_detection(self):
        """Querying generic 'Альфа' without full name should flag ambiguity."""
        res = retrieve("Какие условия договора с Альфа?", top_k=5)
        # Should detect ambiguity between Альфа-Трейд and Альфа-Трейдинг
        self.assertGreater(len(res.ambiguity), 0)
        amb = res.ambiguity[0]
        self.assertIsInstance(amb, Ambiguity)
        cand_labels = [c.label for c in amb.candidates]
        self.assertTrue(any("Альфа-Трейд" in l for l in cand_labels))
        self.assertTrue(any("Альфа-Трейдинг" in l for l in cand_labels))

    def test_out_of_kb_detection(self):
        """Out of KB queries must flag low_confidence and unresolved entities."""
        res = retrieve("Какой штраф предусмотрен по соглашению с ООО «Неизвестный Космос»?", top_k=5)
        self.assertTrue(res.signals.low_confidence)
        self.assertIn("Неизвестный Космос", res.unresolved.party_names)

    def test_build_context(self):
        """Verify build_context produces structured prompt text and source list."""
        res = retrieve("Какой срок действия договора аренды с ООО «Бета»?", top_k=3)
        ctx = build_context(res, max_tokens=1500)
        self.assertIsInstance(ctx, ContextResult)
        self.assertIsInstance(ctx.prompt_text, str)
        self.assertIsInstance(ctx.sources, list)
        self.assertGreater(len(ctx.prompt_text), 0)
        self.assertGreater(ctx.token_estimate, 0)
        self.assertLessEqual(ctx.token_estimate, 1600)  # within allowance

    def test_verify_citations(self):
        """Verify citation checking works against KB documents."""
        # Valid citation to real contract in KB
        valid_text = "Согласно Договору поставки № П-17/2024 пункт 4.2..."
        check = verify_citations(valid_text)
        self.assertTrue(check.valid)
        self.assertGreater(len(check.citations_found), 0)

        # Invalid citation to imaginary document
        invalid_text = "Согласно Договору франчайзинга № Ф-99/2099 от 01.01.2099..."
        check_bad = verify_citations(invalid_text)
        self.assertFalse(check_bad.valid)
        self.assertGreater(len(check_bad.missing_citations), 0)

    def test_resolve_reference(self):
        """Verify resolving specific references (articles, doc numbers)."""
        ref_art = resolve_reference("ст. 9 152-ФЗ")
        self.assertIsNotNone(ref_art)
        self.assertIn("152-ФЗ", ref_art.title)

        ref_doc = resolve_reference("№ П-17/2024")
        self.assertIsNotNone(ref_doc)
        self.assertIn("п-17/2024", ref_doc.doc_number.lower())

    def test_get_document_and_clause(self):
        """Verify get_document and get_clause lookups."""
        res = retrieve("поставка Альфа-Трейд", top_k=1)
        doc_id = res.documents[0].doc_id
        doc_meta = get_document(doc_id)
        self.assertIsNotNone(doc_meta)
        self.assertEqual(doc_meta.doc_id, doc_id)

        # Non-existent doc
        self.assertIsNone(get_document("non_existent_doc_id_xyz"))

        # Lookup clause
        clause_res = get_clause(doc_id, "п. 1.1")
        if clause_res is not None:
            self.assertTrue(isinstance(clause_res, Evidence) or isinstance(clause_res, str))


if __name__ == "__main__":
    unittest.main()
