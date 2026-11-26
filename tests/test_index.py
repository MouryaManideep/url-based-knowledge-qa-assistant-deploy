import unittest

from rag.index import KnowledgeBase
from rag.types import Chunk, Source, SourceError
from tests.fakes import HashingEmbedder

DOCS = {
    "a": ["Llamas are domesticated South American camelids used as pack animals.",
          "Llama wool is soft and lanolin free."],
    "b": ["The Eiffel Tower is in Paris and was completed in 1889.",
          "Error code ZX-4012 means the turbine overheated."],
}


def build(**kwargs):
    kb, embedder = KnowledgeBase(**kwargs), HashingEmbedder()
    for sid, texts in DOCS.items():
        chunks = [Chunk(text=t, source_id=sid, title=f"Doc {sid}", location="Intro") for t in texts]
        kb.add(Source(id=sid, title=f"Doc {sid}", kind="text", origin=f"{sid}.txt"), chunks, embedder)
    return kb, embedder


class KnowledgeBaseTests(unittest.TestCase):
    def test_finds_relevant_passage(self):
        kb, emb = build()
        hits = kb.search(["When was the Eiffel Tower completed?"], emb, k=2, min_relevance=0.1)
        self.assertIn("1889", hits[0].chunk.text)
        self.assertTrue(0 <= hits[0].relevance <= 1)

    def test_keyword_match_survives_relevance_floor(self):
        kb, emb = build()
        hits = kb.search(["ZX-4012"], emb, k=3, min_relevance=0.99)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].keyword_match)
        self.assertIn("turbine", hits[0].chunk.text)

    def test_irrelevant_query_returns_nothing(self):
        kb, emb = build()
        self.assertEqual(kb.search(["quantum chromodynamics gluon"], emb, k=3, min_relevance=0.5), [])

    def test_remove_source_and_duplicates(self):
        kb, emb = build()
        with self.assertRaises(SourceError):
            kb.add(Source(id="a", title="dup", kind="text", origin="a"), [Chunk("x", "a", "dup")], emb)
        kb.remove("a")
        self.assertEqual(set(kb.sources), {"b"})
        self.assertTrue(all(c.source_id == "b" for c in kb.chunks))
        hits = kb.search(["llama wool"], emb, k=3, min_relevance=0.3)
        self.assertTrue(all(h.chunk.source_id == "b" for h in hits))
        kb.remove("b")
        self.assertTrue(kb.is_empty)
        self.assertEqual(kb.search(["anything"], emb), [])

    def test_session_limits(self):
        kb, emb = build(max_sources=2)
        with self.assertRaises(SourceError):
            kb.check_capacity()
        kb2 = KnowledgeBase(max_chunks=1)
        with self.assertRaises(SourceError):
            kb2.add(Source(id="z", title="z", kind="text", origin="z"), [Chunk("one", "z", "z"), Chunk("two", "z", "z")], emb)

    def test_sessions_are_isolated(self):
        first, emb = build()
        second = KnowledgeBase()
        self.assertTrue(second.is_empty)
        self.assertEqual(second.search(["llama"], emb), [])
        self.assertFalse(first.is_empty)

    def test_add_reports_embedding_progress(self):
        kb, emb, seen = KnowledgeBase(), HashingEmbedder(), []
        chunks = [Chunk(f"passage {i}", "big", "Big") for i in range(150)]
        kb.add(Source("big", "Big", "pdf", "big.pdf"), chunks, emb, progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(64, 150), (128, 150), (150, 150)])
        self.assertEqual(len(kb), 150)

    def test_diversify_returns_k_distinct(self):
        kb, emb = build()
        hits = kb.search(["llama"], emb, k=2, min_relevance=0.0, diversify=True)
        self.assertEqual(len(hits), 2)
        self.assertEqual(len({h.chunk.text for h in hits}), 2)


if __name__ == "__main__":
    unittest.main()
