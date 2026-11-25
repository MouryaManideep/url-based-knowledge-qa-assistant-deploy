import unittest
from types import SimpleNamespace

from rag.retrieval import semantic_search


def make_doc(text, source="test"):
    return SimpleNamespace(page_content=text, metadata={"source": source})


class FakeVectorStore:
    def __init__(self, scored, mmr=None):
        self.scored = scored
        self.mmr = mmr if mmr is not None else [doc for doc, _ in scored]
        self.calls = []

    def similarity_search_with_score(self, query, k):
        self.calls.append(("similarity_search_with_score", query, k))
        return self.scored[:k]

    def max_marginal_relevance_search(self, query, k, fetch_k):
        self.calls.append(("max_marginal_relevance_search", query, k, fetch_k))
        return self.mmr[:k]


class SemanticRetrievalTests(unittest.TestCase):
    def test_converts_l2_distance_to_cosine_and_applies_threshold(self):
        strong = make_doc("strong match")
        weak = make_doc("weak match")
        store = FakeVectorStore([(strong, 0.2), (weak, 1.5)])
        results = semantic_search(store, "meaning query", k=2, min_similarity=0.3)
        self.assertEqual([doc.page_content for doc in results], ["strong match"])
        self.assertEqual(results[0].metadata["semantic_similarity"], 0.9)
        self.assertNotIn("semantic_similarity", strong.metadata)

    def test_mmr_returns_diverse_docs_but_keeps_threshold(self):
        first = make_doc("first")
        second = make_doc("second")
        third = make_doc("third")
        store = FakeVectorStore([(first, 0.3), (second, 1.8), (third, 0.5)], mmr=[third, second])
        results = semantic_search(store, "topic", k=2, min_similarity=0.2, search_type="mmr")
        self.assertEqual([doc.page_content for doc in results], ["third"])
        self.assertIn(("max_marginal_relevance_search", "topic", 2, 8), store.calls)

    def test_rejects_invalid_threshold_and_k(self):
        store = FakeVectorStore([])
        with self.assertRaises(ValueError):
            semantic_search(store, "q", min_similarity=1.2)
        with self.assertRaises(ValueError):
            semantic_search(store, "q", k=0)

    def test_empty_knowledge_base_returns_empty(self):
        self.assertEqual(semantic_search(FakeVectorStore([]), "q"), [])


if __name__ == "__main__":
    unittest.main()
